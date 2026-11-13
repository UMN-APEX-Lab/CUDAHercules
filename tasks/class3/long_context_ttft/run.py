#!/usr/bin/env python3
"""Long-context TTFT harness: serve Qwen3.8-27B with vLLM under the fixed serving configuration,
measure the time to first token of 64K / 128K / 256K-token prompts, then check output quality
against the BF16 dense model.

    python run.py candidate --plugin <dir containing hercules_plugin/> --output cand.json
    python run.py reference --output ref.json      # the reference implementation (hidden from the model)
    python run.py stock --output stock.json        # unmodified vLLM under the same configuration
    python run.py make-quality-reference           # BF16 dense outputs/scores (run once per data set)

Environment: HERCULES_DATA_DIR (data root; prompts.json and quality_reference.json live in
$HERCULES_DATA_DIR/long_context_ttft), HERCULES_VLLM_PYTHON (Python of the vLLM 0.30.0
environment; default: this interpreter), HERCULES_LCT_MODEL (default Qwen/Qwen3.8-27B, resolved
through HF_HOME; HERCULES_LCT_REVISION pins a revision), CUDA_VISIBLE_DEVICES (one GPU),
HERCULES_LCT_GPU_MEM_UTIL (lower --gpu-memory-utilization on a GPU shared with other processes; it
only changes the KV cache capacity and must be the same for the configurations compared).
"""
import argparse
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
LENGTHS = (65536, 131072, 260096)
GEN_TOKENS = 384          # greedy generation for the needle check
CONT_TOKENS = 256         # teacher-forced continuation tokens scored per prompt
TOP1_MIN = 0.965          # quality gates (mean over the three lengths)
DNLL_MAX = 0.01

# Fixed for every configuration; a solution cannot change them.
SERVE_ARGS = [
    "--dtype", "bfloat16", "--quantization", "fp8", "--kv-cache-dtype", "fp8",
    "--max-model-len", "262144", "--max-num-batched-tokens", "8192", "--max-num-seqs", "4",
    "--enable-chunked-prefill", "--no-enable-prefix-caching", "--language-model-only",
    "--gpu-memory-utilization", "0.88", "--seed", "20261112",
]
# The quality reference: BF16 weights, BF16 KV cache, unmodified vLLM.
DENSE_ARGS = [
    "--dtype", "bfloat16", "--max-model-len", "262144", "--max-num-batched-tokens", "8192",
    "--max-num-seqs", "4", "--enable-chunked-prefill", "--no-enable-prefix-caching", "--language-model-only",
    "--gpu-memory-utilization", "0.88", "--seed", "20261112",
]
# Activation of the reference implementation (reference/README.md).
REFERENCE_ENV = {"LCP_SPARSE_ATTN": "1", "LCP_SPARSE_KERNEL": "fp8", "LCP_SPARSE_TAU": "0.95"}


def data_dir():
    root = os.environ.get("HERCULES_DATA_DIR")
    if not root:
        sys.exit("HERCULES_DATA_DIR is not set (see prepare_data.sh)")
    return os.path.join(root, "long_context_ttft")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, payload, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ttft(port, ids):
    """Seconds from sending the request to the first streamed token."""
    payload = {"model": "model", "prompt": ids, "max_tokens": 1, "temperature": 0.0, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if line.startswith(b"data:") and line != b"data: [DONE]":
                ev = json.loads(line[5:])
                if ev.get("choices"):
                    t = time.perf_counter() - t0
                    for _ in r:  # drain
                        pass
                    return t
    raise RuntimeError("no token streamed")


class Server:
    def __init__(self, args_list, env_extra, pythonpath, log_path):
        py = os.environ.get("HERCULES_VLLM_PYTHON") or sys.executable
        model = os.environ.get("HERCULES_LCT_MODEL", "Qwen/Qwen3.8-27B")
        self.port = free_port()
        env = os.environ.copy()
        env.update(env_extra)
        if pythonpath:
            env["PYTHONPATH"] = os.pathsep.join(pythonpath + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        util = os.environ.get("HERCULES_LCT_GPU_MEM_UTIL")  # shared GPUs only; same value for every configuration
        if util:
            args_list = [util if i > 0 and args_list[i - 1] == "--gpu-memory-utilization" else a
                         for i, a in enumerate(args_list)]
        cmd = [py, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", "model",
               "--port", str(self.port), "--host", "127.0.0.1", *args_list]
        if os.environ.get("HERCULES_LCT_REVISION"):
            cmd += ["--revision", os.environ["HERCULES_LCT_REVISION"]]
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.log_path = log_path

    def wait_ready(self, timeout=3600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(5)
        return False

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.log.close()

    def tail(self, n=4000):
        try:
            return open(self.log_path, errors="replace").read()[-n:]
        except OSError:
            return ""


def measure_ttft(port, prompts):
    """Per length: one warm-up request (variant 2), then two timed requests (variants 0 and 1)."""
    out = {}
    for L in LENGTHS:
        by_idx = {p["index"]: p for p in prompts if p["length"] == L}
        ttft(port, by_idx[2]["prompt_token_ids"])
        runs = [ttft(port, by_idx[i]["prompt_token_ids"]) for i in (0, 1)]
        out[str(L)] = {"runs_s": runs, "min_s": min(runs)}
        print(f"  TTFT {L}: {', '.join(f'{r:.3f}' for r in runs)} s", flush=True)
    return out


def generate(port, ids):
    r = post(port, {"model": "model", "prompt": ids, "max_tokens": GEN_TOKENS, "temperature": 0.0,
                    "return_token_ids": True})
    c = r["choices"][0]
    return c["text"], c.get("token_ids") or []


def score(port, ids, cont):
    """Teacher-forced: NLL and top-1 agreement of the continuation tokens, plus their logprobs."""
    full = list(ids) + list(cont)
    r = post(port, {"model": "model", "prompt": full, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1})
    plp = r["choices"][0]["prompt_logprobs"]
    lps, top1 = [], []
    for i in range(len(ids), len(full)):
        e = plp[i][str(full[i])]
        lps.append(e["logprob"])
        top1.append(1.0 if e.get("rank") == 1 else 0.0)
    return lps, sum(top1) / len(top1)


def quality(port, prompts, qref):
    rows, ok = {}, True
    for L in LENGTHS:
        p = next(p for p in prompts if p["length"] == L and p["index"] == 0)
        ref = qref[str(L)]
        text, _ = generate(port, p["prompt_token_ids"])
        hit = p["needle_code"] in text
        lps, top1 = score(port, p["prompt_token_ids"], ref["continuation"])
        nll = -sum(lps) / len(lps)
        dnll = nll - ref["nll"]
        rows[str(L)] = {"needle_hit": hit, "nll": nll, "delta_nll": dnll, "top1_agreement": top1,
                        "text_head": text[:200]}
        print(f"  quality {L}: needle {'ok' if hit else 'MISSING'}, top-1 {top1:.3f}, dNLL {dnll:+.4f}", flush=True)
    mean_top1 = sum(r["top1_agreement"] for r in rows.values()) / len(rows)
    mean_dnll = sum(r["delta_nll"] for r in rows.values()) / len(rows)
    ok = all(r["needle_hit"] for r in rows.values()) and mean_top1 >= TOP1_MIN and mean_dnll <= DNLL_MAX
    return {"per_length": rows, "mean_top1_agreement": mean_top1, "mean_delta_nll": mean_dnll,
            "gates": {"needle": "all lengths", "top1_min": TOP1_MIN, "delta_nll_max": DNLL_MAX}, "passed": ok}


def run_config(name, plugin=None, skip_quality=False, log_dir=None):
    prompts = json.load(open(os.path.join(data_dir(), "prompts.json")))
    log_dir = log_dir or tempfile.mkdtemp(prefix="lct_")
    pythonpath, env = [], {}
    if name == "candidate":
        pythonpath = [os.path.join(TASK_DIR, "harness"), plugin]
        env = {"HERCULES_PLUGIN": "1"}
    elif name == "reference":
        ref = os.path.join(TASK_DIR, "reference")
        pythonpath = [ref, os.path.join(ref, "shim")]
        env = dict(REFERENCE_ENV)
    server = Server(DENSE_ARGS if name == "dense" else SERVE_ARGS, env, pythonpath,
                    os.path.join(log_dir, f"server_{name}.log"))
    result = {"config": name, "server_log": server.log_path}
    try:
        t0 = time.time()
        if not server.wait_ready():
            result.update(started=False, error="server did not start:\n" + server.tail())
            if name == "dense":
                server.stop()
            return result
        result.update(started=True, startup_s=time.time() - t0)
        print(f"[{name}] server up in {result['startup_s']:.0f} s", flush=True)
        if name == "dense":
            return result | {"server": server}
        result["ttft"] = measure_ttft(server.port, prompts)
        if not skip_quality:
            qref = json.load(open(os.path.join(data_dir(), "quality_reference.json")))
            result["quality"] = quality(server.port, prompts, qref)
        return result
    except Exception as e:  # report the failure and the server log
        result.update(error=f"{type(e).__name__}: {e}\n" + server.tail())
        return result
    finally:
        if name != "dense":
            server.stop()


def make_quality_reference():
    """BF16 dense model: greedy continuation and its own teacher-forced NLL per length (variant 0)."""
    prompts = json.load(open(os.path.join(data_dir(), "prompts.json")))
    res = run_config("dense")
    if not res.get("started"):
        sys.exit(res.get("error", "dense server failed"))
    server = res["server"]
    out = {}
    try:
        for L in LENGTHS:
            p = next(p for p in prompts if p["length"] == L and p["index"] == 0)
            text, toks = generate(server.port, p["prompt_token_ids"])
            cont = toks[:CONT_TOKENS]
            lps, top1 = score(server.port, p["prompt_token_ids"], cont)
            out[str(L)] = {"prompt_sha256": p["prompt_sha256"], "continuation": cont, "nll": -sum(lps) / len(lps),
                           "top1_agreement": top1, "needle_hit": p["needle_code"] in text, "text": text}
            print(f"  dense {L}: needle {'ok' if out[str(L)]['needle_hit'] else 'MISSING'}, NLL {out[str(L)]['nll']:.4f}",
                  flush=True)
    finally:
        server.stop()
    path = os.path.join(data_dir(), "quality_reference.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", choices=["candidate", "reference", "stock", "make-quality-reference"])
    ap.add_argument("--plugin", help="directory that contains the hercules_plugin package (candidate)")
    ap.add_argument("--output", help="write the result JSON here")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--log-dir")
    a = ap.parse_args()
    if a.config == "make-quality-reference":
        make_quality_reference()
        return
    if a.config == "candidate" and not (a.plugin and os.path.isdir(os.path.join(a.plugin, "hercules_plugin"))):
        sys.exit("candidate needs --plugin <dir> containing hercules_plugin/")
    res = run_config(a.config, os.path.abspath(a.plugin) if a.plugin else None, a.skip_quality, a.log_dir)
    if a.output:
        json.dump(res, open(a.output, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "server"}, indent=1))
    ok = res.get("started") and "ttft" in res and (a.skip_quality or res.get("quality", {}).get("passed"))
    print("Passed" if ok else "Failed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
