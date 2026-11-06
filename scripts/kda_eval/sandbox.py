#!/usr/bin/env python3
"""bwrap sandbox in which every agent tool call for a CUDA-Hercules task runs.

The agent loop (LLM API calls) runs outside; each bash / file tool call is
executed in a fresh sandbox with this view (nothing else exists inside):

  /usr, /etc            read-only host system (compilers, CUDA toolkit, python3)
  /opt/cutlass/...      read-only: exactly the include dirs that
                        get_solution_include_dirs() hands to solution builds
  /opt/kh/...           read-only: in-sandbox helpers (submit client, fstool)
  /opt/kda/...          read-only: KDA skills (ncu-report-skill)
  /workspace            read-write task workspace (the only persistent state)
  /run/kh/judge.sock    the judge's unix socket
  /dev/nvidia<N>        only the assigned GPU (+ ctl/uvm nodes)

No network at all (empty netns, no proxy); the API key never enters. The host
home directory, KernelHell (task dirs, reference.cu, reference_sources beyond
the CUTLASS include dirs, results of other runs), conda envs, /tmp, /mnt and
other users' data are never mounted.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
KH_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SANDBOX_HOME = "/home/agent"
SANDBOX_WORKSPACE = "/workspace"
SANDBOX_SOCK_DIR = "/run/kh"

# Mirrors cuda_hercules.eval.get_solution_include_dirs(): solutions may
# #include these, so the agent may read them. Nothing else from
# reference_sources/ is exposed.
CUTLASS_INCLUDE_SUBDIRS = ["include", "tools/util/include", "examples/common"]

# Strings that only occur in CUDA-Hercules reference code / harnesses or in
# the upstream projects the references come from. Any hit inside the sandbox
# view (outside the allowed CUTLASS include dirs) is a leak.
LEAK_PATTERNS = [
    "Flash_fwd_params", "Flash_bwd_params", "flash_fwd_kernel", "flash_bwd_dq_dk_dv",
    "CollectiveMainloopFwdSm90", "CollectiveMainloopBwdSm90", "FLASH_NAMESPACE",
    "kittens::", "ThunderKittens",
    "deep_gemm", "DeepGEMM",
    "sageattn", "qk_int8_sv",
    "ln_fwd_kernels", "ln_bwd_kernels",
    "VkFFT",
    "KH_TEST_SOLUTION", "cuda_hercules", "CUDA-Hercules", "KernelHell",
    "HAS_CUDA_REFERENCE", "REFERENCE_EXTRA_INCLUDES",
]


def gpu_minor_number(index: int) -> int:
    """Map an nvidia-smi index to its /dev/nvidiaN minor number."""
    out = subprocess.run(["nvidia-smi", "-q", "-i", str(index)],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("Minor Number"):
            return int(line.split(":")[1])
    raise RuntimeError(f"no minor number for GPU {index}")


@dataclass
class SandboxSpec:
    workspace: str
    sock_dir: str
    gpu_index: int
    opt_kh: str                                   # host dir mounted at /opt/kh
    ro_binds: list = field(default_factory=list)  # extra (src, dst)
    env: dict = field(default_factory=dict)
    _minor: int = -1


def base_env() -> dict:
    return {
        "HOME": SANDBOX_HOME,
        "USER": "agent",
        "LANG": "C.UTF-8",
        "TERM": "dumb",
        "PATH": "/opt/kh/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_HOME": "/usr/local/cuda",
        "CUTLASS_INCLUDE_DIRS": ":".join(f"/opt/cutlass/{d}" for d in CUTLASS_INCLUDE_SUBDIRS),
        "PYTHONPATH": "/usr/local/cuda/nsight-compute-2025.3.1/extras/python",
        "TMPDIR": "/tmp",
        "KH_JUDGE_SOCKET": f"{SANDBOX_SOCK_DIR}/judge.sock",
    }


def build_bwrap_argv(spec: SandboxSpec, command: list[str]) -> list[str]:
    if spec._minor < 0:
        spec._minor = gpu_minor_number(spec.gpu_index)
    argv = [
        "bwrap",
        "--unshare-all", "--die-with-parent", "--new-session",
        "--hostname", "kda-sandbox",
        "--ro-bind", "/usr", "/usr",
        # System-wide pip packages (transformers/HF hub) are not needed and
        # mention DeepGEMM; hide them.
        "--tmpfs", "/usr/local/lib/python3.10/dist-packages",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib32", "/lib32",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/libx32", "/libx32",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", "/sys", "/sys",
        "--proc", "/proc",
        "--dev", "/dev",
        "--dev-bind", "/dev/nvidiactl", "/dev/nvidiactl",
        "--dev-bind", "/dev/nvidia-uvm", "/dev/nvidia-uvm",
        "--dev-bind", "/dev/nvidia-uvm-tools", "/dev/nvidia-uvm-tools",
        "--dev-bind", f"/dev/nvidia{spec._minor}", f"/dev/nvidia{spec._minor}",
        "--tmpfs", "/tmp",
        "--tmpfs", "/var/tmp",
        "--tmpfs", "/run",
        "--tmpfs", "/home",
        "--dir", SANDBOX_HOME,
        "--bind", spec.sock_dir, SANDBOX_SOCK_DIR,
        "--bind", spec.workspace, SANDBOX_WORKSPACE,
        "--ro-bind", spec.opt_kh, "/opt/kh",
    ]
    cutlass = os.path.join(KH_ROOT, "reference_sources", "cutlass")
    for sub in CUTLASS_INCLUDE_SUBDIRS:
        src = os.path.join(cutlass, sub)
        if os.path.isdir(src):
            argv += ["--ro-bind", src, f"/opt/cutlass/{sub}"]
    for src, dst in spec.ro_binds:
        argv += ["--ro-bind", src, dst]
    argv += ["--clearenv"]
    env = base_env()
    env.update(spec.env)
    for k, v in env.items():
        argv += ["--setenv", k, v]
    argv += ["--chdir", SANDBOX_WORKSPACE, "--"] + command
    return argv


def run_in_sandbox(spec: SandboxSpec, command: list[str], stdin: str = "",
                   timeout_s: float = 600) -> tuple[int, str, str, bool]:
    """Run one command in a fresh sandbox. Returns (rc, stdout, stderr, timed_out)."""
    proc = subprocess.Popen(build_bwrap_argv(spec, command), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        out, err = proc.communicate(stdin.encode(), timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        timed_out = True
    return (proc.returncode, out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"), timed_out)


def install_opt_kh(dest: str) -> None:
    """Populate the /opt/kh tree: submit client and file-tool helper."""
    os.makedirs(os.path.join(dest, "bin"), exist_ok=True)
    submit = os.path.join(dest, "bin", "submit")
    shutil.copy2(os.path.join(HERE, "submit.py"), submit)
    os.chmod(submit, 0o755)
    shutil.copy2(os.path.join(HERE, "fstool.py"), os.path.join(dest, "fstool.py"))


# ---------------------------------------------------------------------------
# Isolation probe: `python scripts/kda_eval/sandbox.py probe --gpu 2 [--scan]`
# ---------------------------------------------------------------------------

PROBE_SCRIPT = r"""
set +e
echo "== whoami / hostname"; id; hostname
echo "== host paths (expect all absent)"
for p in /home/li004074 /home/li004074/KernelHell /mnt /opt/miniconda3 /data; do test -e $p && echo "VISIBLE $p" || echo "absent $p"; done
echo "== /home /opt"; ls -la /home; find /opt -maxdepth 2 | sort
echo "== network (expect failure)"
ip -brief addr 2>/dev/null || cat /proc/net/dev
curl -sS -m 5 -o /dev/null -w '%{http_code}\n' https://api.anthropic.com 2>&1 | tail -1
curl -sS -m 5 -o /dev/null -w '%{http_code}\n' https://github.com 2>&1 | tail -1
echo "== GPUs"; nvidia-smi -L
cat > /tmp/dq.cu <<'EOF'
#include <cstdio>
__global__ void k(float* x){x[threadIdx.x]=threadIdx.x;}
int main(){int n=0;cudaGetDeviceCount(&n);printf("cuda devices=%d\n",n);
for(int i=0;i<n;i++){cudaDeviceProp p;cudaGetDeviceProperties(&p,i);printf("  %d %s sm_%d%d\n",i,p.name,p.major,p.minor);}
float*d;cudaMalloc(&d,1024);k<<<1,32>>>(d);printf("sync=%d\n",(int)cudaDeviceSynchronize());return 0;}
EOF
nvcc -o /tmp/dq /tmp/dq.cu && /tmp/dq
echo "== ncu can read counters"; ncu --metrics sm__cycles_elapsed.avg /tmp/dq 2>&1 | grep -E 'sm__cycles|ERR|==PROF==' | head -4
echo "== judge socket dir"; ls -la /run/kh
"""


def _leak_scan_cmd() -> str:
    pats = "\n".join(LEAK_PATTERNS)
    return (
        "cat > /tmp/leak_patterns.txt <<'EOF'\n" + pats + "\nEOF\n"
        "echo '== leak scan (paths with reference-only strings; /opt/cutlass excluded)'\n"
        "grep -rIlF -f /tmp/leak_patterns.txt /usr /etc /home /workspace /run /tmp "
        "$(ls -d /opt/* | grep -v '^/opt/cutlass$') "
        "2>/dev/null | grep -v -e '^/tmp/leak_patterns.txt$' -e '^/opt/kh/' | head -200\n"
        "echo '== leak scan done'\n"
    )


def probe(gpu: int, scan: bool, ro_binds: list) -> int:
    tmp = tempfile.mkdtemp(prefix="kda_probe_")
    dirs = {k: os.path.join(tmp, k) for k in ("ws", "sock", "opt_kh")}
    for d in dirs.values():
        os.makedirs(d)
    install_opt_kh(dirs["opt_kh"])
    try:
        spec = SandboxSpec(workspace=dirs["ws"], sock_dir=dirs["sock"], gpu_index=gpu,
                           opt_kh=dirs["opt_kh"], ro_binds=ro_binds)
        script = PROBE_SCRIPT + (_leak_scan_cmd() if scan else "")
        return subprocess.run(build_bwrap_argv(spec, ["/bin/bash", "-c", script])).returncode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="Run isolation checks inside a throwaway sandbox")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--scan", action="store_true", help="Also grep the whole sandbox view for reference-only strings")
    args = ap.parse_args()
    if args.cmd == "probe":
        ncu_skill = os.path.join(KH_ROOT, "kda", "skills", "ncu-report-skill")
        sys.exit(probe(args.gpu, args.scan, [(ncu_skill, "/opt/kda/skills/ncu-report-skill")]))


if __name__ == "__main__":
    main()
