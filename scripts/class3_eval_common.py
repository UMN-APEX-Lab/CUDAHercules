#!/usr/bin/env python3
"""
Shared helper for per-task Class 3 LLM evaluation scripts.

Each public entrypoint in `scripts/eval_class3_<task>.py` targets exactly one
Class 3 task and calls into this module for the common generation/evaluation
flow.
"""

import argparse
import json
import os
import posixpath
import subprocess
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuda_hercules.llm_api import query_server
from cuda_hercules.prompt_builder import build_prompt
from cuda_hercules.result import TaskResult, TaskStatus
from cuda_hercules.runner import discover_tasks, get_gpu_sm, parse_filters
from cuda_hercules.score import compute_scores, format_score
from cuda_hercules.static_checker import validate_cuda_solution
from cuda_hercules.task_schema import TaskConfig
from cuda_hercules.utils import (
    extract_cuda_code,
    extract_fenced_code,
    extract_json_payload,
    get_project_root,
)


ARCH_CHOICES = ("general", "hopper", "blackwell")


def build_parser(task_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"CUDA-Hercules Class 3 evaluation for {task_name}"
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-base", default="", help="API base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument(
        "--backend",
        default="openai",
        choices=["openai", "vertex"],
        help="LLM backend: openai (default) or vertex (Vertex AI)",
    )
    parser.add_argument("--vertex-region", default="global", help="Vertex AI region")
    parser.add_argument(
        "--vertex-project", default="neu-research", help="Vertex AI project ID"
    )
    parser.add_argument(
        "--arch", default="general", choices=ARCH_CHOICES, help="Task architecture variant"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--reasoning", action="store_true", help="Enable reasoning/thinking mode"
    )
    parser.add_argument(
        "--reasoning-effort", default="low", choices=["low", "medium", "high"]
    )
    parser.add_argument(
        "--num-samples", type=int, default=1, help="Samples per task (pass@N)"
    )
    parser.add_argument(
        "--sample-timeout-sec",
        type=int,
        default=1800,
        help="Per-sample timeout for task-local evaluation.",
    )
    parser.add_argument("--run-name", default="", help="Run name")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--quiet", action="store_true")
    return parser


def add_solution_target_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--solution-manifest",
        default="",
        help="Text or JSON manifest listing generated files for directory-based tasks",
    )
    parser.add_argument(
        "--solution-target",
        action="append",
        default=[],
        help="Relative file path to generate inside the task solution directory; repeat as needed",
    )
    return parser


def load_solution_targets(manifest_path: str = "", inline_targets: list[str] | None = None) -> list[str]:
    targets: list[str] = []

    if manifest_path:
        with open(manifest_path) as f:
            raw = f.read().strip()
        if raw:
            if manifest_path.endswith(".json"):
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    payload = payload.get("targets", [])
                if not isinstance(payload, list):
                    raise ValueError(f"Invalid JSON manifest format: {manifest_path}")
                targets.extend(str(item).strip() for item in payload if str(item).strip())
            else:
                for line in raw.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        targets.append(line)

    if inline_targets:
        targets.extend(t.strip() for t in inline_targets if t.strip())

    deduped: list[str] = []
    seen = set()
    for target in targets:
        if target not in seen:
            deduped.append(target)
            seen.add(target)
    return deduped


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        parts = [p.strip() for p in visible.split(",") if p.strip()]
        return len(parts)
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _task_min_gpus(config: TaskConfig) -> int:
    yaml_dir = getattr(config, "_yaml_dir", "")
    yaml_path = os.path.join(yaml_dir, "task.yaml")
    try:
        with open(yaml_path) as f:
            raw = json.loads(json.dumps(__import__("yaml").safe_load(f)))
        return int(raw.get("hardware", {}).get("min_gpus", 1))
    except Exception:
        return 1


def _load_task_config(task_name: str, arch: str) -> TaskConfig:
    root = get_project_root()
    task_id = f"class3/{task_name}/{arch}"
    tasks = {
        t.task_id: t
        for t in discover_tasks(root, parse_filters(["backend=class3_app"]))
    }
    if task_id not in tasks:
        raise ValueError(f"Task not found: {task_id}")
    return tasks[task_id]


def _solution_targets_from_config(config: TaskConfig) -> tuple[list[str], list[str]]:
    entries = []
    if config.runner.solution_files:
        entries.extend(config.runner.solution_files)
    elif config.runner.solution_file:
        entries.append(config.runner.solution_file)
    file_targets = [entry for entry in entries if entry and not entry.endswith("/")]
    dir_targets = [entry for entry in entries if entry and entry.endswith("/")]
    return file_targets, dir_targets


def _fence_lang_for_file(path: str) -> str:
    name = os.path.basename(path)
    if name == "CMakeLists.txt" or path.endswith(".cmake"):
        return "cmake"
    if path.endswith(".sh"):
        return "bash"
    if path.endswith(".py"):
        return "python"
    if path.endswith(".cu") or path.endswith(".cuh"):
        return "cuda"
    if path.endswith(".cpp") or path.endswith(".cc") or path.endswith(".cxx"):
        return "cpp"
    if path.endswith(".h") or path.endswith(".hpp"):
        return "cpp"
    if path.endswith(".json"):
        return "json"
    if path.endswith(".md"):
        return "markdown"
    if name == "Makefile":
        return "makefile"
    return "text"


def _extract_artifact_for_target(path: str, response: str) -> str:
    langs = []
    name = os.path.basename(path)
    if path.endswith(".json"):
        langs = ["json"]
    elif path.endswith(".sh"):
        langs = ["bash", "sh"]
    elif path.endswith(".py"):
        langs = ["python"]
    elif path.endswith(".cu") or path.endswith(".cuh"):
        return extract_cuda_code(response)
    elif path.endswith(".cpp") or path.endswith(".cc") or path.endswith(".cxx"):
        langs = ["cpp", "c++", "c"]
    elif path.endswith(".h") or path.endswith(".hpp"):
        langs = ["cpp", "c++", "c"]
    elif name == "CMakeLists.txt" or path.endswith(".cmake"):
        langs = ["cmake"]
    elif name == "Makefile":
        langs = ["makefile", "text"]
    elif path.endswith(".md"):
        langs = ["markdown", "text"]
    return extract_fenced_code(response, langs)


def _is_safe_relative_path(path: str) -> bool:
    if not path or path.startswith("/"):
        return False
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized in ("", ".", ".."):
        return False
    if normalized.startswith("../") or "/../" in normalized:
        return False
    return normalized == path.replace("\\", "/")


def _parse_directory_manifest(payload, directory_target: str) -> tuple[dict | None, str]:
    if not isinstance(payload, dict):
        return None, "manifest must be a JSON object"

    build_script = str(payload.get("build_script", "")).strip()
    run_script = str(payload.get("run_script", "")).strip()
    files = payload.get("files", [])
    cuda_sources = payload.get("cuda_sources", [])
    ops = payload.get("ops", {})

    if not build_script or not run_script:
        return None, "manifest must define build_script and run_script"
    if not isinstance(files, list) or not files:
        return None, "manifest must define a non-empty files list"
    if not isinstance(cuda_sources, list) or not cuda_sources:
        return None, "manifest must define a non-empty cuda_sources list"
    if not isinstance(ops, dict):
        return None, "manifest ops must be an object"

    clean_files: list[str] = []
    seen = set()
    for entry in files:
        rel = str(entry).strip()
        if not _is_safe_relative_path(rel):
            return None, f"unsafe manifest file path: {rel!r}"
        if rel not in seen:
            clean_files.append(rel)
            seen.add(rel)

    if len(clean_files) > 24:
        return None, "manifest lists too many files (max 24)"

    if build_script not in seen or run_script not in seen:
        return None, "build_script and run_script must appear in files"

    clean_cuda_sources: list[str] = []
    for entry in cuda_sources:
        rel = str(entry).strip()
        if not _is_safe_relative_path(rel):
            return None, f"unsafe cuda_sources path: {rel!r}"
        if rel not in seen:
            return None, f"cuda_sources entry not present in files: {rel}"
        clean_cuda_sources.append(rel)

    if not any(rel.endswith((".cu", ".cuh", ".cpp", ".cc", ".cxx")) for rel in clean_cuda_sources):
        return None, "cuda_sources must include at least one CUDA/C++ source file"

    clean_ops: dict[str, str] = {}
    for key, value in ops.items():
        op_name = str(key).strip()
        rel = str(value).strip()
        if not op_name:
            return None, "manifest ops must use non-empty keys"
        if not _is_safe_relative_path(rel):
            return None, f"unsafe ops path: {rel!r}"
        if rel not in seen:
            return None, f"ops entry not present in files: {rel}"
        clean_ops[op_name] = rel

    return {
        "project_root": directory_target.rstrip("/"),
        "build_script": build_script,
        "run_script": run_script,
        "files": clean_files,
        "cuda_sources": clean_cuda_sources,
        "ops": clean_ops,
    }, ""


def _resolve_generation_targets(
    config: TaskConfig,
    explicit_targets: list[str] | None,
) -> tuple[list[str], bool, str]:
    file_targets, dir_targets = _solution_targets_from_config(config)
    directory_target = dir_targets[0] if dir_targets else ""

    if explicit_targets:
        return explicit_targets, bool(dir_targets), directory_target

    if file_targets:
        return file_targets, bool(dir_targets), directory_target

    if dir_targets:
        return [], True, directory_target

    raise ValueError(f"No solution targets configured for {config.task_id}")


def _write_sample_files(sample_dir: str, sample_codes: dict[str, str]) -> None:
    for filename, code in sample_codes.items():
        filepath = os.path.join(sample_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(code or "")

        basename_alias = os.path.join(sample_dir, os.path.basename(filename))
        if os.path.abspath(basename_alias) != os.path.abspath(filepath):
            with open(basename_alias, "w") as f:
                f.write(code or "")


def _static_check_sample(config: TaskConfig, sample_codes: dict[str, str]) -> tuple[bool, str]:
    combined_entries: list[tuple[str, str]] = []

    manifest_entry = next(
        (name for name in sample_codes if os.path.basename(name) == "project_manifest.json"),
        "",
    )
    if manifest_entry:
        manifest = extract_json_payload(sample_codes.get(manifest_entry, ""))
        if not isinstance(manifest, dict):
            return False, "invalid project_manifest.json"
        prefix = os.path.dirname(manifest_entry)
        cuda_sources = manifest.get("cuda_sources", [])
        if not isinstance(cuda_sources, list) or not cuda_sources:
            return False, "project_manifest.json must declare cuda_sources"
        missing = []
        for rel in cuda_sources:
            rel = str(rel).strip()
            path = os.path.join(prefix, rel) if prefix else rel
            code = sample_codes.get(path, "")
            if not code.strip():
                missing.append(rel)
            else:
                combined_entries.append((path, code))
        if missing:
            return False, f"missing declared cuda_sources: {', '.join(missing[:6])}"

    if not combined_entries:
        for filename, code in sample_codes.items():
            if filename.endswith((".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp")):
                combined_entries.append((filename, code or ""))

    combined_code = "\n\n".join(
        f"// FILE: {filename}\n{code}" for filename, code in combined_entries
    ).strip()
    if not combined_code:
        return False, "empty"

    allow_torch_cpp_api = config.task_id in {
        "class3/mgg_agnn/general",
        "class3/mgg_agnn/hopper",
        "class3/mgg_agnn/blackwell",
        "class3/tcgnn_gcn/general",
    }

    check = validate_cuda_solution(
        combined_code,
        blocked_patterns=config.anti_cheat.blocked_patterns or [],
        required_patterns=config.anti_cheat.required_patterns or [],
        allow_torch_cpp_api=allow_torch_cpp_api,
    )
    if not check.valid:
        return False, "; ".join(check.errors[:6])
    return True, ""


def evaluate_class3_sample_in_subprocess(
    task_id: str,
    solution_path: str,
    timeout_sec: int,
) -> dict:
    root = get_project_root()
    solution_path = os.path.abspath(solution_path)
    script = f"""
import json
import os
import sys
sys.path.insert(0, os.path.join({root!r}, 'src'))
from cuda_hercules.runner import discover_tasks, run_task, parse_filters

tasks = {{t.task_id: t for t in discover_tasks({root!r}, parse_filters(['backend=class3_app']))}}
config = tasks[{task_id!r}]
tr = run_task(config, {solution_path!r}, measure_perf=True, verbose=False)
print(json.dumps({{
    "compiled": tr.compiled,
    "correct": tr.correct,
    "speedup": tr.speedup,
    "latency_ms": tr.latency_mean_ms,
    "error": tr.error_msg or "",
}}))
"""
    # start_new_session + killpg so that grandchildren (compiled binaries the
    # task script launches) die with the parent on timeout — otherwise they
    # outlive us and keep GPU memory locked.
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        import os as _os
        import signal as _signal
        try:
            _os.killpg(proc.pid, _signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "latency_ms": 0.0,
            "error": f"timeout after {timeout_sec}s",
        }

    output_lines = [
        line.strip()
        for line in (stdout or "").splitlines()
        if line.strip().startswith("{")
    ]
    if rc != 0 and not output_lines:
        error = ((stderr or "") or (stdout or ""))[-1000:]
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "latency_ms": 0.0,
            "error": f"subprocess error: {error}",
        }

    if not output_lines:
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "latency_ms": 0.0,
            "error": "subprocess produced no JSON result",
        }

    try:
        return json.loads(output_lines[-1])
    except json.JSONDecodeError:
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "latency_ms": 0.0,
            "error": f"invalid subprocess json: {output_lines[-1][:200]}",
        }


def _generate_samples(
    prompt: str,
    solution_targets: list[str],
    directory_target: str,
    num_samples: int,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    backend: str,
    vertex_region: str,
    vertex_project: str,
    is_reasoning_model: bool,
    reasoning_effort: str,
    verbose: bool,
) -> tuple[list[dict[str, str]], list[dict], float, str]:
    generation_log: list[dict] = []
    all_sample_codes: list[dict[str, str]] = []
    t0 = time.time()

    if directory_target:
        project_root = directory_target.rstrip("/")
        if verbose:
            print(
                f"  Directory-based task ({num_samples} samples, project root {project_root}/)...",
                flush=True,
            )
        for sample_idx in range(num_samples):
            sample_codes: dict[str, str] = {}
            manifest_prompt = (
                prompt
                + "\n\nYou are planning a self-contained project under the directory "
                + f"`{project_root}/`.\n"
                + "Return ONLY a JSON object for `project_manifest.json` with this schema:\n"
                + "{\n"
                + '  "build_script": "build.sh",\n'
                + '  "run_script": "run.sh",\n'
                + '  "files": ["build.sh", "run.sh", "src/main.cu"],\n'
                + '  "cuda_sources": ["src/main.cu"],\n'
                + '  "ops": {\n'
                + '    "required_component_name": "src/main.cu"\n'
                + "  }\n"
                + "}\n"
                + "Constraints:\n"
                + "- Use relative paths only\n"
                + "- Include build.sh and run.sh\n"
                + "- `cuda_sources` must list the files that contain your custom CUDA kernels\n"
                + "- `ops` must include every required component named by this task's contract\n"
                + "- Do not include project_manifest.json in the files list; the benchmark writes it for you\n"
            )
            if verbose:
                print(f"    s{sample_idx} [manifest] project_manifest.json ...", end=" ", flush=True)
            manifest_t0 = time.time()
            try:
                manifest_resp = query_server(
                    prompt=manifest_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_base=api_base,
                    api_key=api_key,
                    backend=backend,
                    vertex_region=vertex_region,
                    vertex_project=vertex_project,
                    is_reasoning_model=is_reasoning_model,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                elapsed = time.time() - manifest_t0
                generation_log.append(
                    {
                        "sample": sample_idx,
                        "file": os.path.join(project_root, "project_manifest.json"),
                        "elapsed_s": elapsed,
                        "error": f"manifest generation failed: {e}",
                    }
                )
                if verbose:
                    print(f"ERROR ({elapsed:.0f}s)", flush=True)
                all_sample_codes.append(sample_codes)
                continue

            manifest_payload = extract_json_payload(manifest_resp or "")
            manifest, manifest_error = _parse_directory_manifest(manifest_payload, directory_target)
            elapsed = time.time() - manifest_t0
            if not manifest:
                generation_log.append(
                    {
                        "sample": sample_idx,
                        "file": os.path.join(project_root, "project_manifest.json"),
                        "elapsed_s": elapsed,
                        "error": manifest_error or "invalid manifest",
                    }
                )
                if verbose:
                    print(f"ERROR ({elapsed:.0f}s)", flush=True)
                all_sample_codes.append(sample_codes)
                continue

            manifest_rel = os.path.join(project_root, "project_manifest.json")
            manifest_text = json.dumps(manifest, indent=2)
            sample_codes[manifest_rel] = manifest_text
            generation_log.append(
                {
                    "sample": sample_idx,
                    "file": manifest_rel,
                    "elapsed_s": elapsed,
                    "lines": len(manifest_text.splitlines()),
                    "chars": len(manifest_text),
                    "empty": False,
                }
            )
            if verbose:
                print(f"{len(manifest['files'])} files ({elapsed:.0f}s)", flush=True)

            for file_idx, relpath in enumerate(manifest["files"], 1):
                generated_context = (
                    "\n\n## Project Manifest\n\n```json\n"
                    + manifest_text
                    + "\n```"
                )
                if sample_codes:
                    sections = []
                    for prev_file, prev_code in sample_codes.items():
                        lang = _fence_lang_for_file(prev_file)
                        sections.append(
                            f"\n\n## Already Generated File: {prev_file}\n\n"
                            f"```{lang}\n{prev_code or ''}\n```"
                        )
                    generated_context += (
                        "\n\nUse these previously generated project files as fixed context and keep the new file "
                        "consistent with them."
                        + "".join(sections)
                    )
                file_prompt = (
                    prompt
                    + generated_context
                    + f"\n\nYou are writing the file `{relpath}` inside the project root `{project_root}/`.\n"
                    + "Return ONLY the complete content of this file."
                )
                if verbose:
                    print(
                        f"    s{sample_idx} [{file_idx}/{len(manifest['files'])}] {relpath} ...",
                        end=" ",
                        flush=True,
                    )
                file_t0 = time.time()
                try:
                    resp = query_server(
                        prompt=file_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_base=api_base,
                        api_key=api_key,
                        backend=backend,
                        vertex_region=vertex_region,
                        vertex_project=vertex_project,
                        is_reasoning_model=is_reasoning_model,
                        reasoning_effort=reasoning_effort,
                    )
                    code = _extract_artifact_for_target(relpath, resp or "")
                    elapsed = time.time() - file_t0
                    full_path = os.path.join(project_root, relpath)
                    sample_codes[full_path] = code
                    generation_log.append(
                        {
                            "sample": sample_idx,
                            "file": full_path,
                            "elapsed_s": elapsed,
                            "lines": len(code.splitlines()) if code else 0,
                            "chars": len(code) if code else 0,
                            "empty": not bool(code.strip()),
                        }
                    )
                    if verbose:
                        print(f"{len(code.splitlines()) if code else 0}L ({elapsed:.0f}s)", flush=True)
                except Exception as e:
                    elapsed = time.time() - file_t0
                    full_path = os.path.join(project_root, relpath)
                    sample_codes[full_path] = ""
                    generation_log.append(
                        {
                            "sample": sample_idx,
                            "file": full_path,
                            "elapsed_s": elapsed,
                            "error": str(e),
                        }
                    )
                    if verbose:
                        print(f"ERROR ({elapsed:.0f}s)", flush=True)
            all_sample_codes.append(sample_codes)
        return all_sample_codes, generation_log, time.time() - t0, ""

    is_multi_file = len(solution_targets) > 1
    if not is_multi_file:
        sol_name = solution_targets[0]
        if verbose:
            print(f"  Generating {num_samples} samples...", end=" ", flush=True)
        try:
            responses = query_server(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=api_base,
                api_key=api_key,
                num_completions=num_samples,
                backend=backend,
                vertex_region=vertex_region,
                vertex_project=vertex_project,
                is_reasoning_model=is_reasoning_model,
                reasoning_effort=reasoning_effort,
            )
            if isinstance(responses, str):
                responses = [responses]
        except Exception as e:
            return [], generation_log, time.time() - t0, f"Generation failed: {e}"

        for sample_idx, resp in enumerate(responses):
            code = extract_cuda_code(resp) if resp else ""
            all_sample_codes.append({sol_name: code})
            generation_log.append(
                {
                    "sample": sample_idx,
                    "file": sol_name,
                    "lines": len(code.splitlines()) if code else 0,
                    "chars": len(code) if code else 0,
                    "empty": not bool(code.strip()),
                }
            )
        if verbose:
            print(flush=True)
    else:
        if verbose:
            print(
                f"  Multi-file task ({len(solution_targets)} files x {num_samples} samples)...",
                flush=True,
            )
        for sample_idx in range(num_samples):
            sample_codes: dict[str, str] = {}
            for file_idx, sol_file in enumerate(solution_targets, 1):
                generated_context = ""
                if sample_codes:
                    sections = []
                    for prev_file, prev_code in sample_codes.items():
                        lang = _fence_lang_for_file(prev_file)
                        sections.append(
                            f"\n\n## Already Generated File: {prev_file}\n\n"
                            f"```{lang}\n{prev_code or ''}\n```"
                        )
                    generated_context = (
                        "\n\nUse these previously generated files as fixed context and keep the new file "
                        "consistent with them."
                        + "".join(sections)
                    )
                file_prompt = (
                    prompt
                    + generated_context
                    + f"\n\nYou are writing the file: `{sol_file}`\n"
                    + "Return ONLY the complete content of this file."
                )
                if verbose:
                    print(
                        f"    s{sample_idx} [{file_idx}/{len(solution_targets)}] {sol_file} ...",
                        end=" ",
                        flush=True,
                    )
                file_t0 = time.time()
                try:
                    resp = query_server(
                        prompt=file_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_base=api_base,
                        api_key=api_key,
                        backend=backend,
                        vertex_region=vertex_region,
                        vertex_project=vertex_project,
                        is_reasoning_model=is_reasoning_model,
                        reasoning_effort=reasoning_effort,
                    )
                    code = _extract_artifact_for_target(sol_file, resp or "")
                    elapsed = time.time() - file_t0
                    sample_codes[sol_file] = code
                    generation_log.append(
                        {
                            "sample": sample_idx,
                            "file": sol_file,
                            "elapsed_s": elapsed,
                            "lines": len(code.splitlines()) if code else 0,
                            "chars": len(code) if code else 0,
                            "empty": not bool(code.strip()),
                        }
                    )
                    if verbose:
                        print(f"{len(code.splitlines()) if code else 0}L ({elapsed:.0f}s)", flush=True)
                except Exception as e:
                    elapsed = time.time() - file_t0
                    sample_codes[sol_file] = ""
                    generation_log.append(
                        {
                            "sample": sample_idx,
                            "file": sol_file,
                            "elapsed_s": elapsed,
                            "error": str(e),
                        }
                    )
                    if verbose:
                        print(f"ERROR ({elapsed:.0f}s)", flush=True)
            all_sample_codes.append(sample_codes)

    return all_sample_codes, generation_log, time.time() - t0, ""


def run_single_task_eval(
    task_name: str,
    args: argparse.Namespace,
    explicit_solution_targets: list[str] | None = None,
) -> int:
    config = _load_task_config(task_name, args.arch)
    root = get_project_root()
    gpu_sm = get_gpu_sm()
    visible_gpus = _visible_gpu_count()
    min_gpus = _task_min_gpus(config)

    run_name = args.run_name or (
        f"{args.model.replace('/', '_')}_class3_{task_name}_{args.arch}_pass{args.num_samples}_{int(time.time())}"
    )
    run_dir = os.path.join(args.output, run_name)
    os.makedirs(run_dir, exist_ok=True)

    task_out = os.path.join(run_dir, config.task_id.replace("/", "_"))
    os.makedirs(task_out, exist_ok=True)

    print(f"CUDA-Hercules Class 3 Task Evaluation")
    print(f"  Task:        {config.task_id}")
    print(f"  Model:       {args.model}")
    print(f"  API:         {args.api_base or 'default'}")
    print(f"  Samples:     {args.num_samples}")
    print(f"  GPU:         SM {gpu_sm}")
    print(f"  Visible GPU: {visible_gpus}")
    print(f"  Output:      {run_dir}")
    print()

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(
            {
                "task_id": config.task_id,
                "model": args.model,
                "api_base": args.api_base,
                "backend": args.backend,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "num_samples": args.num_samples,
                "gpu_sm": gpu_sm,
                "visible_gpus": visible_gpus,
                "min_required_gpus": min_gpus,
                "explicit_solution_targets": explicit_solution_targets or [],
            },
            f,
            indent=2,
        )

    result = {
        "task_id": config.task_id,
        "task_class": config.task_class,
        "domain": config.domain,
        "model": args.model,
        "num_samples": args.num_samples,
        "num_compiled": 0,
        "num_correct": 0,
        "pass_at_n": False,
        "best_speedup": 0.0,
        "best_latency_ms": 0.0,
        "generation_time_s": 0.0,
        "error": "",
        "generation_log": [],
        "sample_details": [],
    }

    yaml_dir = getattr(config, "_yaml_dir", os.path.join(root, config.runner.workdir))
    prompt = build_prompt(os.path.join(root, config.runner.workdir), config, description_dir=yaml_dir)
    with open(os.path.join(task_out, "prompt.txt"), "w") as f:
        f.write(prompt)

    status = TaskStatus.FAIL

    if gpu_sm < config.hardware.min_sm:
        result["error"] = f"Current GPU SM {gpu_sm} is below required SM {config.hardware.min_sm}"
        status = TaskStatus.SKIP_ARCH
    elif visible_gpus > 0 and min_gpus > visible_gpus:
        result["error"] = f"Task needs {min_gpus} visible GPUs, but only {visible_gpus} are available"
        status = TaskStatus.SKIP_ARCH
    else:
        try:
            solution_targets, force_directory_eval, directory_target = _resolve_generation_targets(
                config, explicit_solution_targets
            )
        except ValueError as e:
            result["error"] = str(e)
            force_directory_eval = False
            directory_target = ""
        else:
            all_sample_codes, generation_log, gen_time, gen_error = _generate_samples(
                prompt=prompt,
                solution_targets=solution_targets,
                directory_target=directory_target,
                num_samples=args.num_samples,
                model=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                backend=args.backend,
                vertex_region=args.vertex_region,
                vertex_project=args.vertex_project,
                is_reasoning_model=args.reasoning,
                reasoning_effort=args.reasoning_effort,
                verbose=not args.quiet,
            )
            result["generation_log"] = generation_log
            result["generation_time_s"] = gen_time
            if gen_error:
                result["error"] = gen_error
            else:
                is_multi_file = len(solution_targets) > 1
                for sample_idx, sample_codes in enumerate(all_sample_codes):
                    total_lines = sum(
                        len((code or "").splitlines()) for code in sample_codes.values()
                    )
                    non_empty = sum(
                        1 for code in sample_codes.values() if code and len(code.strip()) > 10
                    )
                    sample_dir = os.path.abspath(os.path.join(task_out, f"sample_{sample_idx}"))
                    os.makedirs(sample_dir, exist_ok=True)
                    _write_sample_files(sample_dir, sample_codes)

                    if non_empty == 0:
                        result["sample_details"].append(
                            {
                                "sample": sample_idx,
                                "compiled": False,
                                "correct": False,
                                "error": "empty",
                            }
                        )
                        if not args.quiet:
                            print(f"    s{sample_idx}: empty")
                        continue

                    ok, static_error = _static_check_sample(config, sample_codes)
                    if not ok:
                        result["sample_details"].append(
                            {
                                "sample": sample_idx,
                                "compiled": False,
                                "correct": False,
                                "error": f"static check failed: {static_error}",
                            }
                        )
                        if not args.quiet:
                            print(f"    s{sample_idx}: BLOCKED", flush=True)
                        continue

                    if not args.quiet:
                        print(
                            f"    s{sample_idx}: {total_lines}L ({len(sample_codes)} files)",
                            end=" -> ",
                            flush=True,
                        )

                    if directory_target:
                        sol_path = os.path.join(sample_dir, directory_target.rstrip("/"))
                    elif force_directory_eval or is_multi_file:
                        sol_path = sample_dir
                    else:
                        sol_path = os.path.join(sample_dir, solution_targets[0])
                    sample_result = evaluate_class3_sample_in_subprocess(
                        config.task_id,
                        sol_path,
                        args.sample_timeout_sec,
                    )

                    compiled = sample_result["compiled"]
                    correct = sample_result["correct"]
                    speedup = sample_result["speedup"]
                    latency_ms = sample_result["latency_ms"]
                    error = sample_result.get("error", "")

                    result["sample_details"].append(
                        {
                            "sample": sample_idx,
                            "compiled": compiled,
                            "correct": correct,
                            "speedup": speedup,
                            "latency_ms": latency_ms,
                            "error": error,
                        }
                    )

                    if compiled:
                        result["num_compiled"] += 1
                    if correct:
                        result["num_correct"] += 1
                        result["pass_at_n"] = True
                        if speedup > result["best_speedup"]:
                            result["best_speedup"] = speedup
                            result["best_latency_ms"] = latency_ms

                    if not args.quiet:
                        sample_status = "PASS" if correct else ("COMPILED" if compiled else "FAIL")
                        extra = f" {speedup:.2f}x" if speedup > 0 else ""
                        print(f"{sample_status}{extra}", flush=True)

                status = TaskStatus.PASS if result["pass_at_n"] else TaskStatus.FAIL

    with open(os.path.join(task_out, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    task_result = TaskResult(
        task_id=result["task_id"],
        compiled=result["num_compiled"] > 0,
        correct=result["pass_at_n"],
        speedup=result["best_speedup"],
        latency_mean_ms=result["best_latency_ms"],
        domain=result["domain"],
        level=result["task_class"],
        status=status,
        error_msg=result["error"],
    )

    score = compute_scores([task_result])
    report = format_score(score)

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump([result], f, indent=2, default=str)
    with open(os.path.join(run_dir, "score.json"), "w") as f:
        json.dump(asdict(score), f, indent=2, default=str)
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(report)
        f.write(
            f"\n\nPass@{args.num_samples}: "
            f"{1 if result['pass_at_n'] else 0}/1 ({1.0 if result['pass_at_n'] else 0.0:.1%})"
        )

    print(f"\n{report}")
    print(
        f"\nPass@{args.num_samples}: "
        f"{1 if result['pass_at_n'] else 0}/1 ({1.0 if result['pass_at_n'] else 0.0:.1%})"
    )
    print(f"\nResults saved to: {run_dir}")
    return 0


def main_for_task(task_name: str) -> int:
    parser = build_parser(task_name)
    args = parser.parse_args()
    return run_single_task_eval(task_name, args)
