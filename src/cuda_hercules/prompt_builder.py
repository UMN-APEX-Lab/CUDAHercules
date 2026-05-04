"""
Prompt builder for CUDA-Hercules.

Builds LLM prompts from description.txt for all task classes.
"""

import os


PYTORCH_EXTENSION_TASKS = {
    "class3/mgg_agnn/general",
    "class3/mgg_agnn/hopper",
    "class3/mgg_agnn/blackwell",
    "class3/tcgnn_gcn/general",
}

PROMPT_DISABLE_TEMPLATE_TASKS = {
    "class3/exachem_ccsd_t/general",
    "class3/exachem_ccsd_t/hopper",
    "class3/exachem_ccsd_t/blackwell",
}

PROMPT_CONTEXT_FILES = {
    "class3/mgg_agnn/general": ["wrapper.cpp", "run.py"],
    "class3/mgg_agnn/hopper": ["wrapper.cpp", "run.py"],
    "class3/mgg_agnn/blackwell": ["wrapper.cpp", "run.py"],
    "class3/tcgnn_gcn/general": ["wrapper.cpp"],
    "class3/icicle_zk/general": [
        "run.py",
        "build_contract.md",
        "custom_backend_api.h",
        "backend_template.cu",
    ],
    "class3/icicle_zk/hopper": [
        "run.py",
        "build_contract.md",
        "custom_backend_api.h",
        "backend_template.cu",
    ],
    "class3/icicle_zk/blackwell": [
        "run.py",
        "build_contract.md",
        "custom_backend_api.h",
        "backend_template.cu",
    ],
    "class3/liberator/general": [
        "build_contract.md",
        "custom_backend_api.h",
        "backend_template.cu",
    ],
    "class3/liberator/hopper": [
        "build_contract.md",
        "custom_backend_api.h",
        "backend_template.cu",
    ],
    "class3/liberator/blackwell": [
        "build_contract.md",
        "custom_backend_api.h",
        "backend_template.cu",
    ],
    "class3/llmc/general": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
    "class3/llmc/hopper": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
    "class3/llmc/blackwell": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
    "class3/gpumd/general": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
    "class3/gpumd/hopper": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
    "class3/gpumd/blackwell": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
    "class3/parafrost_sat/general": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
    "class3/parafrost_sat/hopper": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
    "class3/parafrost_sat/blackwell": [
        "build_contract.md",
        "project_manifest_example.json",
    ],
}


def build_prompt(task_dir: str, task_config=None, description_dir: str = None) -> str:
    """
    Build a prompt for the LLM from a task's description.txt.

    Reads description.txt as the primary prompt content.
    Appends solution requirements and output format instructions.

    Args:
        task_dir: Path to the task directory (for solution templates, etc.).
        task_config: Optional TaskConfig for additional context.
        description_dir: Optional override directory for description.txt.
            Used when description.txt lives in an arch subdirectory
            separate from the shared task workdir.

    Returns:
        Complete prompt string.
    """
    # Read description.txt (from description_dir if provided, else task_dir)
    desc_dir = description_dir or task_dir
    desc_path = os.path.join(desc_dir, "description.txt")
    if not os.path.isfile(desc_path):
        raise FileNotFoundError(f"No description.txt found in {desc_dir}")

    with open(desc_path) as f:
        description = f.read().strip()

    # Determine solution file name(s)
    solution_info = ""
    task_id = getattr(task_config, "task_id", "") if task_config else ""
    if task_config:
        if task_config.runner.solution_files:
            files = ", ".join(task_config.runner.solution_files)
            if any(entry.endswith("/") for entry in task_config.runner.solution_files):
                solution_info = (
                    f"\nYou are generating a self-contained project under: {files}"
                )
            else:
                solution_info = f"\nYou are optimizing these files: {files}"
        elif task_config.runner.solution_file:
            solution_info = f"\nYou are writing: {task_config.runner.solution_file}"

    # Read solution template if it exists
    template = ""
    if task_id not in PROMPT_DISABLE_TEMPLATE_TASKS:
        for candidate in ["solution.cu", "solution.h"]:
            tpl_path = os.path.join(task_dir, candidate)
            if os.path.isfile(tpl_path):
                with open(tpl_path) as f:
                    content = f.read().strip()
                if len(content) > 50:  # non-trivial template
                    template = f"\n\n## Solution Template\n\nFill in this template:\n```cuda\n{content}\n```"
                break

    context_sections = []
    for relpath in PROMPT_CONTEXT_FILES.get(task_id, []):
        ctx_path = os.path.join(task_dir, relpath)
        if not os.path.isfile(ctx_path):
            continue
        with open(ctx_path) as f:
            content = f.read().strip()
        if content:
            language = "cpp"
            if relpath.endswith(".py"):
                language = "python"
            elif relpath.endswith(".sh"):
                language = "bash"
            elif relpath.endswith(".md"):
                language = "markdown"
            elif relpath.endswith(".txt"):
                language = "text"
            elif relpath.endswith(".cu"):
                language = "cuda"
            context_sections.append(
                f"\n\n## Existing File: {relpath}\n\n"
                f"This file already exists in the project and you may modify it if needed:\n"
                f"```{language}\n{content}\n```"
            )

    requirements = [
        "- Write pure CUDA/C++ code",
        "- Include all necessary headers (#include <cuda_runtime.h>, etc.)",
        "- All kernel functions must use `__global__` and launch via `<<<>>>`",
        "- Do not use cuFFT/cuBLAS/cuDNN/CUTLASS (or any other high-level CUDA library)",
        "- Optimize for maximum performance on the target GPU",
    ]

    if task_id not in PYTORCH_EXTENSION_TASKS:
        requirements.insert(2, "- The entry function MUST be declared with `extern \"C\"` linkage")

    output_format = "Return the complete .cu file content in a ```cuda code block."
    if task_config and task_config.runner.solution_files and any(
        entry.endswith("/") for entry in task_config.runner.solution_files
    ):
        output_format = (
            "This is a directory-based task. You may first be asked for a JSON project manifest, "
            "and then for individual file contents. For each request, return only the requested "
            "artifact and nothing else."
        )
    elif task_config and task_config.runner.solution_files and len(task_config.runner.solution_files) > 1:
        output_format = (
            "Return only the complete content of the requested file in a fenced code block. "
            "Match the file's language (for example ```cuda or ```cpp) when appropriate."
        )

    # Build prompt
    parts = [
        description,
        solution_info,
        template,
        "".join(context_sections),
        "",
        "## Requirements",
        *requirements,
        "",
        "## Output Format",
        output_format,
    ]

    return "\n".join(parts)


def build_prompt_from_yaml(yaml_path: str) -> str:
    """
    Build a prompt from a task.yaml path.

    Convenience wrapper that loads the config and finds the task directory.
    """
    from .task_schema import load_task_config
    from .utils import get_project_root

    config = load_task_config(yaml_path)
    yaml_dir = os.path.dirname(yaml_path)
    task_dir = os.path.join(get_project_root(), config.runner.workdir) if config.runner.workdir else yaml_dir
    return build_prompt(task_dir, config, description_dir=yaml_dir)
