"""Pydantic models for task.yaml schema (v1)."""

from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    min_sm: int = 80
    tested_sms: list[int] = Field(default_factory=list)
    required_features: list[str] = Field(default_factory=list)


class RunnerConfig(BaseModel):
    backend: str  # "class1_make" | "class2_defpy" | "class3_app" | "class4_challenge"
    workdir: str = ""
    timeout_sec: int = 180
    env: dict[str, str] = Field(default_factory=dict)
    solution_file: str = ""  # single solution file (Class 1/2/4)
    solution_files: list[str] = Field(default_factory=list)  # multiple files (Class 3)


class SuccessConfig(BaseModel):
    exit_code: int = 0
    stdout_regex: str = ""


class BuildConfig(BaseModel):
    cmd: str = "make test"
    clean_cmd: str = "make clean"


class ExecuteConfig(BaseModel):
    cmd: str = "./test"
    success: SuccessConfig = Field(default_factory=SuccessConfig)


class CorrectnessConfig(BaseModel):
    mode: str = "stdout_or_exit"  # "stdout_or_exit" | "tensor_compare"
    tolerances: dict[str, float] = Field(
        default_factory=lambda: {"atol": 1e-2, "rtol": 1e-2}
    )


class PerformanceConfig(BaseModel):
    enabled: bool = True
    parser: dict[str, str] = Field(default_factory=dict)
    metric: str = "speedup_ref_over_sol"


class SourceConfig(BaseModel):
    repo: str = ""
    files: list[str] = Field(default_factory=list)
    commit: str = ""
    license: str = ""


class AntiCheatConfig(BaseModel):
    blocked_patterns: list[str] = Field(default_factory=list)
    required_patterns: list[str] = Field(default_factory=list)

    @classmethod
    def __get_validators__(cls):
        yield cls._coerce_none

    from pydantic import model_validator

    @model_validator(mode='before')
    @classmethod
    def _coerce_none_lists(cls, values):
        if isinstance(values, dict):
            for k in ('blocked_patterns', 'required_patterns'):
                if k in values and values[k] is None:
                    values[k] = []
        return values


class TaskConfig(BaseModel):
    schema_version: int = 1
    task_id: str
    name: str = ""
    task_class: int = 0
    domain: str = "ml"
    tags: list[str] = Field(default_factory=list)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    runner: RunnerConfig
    build: BuildConfig = Field(default_factory=BuildConfig)
    execute: ExecuteConfig = Field(default_factory=ExecuteConfig)
    correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    anti_cheat: AntiCheatConfig = Field(default_factory=AntiCheatConfig)


def load_task_config(yaml_path: str) -> TaskConfig:
    """Load and validate a task.yaml file."""
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**data)
