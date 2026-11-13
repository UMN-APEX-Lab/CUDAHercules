# Imported at interpreter start in every vLLM process (the harness puts this directory on
# PYTHONPATH): loads the solution's plugin package before vLLM initializes.
import os

if os.environ.get("HERCULES_PLUGIN") == "1":
    import hercules_plugin  # noqa: F401
