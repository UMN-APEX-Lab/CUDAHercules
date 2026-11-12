"""Naive solution files must never reach the model (prompt or agent tools)."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cuda_hercules.prompt_builder import build_prompt  # noqa: E402
from cuda_hercules.task_schema import load_task_config  # noqa: E402

CLASS1_TASK = os.path.join(ROOT, "tasks", "class1", "general", "00_basic_gemm")
CLASS4_TASK = os.path.join(ROOT, "tasks", "class4", "flash_attn4_fwd")


def _body(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


class PromptLeakTest(unittest.TestCase):
    def _check(self, task_dir: str, sol: str):
        config = load_task_config(os.path.join(task_dir, "task.yaml"))
        prompt = build_prompt(task_dir, config)
        body = _body(os.path.join(task_dir, sol))
        self.assertGreater(len(body), 50)
        self.assertNotIn(body, prompt)
        self.assertNotIn("## Solution Template", prompt)
        self.assertNotIn("Fill in this template", prompt)
        # distinctive body lines (not shared with the description) must be absent
        desc = _body(os.path.join(task_dir, "description.txt"))
        lines = [l.strip() for l in body.splitlines() if len(l.strip()) > 30]
        leaked = [l for l in lines if l in prompt and l not in desc]
        self.assertEqual(leaked, [])

    def test_class1_prompt_has_no_naive_solution(self):
        self._check(CLASS1_TASK, "solution.h")

    def test_class4_prompt_has_no_skeleton(self):
        self._check(CLASS4_TASK, "solution.cu")


class AgentToolLeakTest(unittest.TestCase):
    def setUp(self):
        try:
            from cuda_hercules import agent
        except ImportError as e:  # optional [agent] extras
            self.skipTest(str(e))
        self.agent = agent
        agent._state.clear()
        agent._state["task_dir"] = CLASS1_TASK
        agent._state["config"] = load_task_config(os.path.join(CLASS1_TASK, "task.yaml"))

    def tearDown(self):
        self.agent._state.clear()

    def test_list_hides_solution(self):
        listing = self.agent.list_task_files.invoke({})
        self.assertIn("description.txt", listing)
        self.assertNotIn("solution.h", listing)

    def test_read_refuses_solution(self):
        for name in ("solution.h", "./solution.h", "sub/../solution.h"):
            out = self.agent.read_file.invoke({"filename": name})
            self.assertTrue(out.startswith("Error"), name)
            self.assertNotIn("naive_sgemm_kernel", out)
        self.assertIn("SgemmNN", self.agent.read_file.invoke({"filename": "description.txt"}))


class KdaWorkspaceTest(unittest.TestCase):
    def test_make_workspace_has_no_solution(self):
        import importlib.util
        import tempfile
        path = os.path.join(ROOT, "scripts", "eval_kda.py")
        spec = importlib.util.spec_from_file_location("eval_kda_mod", path)
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            self.skipTest(f"eval_kda not importable: {e}")
        config = load_task_config(os.path.join(CLASS1_TASK, "task.yaml"))
        with tempfile.TemporaryDirectory() as ws:
            mod.make_workspace(config, ws, "TASK")
            self.assertFalse(os.path.exists(os.path.join(ws, "solution.h")))


if __name__ == "__main__":
    unittest.main()
