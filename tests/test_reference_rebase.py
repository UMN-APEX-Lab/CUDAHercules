"""Recorded timing migration must preserve kernels, budgets and case identity."""
import importlib.util
import math
import json
from pathlib import Path
import shutil
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('reference_rebase',Path(__file__).parents[1]/'scripts/rebase_reference_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ReferenceRebaseTests(unittest.TestCase):
    def test_replaces_only_reference_times_and_uses_geomean(self):
        original={'speedup':2,'per_size':[
            {'name':'decode','ref_min_ms':4,'kernel_min_ms':2,'speedup':2},
            {'name':'prefill','ref_min_ms':8,'kernel_min_ms':4,'speedup':2}]}
        result=module.rebase_metrics(original,{'decode':1,'prefill':4})
        self.assertAlmostEqual(result['speedup'],math.sqrt(.5))
        self.assertEqual([x['kernel_min_ms'] for x in result['per_size']],[2,4])
        self.assertEqual(original['per_size'][0]['ref_min_ms'],4)
        self.assertEqual(original['speedup'],2)

    def test_rejects_missing_or_renamed_case(self):
        record={'per_size':[{'name':'decode','kernel_min_ms':2}]}
        with self.assertRaises(ValueError):module.rebase_metrics(record,{'prefill':1})
        with self.assertRaises(ValueError):module.rebase_metrics(record,{'decode':1,'prefill':4})

    def test_rejects_invalid_or_missing_measurements(self):
        for time in [0,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                module.rebase_metrics({'per_size':[{'name':'a','kernel_min_ms':time}]},{'a':1})
        with self.assertRaises(ValueError):module.rebase_metrics({'speedup':2},{'a':1})

    def test_completed_task_preserves_source_and_spent_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'old';destination=Path(directory)/'new';source.mkdir()
            cases=[{'name':'decode','ref_min_ms':4,'kernel_min_ms':2,'speedup':2}]
            final={'index':3,'status':'PASS','speedup':2,'per_size':cases,'in_loop_speedup':2}
            original=dict(task_id='task',source_hash='old-hash',final_status='PASS',best_speedup=2,
                evals_used=3,session={'llm_calls':47},in_loop_best_submission=3,
                final_reeval={'final':final,'tried':[final]})
            (source/'result.json').write_text(json.dumps(original))
            (source/'submissions.jsonl').write_text(json.dumps(dict(final,mode='full'))+'\n')
            state=dict(evals_used=3,evals_limit=10,elapsed_s=3300,best_speedup=2,promoted=True)
            (source/'judge_state.json').write_text(json.dumps(state))
            shutil.copytree(source,destination)
            result=module.rebase_task(destination,source,{'source_hash':'new-hash',
                'per_size':[{'name':'decode','ref_min_ms':1}]})
            self.assertEqual(result['best_speedup'],.5)
            self.assertEqual(result['session']['llm_calls'],47)
            self.assertEqual(result['evals_used'],3)
            self.assertFalse(result['reference_rebase']['kernel_execution_performed'])
            updated=json.loads((destination/'judge_state.json').read_text())
            self.assertEqual(updated['elapsed_s'],3300)
            self.assertFalse(updated['promoted'])
            self.assertEqual(json.loads((source/'result.json').read_text()),original)
            self.assertTrue((destination/'reference_history/old-hash/result.json').exists())


if __name__=='__main__':unittest.main()
