"""Regression tests for audited scoring, numerical checks and task contracts."""
from pathlib import Path
import subprocess
import sys
import torch
import pytest
from cuda_hercules.backend_class1 import Class1MakeBackend
from cuda_hercules.eval import check_correctness, load_task_def
from cuda_hercules.task_schema import load_task_config
from cuda_hercules.static_checker import validate_cuda_solution

ROOT=Path(__file__).resolve().parents[1]


def test_every_timing_case_including_reversed_order_is_scored(monkeypatch):
    monkeypatch.setattr('cuda_hercules.backend_class1._load_ref_benchmarks_c1',lambda: {})
    task=next((ROOT/'tasks/class1/hopper').glob('48_*/task.yaml'))
    backend=Class1MakeBackend();backend._correct=True
    backend._stdout='''=== first ===
Kernel time: 0.04 ms (min: 0.02 ms)
Ref time: 0.2 ms (min: 0.08 ms)
=== second ===
Ref time: 0.4 ms (min: 0.16 ms)
Kernel time: 0.1 ms (min: 0.04 ms)
'''
    metrics=backend.run_performance(load_task_config(str(task)))
    assert len(metrics.per_size)==2
    assert metrics.speedup==pytest.approx(4.0)


def test_timing_blocks_cannot_borrow_another_cases_measurement(monkeypatch):
    monkeypatch.setattr('cuda_hercules.backend_class1._load_ref_benchmarks_c1',lambda: {})
    backend=Class1MakeBackend();backend._correct=True
    backend._stdout='''=== incomplete_ref ===
Ref time: 0.2 ms (min: 0.1 ms)
=== incomplete_solution ===
Kernel time: 0.01 ms (min: 0.005 ms)
'''
    cfg=load_task_config(str(next((ROOT/'tasks/class1/hopper').glob('48_*/task.yaml'))))
    assert backend.run_performance(cfg).per_size==[]


def test_outputs_are_matched_by_name_and_cannot_disappear():
    ref=[('O',torch.ones(2)),('lse',torch.ones(2))]
    assert check_correctness(ref,list(reversed(ref)),{})[0]
    assert not check_correctness(ref,ref[:1],{})[0]
    assert not check_correctness([],[],{})[0]


def test_nonfinite_outputs_and_lse_bias_are_rejected():
    ref=[('O',torch.ones(8)),('lse',torch.full((2,),10.0))]
    tol={'atol':0.04,'rtol':0.06,'per_output':{'lse':{'atol':2e-4,'rtol':2e-5}}}
    assert not check_correctness(ref,[('O',torch.full((8,),float('nan'))),ref[1]],tol)[0]
    assert not check_correctness(ref,[ref[0],('lse',ref[1][1]+0.3)],tol)[0]
    assert not check_correctness(ref,[('O',torch.full((8,),float('inf'))),ref[1]],tol)[0]


def test_rms_rejects_systematic_errors_hidden_by_elementwise_atol():
    ref=[('dV',torch.full((32,),0.05))]
    sol=[('dV',ref[0][1]*0.9)]
    tol={'atol':0.04,'rtol':0.06,'max_normalized_rms':0.01}
    assert not check_correctness(ref,sol,tol)[0]


def test_masked_negative_infinity_remains_a_valid_value():
    ref=[('logits',torch.tensor([1.,-float('inf')]))]
    assert check_correctness(ref,ref,{})[0]
    assert not check_correctness(ref,[('logits',torch.tensor([1.,float('nan')]))],{})[0]
    overflow=[('O',torch.tensor([float('inf')]))]
    assert not check_correctness(overflow,overflow,{})[0]


def test_hopper_policies_block_precision_reduction():
    code='''#include <cuda_fp8.h>
__global__ void kernel() { __nv_fp8_e4m3 v; }
void launch() {kernel<<<1,1>>>();}
// substantive kernel code would go here
'''
    cfg=load_task_config(str(ROOT/'tasks/class2/hopper/fa3_fwd_hdim128_fp16/task.yaml'))
    assert not validate_cuda_solution(code,blocked_patterns=cfg.anti_cheat.blocked_patterns).valid


def test_all_hopper_configs_load_and_use_explicit_arch():
    paths=[*ROOT.glob('tasks/class1/hopper/*/task.yaml'),*ROOT.glob('tasks/class2/hopper/*/task.yaml')]
    assert len(paths)==85
    for path in paths:
        cfg=load_task_config(str(path))
        assert cfg.hardware.required_features==['sm_90a']


def test_generated_tasks_and_prompts_are_reproducible():
    for name in ['generate_fa3_tasks.py','generate_hopper_descriptions.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--check'],check=True,cwd=ROOT)


def test_fa3_decode_gqa_and_backward_padding():
    path=ROOT/'tasks/class2/hopper/fa3_fwd_split_hdim128_fp16'
    task=load_task_def(str(path));args=task['SCALAR_ARGS']
    assert args['Hkv']<args['H'] and args['Sq']<args['Sk']
    module_globals=load_task_def(str(ROOT/'tasks/class2/hopper/fa3_bwd_hdim128_fp16'))['get_outputs'].__globals__
    assert module_globals['BLOCK_M']==80


def test_cache_hash_tracks_contract_and_linked_headers_but_ignores_binaries(tmp_path):
    from cuda_hercules.utils import task_source_hash
    task=tmp_path/'task';task.mkdir()
    shared=tmp_path/'shared.cuh';shared.write_text('// original\n')
    (task/'validation.cuh').symlink_to(shared)
    definition=task/'def.py';definition.write_text('SCALAR_ARGS={"Sk":8192}\n')
    original=task_source_hash(str(task))
    (task/'ref').write_bytes(b'compiled executable')
    assert task_source_hash(str(task))==original
    shared.write_text('// changed validation\n')
    assert task_source_hash(str(task))!=original
    linked=task_source_hash(str(task))
    definition.write_text('SCALAR_ARGS={"Sk":16384}\n')
    assert task_source_hash(str(task))!=linked


def test_fa3_uses_same_minimum_statistic_for_both_timings():
    task=load_task_def(str(ROOT/'tasks/class2/hopper/fa3_fwd_hdim128_fp16'))
    assert task['TIMING_STATISTIC']=='min'


@pytest.mark.parametrize('fingerprint,expected',[('stale',2.0),('current',10.0)])
def test_hopper_cache_is_used_only_for_matching_sources(monkeypatch,fingerprint,expected):
    cfg=load_task_config(str(next((ROOT/'tasks/class1/hopper').glob('48_*/task.yaml'))))
    monkeypatch.setattr('cuda_hercules.backend_class1.task_source_hash',lambda _: 'current')
    monkeypatch.setattr('cuda_hercules.backend_class1._load_ref_benchmarks_c1',lambda: {
        cfg.task_id:{'source_hash':fingerprint,'per_size':[{'ref_min_ms':10.0}]}})
    backend=Class1MakeBackend();backend._correct=True
    backend._stdout='=== test ===\nRef time: 2 ms (min: 2 ms)\nKernel time: 1 ms (min: 1 ms)\n'
    assert backend.run_performance(cfg).speedup==pytest.approx(expected)
