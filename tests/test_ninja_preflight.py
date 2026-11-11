"""CUDA extension builds must not require manually activating the virtualenv."""
import os
from pathlib import Path

import pytest
import cuda_hercules.compiler as compiler


def test_interpreter_local_ninja_is_added_to_child_path(tmp_path,monkeypatch):
    directory=tmp_path/'venv/bin';directory.mkdir(parents=True)
    executable=directory/'ninja';executable.write_text('#!/bin/sh\nprintf "test-ninja-version\\n"\n');executable.chmod(0o755)
    monkeypatch.setenv('PATH','/missing')
    monkeypatch.setattr(compiler.sys,'executable',str(directory/'python'))
    monkeypatch.setattr(compiler.sys,'prefix',str(tmp_path/'venv'))
    result=compiler.ensure_ninja_available()
    assert result['ninja']==str(executable)
    assert result['version']=='test-ninja-version'
    assert os.environ['PATH'].split(os.pathsep)[0]==str(directory)


def test_missing_ninja_fails_preflight_explicitly(tmp_path,monkeypatch):
    monkeypatch.setenv('PATH','/missing')
    monkeypatch.setattr(compiler.sys,'executable',str(tmp_path/'python'))
    monkeypatch.setattr(compiler.sys,'prefix',str(tmp_path/'no-environment'))
    with pytest.raises(RuntimeError,match='CUDA build infrastructure unavailable'):
        compiler.ensure_ninja_available()


def test_ninja_failure_happens_before_compiler_load(tmp_path,monkeypatch):
    def unavailable():raise RuntimeError('infrastructure unavailable')
    monkeypatch.setattr(compiler,'ensure_ninja_available',unavailable)
    monkeypatch.setattr(compiler,'load',lambda *a,**k:pytest.fail('build must not start'))
    with pytest.raises(RuntimeError,match='infrastructure unavailable'):
        compiler.compile_cuda_module('invalid source','invalid signature')


def test_infrastructure_annotation_preserves_original_failed_result(tmp_path):
    import json,sys
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from eval_kda import summarize
    task=SimpleNamespace(task_id='class2/hopper/test')
    directory=tmp_path/task.task_id.replace('/','_');directory.mkdir()
    result=dict(task_id=task.task_id,final_status='FAIL',best_speedup=0,evals_used=10)
    (directory/'result.json').write_text(json.dumps(result))
    (tmp_path/'infrastructure_errors.json').write_text(json.dumps({task.task_id:{'reason':'Missing Ninja'}}))
    summary=summarize(str(tmp_path),[task])
    assert summary['num_infrastructure_error']==1 and summary['num_error']==1
    assert summary['tasks'][0]['final_status']=='ERROR'
    assert summary['tasks'][0]['recorded_final_status']=='FAIL'
    assert json.loads((directory/'result.json').read_text())==result
