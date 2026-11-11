"""Budget and prompt regressions for unattended KDA evaluation."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from kda_eval.agent_loop import AgentLoop
from kda_eval.judge import Judge
from cuda_hercules.result import TaskResult,TaskStatus
from cuda_hercules.task_schema import load_task_config
from eval_kda import _sandbox_notes


def judge(tmp_path,limit=60,max_evals=1):
    config=load_task_config(str(ROOT/'tasks/class2/hopper/fa3_fwd_hdim128_fp16/task.yaml'))
    return Judge(config,str(tmp_path),max_evals,limit,900,1)


def test_judge_enforces_combined_check_and_full_evaluation_cap(tmp_path,monkeypatch):
    instance=judge(tmp_path)
    monkeypatch.setattr(instance,'_evaluate',lambda *args: {'text':'evaluated'})
    assert not instance.submit('check','candidate','/workspace/solution.cu').get('rejected')
    assert instance.submit('full','candidate','/workspace/solution.cu')['rejected']
    assert instance.evals_used==1


def test_expired_judge_does_not_charge_or_evaluate(tmp_path):
    instance=judge(tmp_path,limit=0)
    assert instance.submit('full','candidate','solution.cu')['rejected']
    assert instance.evals_used==0


def test_evaluation_timeout_cannot_exceed_remaining_task_time(tmp_path,monkeypatch):
    instance=judge(tmp_path,limit=3)
    import kda_eval.judge as module
    monkeypatch.setattr(module,'validate_cuda_solution',lambda *args,**kwargs: SimpleNamespace(valid=True))
    seen=[]
    def evaluate(*args,**kwargs):
        seen.append(kwargs['timeout_sec'])
        return TaskResult(compiled=True,correct=False,status=TaskStatus.FAIL)
    monkeypatch.setattr(module.ea,'_run_task_isolated',evaluate)
    instance.submit('full','candidate','solution.cu')
    assert 0 < seen[0] <= 3


def test_expired_agent_cannot_make_api_or_tool_calls():
    loop=object.__new__(AgentLoop)
    loop.deadline=time.time()-1
    loop.model='claude-opus-5';loop.max_completion_tokens=100;loop.reasoning_effort=''
    assert 'REJECTED' in loop._run_tool('bash',{'command':'must not run'})
    with pytest.raises(TimeoutError):loop._complete([])


def test_class2_prompt_matches_cxx20_and_hopper_arch():
    config=load_task_config(str(ROOT/'tasks/class2/hopper/fa3_fwd_hdim128_fp16/task.yaml'))
    notes=_sandbox_notes(config,'NVIDIA H200 NVL',90)
    assert '-std=c++20' in notes
    assert '`sm_90a`' in notes


def test_evaluator_timeout_kills_compilation_grandchildren(tmp_path,monkeypatch):
    import os
    import eval_agent
    pid_file=tmp_path/'child.pid'
    child="import os,time;open("+repr(str(pid_file))+",'w').write(str(os.getpid()));time.sleep(30)"
    code="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);time.sleep(30)"
    monkeypatch.setattr(eval_agent,'_ISOLATED_RUN_TASK_TEMPLATE',code)
    config=load_task_config(str(ROOT/'tasks/class2/hopper/fa3_fwd_hdim128_fp16/task.yaml'))
    result=eval_agent._run_task_isolated(config,'unused.cu',timeout_sec=1)
    assert result.status==TaskStatus.ERROR and 'timeout' in result.error_msg
    pid=int(pid_file.read_text())
    status=Path(f'/proc/{pid}/stat')
    assert not status.exists() or status.read_text().split()[2]=='Z'


def test_resumed_judge_keeps_elapsed_time_and_evaluation_count(tmp_path,monkeypatch):
    config=load_task_config(str(ROOT/'tasks/class2/hopper/fa3_fwd_hdim128_fp16/task.yaml'))
    instance=Judge(config,str(tmp_path),10,3600,900,1,
                   resume_state=dict(elapsed_s=2440.5,evals_used=9))
    assert 2440.5<=instance.elapsed()<2442
    assert 1158<instance.budget()['deadline_epoch']-time.time()<=1159.5
    monkeypatch.setattr(instance,'_evaluate',lambda *args: {'text':'evaluated'})
    assert not instance.submit('full','candidate','candidate.cu').get('rejected')
    assert instance.submit('check','candidate','candidate.cu')['rejected']


def test_interrupted_evaluation_is_charged_before_compilation(tmp_path,monkeypatch):
    import json
    instance=judge(tmp_path)
    def interrupted(*args):raise RuntimeError('interrupted compilation')
    monkeypatch.setattr(instance,'_evaluate',interrupted)
    with pytest.raises(RuntimeError):instance.submit('full','candidate','candidate.cu')
    assert json.loads((tmp_path/'judge_state.json').read_text())['evals_used']==1


def test_resume_preserves_workspace_and_rejects_stale_evaluated_benchmark(tmp_path):
    import json
    from kda_eval.resume import load_pause_checkpoints,restore_workspace
    task='class1/hopper/test';source=tmp_path/'source';checkpoint=source/'checkpoint'
    (checkpoint/'workspace').mkdir(parents=True)
    (checkpoint/'workspace/solution.h').write_text('preserved kernel')
    (checkpoint/'workspace/TASK.md').write_text('old task')
    (checkpoint/'messages.json').write_text('[]')
    (checkpoint/'transcript.jsonl').write_text('')
    (source/'config.json').write_text(json.dumps(dict(model='claude-opus-5',time_limit_min=60,
        max_evals=10,task_source_hashes={task:'old'})))
    pause=dict(status='PAUSED',active_tasks=[dict(task=task.replace('/','_'),
        checkpoint_directory=str(checkpoint),elapsed_seconds=2440,remaining_seconds=1160,
        judge_state=dict(evals_used=0))])
    (source/'pause.json').write_text(json.dumps(pause))
    restored=load_pause_checkpoints(source,{task:'new'},'claude-opus-5',3600,10)[task]
    assert restored['judge_state']['elapsed_s']==2440 and restored['benchmark_changed']
    dest=tmp_path/'new';dest.mkdir();restore_workspace(checkpoint,dest,'current task')
    assert (dest/'workspace/solution.h').read_text()=='preserved kernel'
    assert (dest/'workspace/TASK.md').read_text()=='current task'
    assert (checkpoint/'workspace/TASK.md').read_text()=='old task'
    pause['active_tasks'][0]['judge_state']['evals_used']=1
    (source/'pause.json').write_text(json.dumps(pause))
    with pytest.raises(ValueError,match='prior evaluations'):
        load_pause_checkpoints(source,{task:'new'},'claude-opus-5',3600,10)


def test_resumed_dialogue_closes_interrupted_tool_without_reexecuting(tmp_path):
    import json
    loop=AgentLoop(None,'claude-opus-5',None,str(tmp_path),'unused',time.time()-1,
                   max_completion_tokens=100,context_limit=1000,max_llm_calls=1000,max_total_tokens=0)
    messages=[dict(role='system',content='system'),dict(role='user',content='task'),
              dict(role='assistant',content='',tool_calls=[dict(id='pending',type='function',
                  function=dict(name='bash',arguments='{}'))])]
    result=loop.run('Resume with remaining budget',initial_messages=messages,
                    initial_stats=dict(llm_calls=33),elapsed_s=2440)
    saved=json.loads((tmp_path/'messages.json').read_text())
    assert result['llm_calls']==33 and result['wall_s']>=2440
    assert saved[-2]['tool_call_id']=='pending' and 'Interrupted' in saved[-2]['content']
    assert saved[-1]['role']=='user'


def test_stopping_judge_also_stops_evaluator_in_separate_session(tmp_path):
    import subprocess
    from eval_kda import _stop
    pid_file=tmp_path/'evaluator.pid'
    child="import os,time;open("+repr(str(pid_file))+",'w').write(str(os.getpid()));time.sleep(60)"
    parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"],start_new_session=True);time.sleep(60)"
    proc=subprocess.Popen([sys.executable,'-c',parent],start_new_session=True)
    for _ in range(100):
        if pid_file.exists():break
        time.sleep(0.01)
    assert pid_file.exists()
    pid=int(pid_file.read_text());_stop(proc)
    for _ in range(100):
        path=Path(f'/proc/{pid}/stat')
        if not path.exists() or path.read_text().split()[2]=='Z':break
        time.sleep(0.01)
    assert not path.exists() or path.read_text().split()[2]=='Z'


def test_live_output_token_limit_is_sent_to_next_api_request(tmp_path):
    import json
    loop=object.__new__(AgentLoop)
    loop.task_out=str(tmp_path/'task');loop.deadline=time.time()+60
    loop.model='claude-opus-5';loop.max_completion_tokens=32000;loop.reasoning_effort=''
    loop._log=lambda **kwargs:None
    requests=[]
    def complete(**kwargs):requests.append(kwargs);return 'response'
    loop.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    assert loop._complete([])=='response'
    (tmp_path/'runtime_settings.json').write_text(json.dumps(dict(max_completion_tokens=65536)))
    assert loop._complete([])=='response'
    assert [r['max_completion_tokens'] for r in requests]==[32000,65536]
