"""Trusted host-side pause checkpoints. Never mounted in the agent sandbox."""
import json
import os
from pathlib import Path
import shutil


def transcript_stats(checkpoint):
    stats=dict(llm_calls=0,prompt_tokens=0,completion_tokens=0,tool_calls={},
               stop_blocks=0,api_retries=0,elisions=0)
    for line in (Path(checkpoint)/'transcript.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['type']=='response':
            stats['llm_calls']+=1
            for key in ('prompt_tokens','completion_tokens'):
                stats[key]+=(row.get('usage') or {}).get(key) or 0
        elif row['type']=='tool':
            name=row['name'];stats['tool_calls'][name]=stats['tool_calls'].get(name,0)+1
        elif row['type']=='stop_blocked':stats['stop_blocks']+=1
        elif row['type']=='api_error':stats['api_retries']+=1
    return stats


def load_pause_checkpoints(source_run,task_hashes,model,limit_s,max_evals):
    source=Path(source_run).resolve()
    config=json.loads((source/'config.json').read_text())
    if (config['model']!=model or config['time_limit_min']*60!=limit_s
            or config['max_evals']!=max_evals):
        raise ValueError('Resume must retain the original model/time/evaluation limits')
    pause=json.loads((source/'pause.json').read_text())
    if pause['status']!='PAUSED':raise ValueError('Source is not an explicit paused run')
    aliases={task.replace('/','_'):task for task in task_hashes}
    result={}
    for entry in pause['active_tasks']:
        task=aliases.get(entry['task'])
        if task is None:raise ValueError(f'Paused task not selected: {entry["task"]}')
        checkpoint=Path(entry['checkpoint_directory']).resolve()
        if not checkpoint.is_relative_to(source):raise ValueError('Checkpoint outside original run')
        state=dict(entry['judge_state'])
        elapsed=float(entry['elapsed_seconds'])
        if abs(elapsed+float(entry['remaining_seconds'])-limit_s)>0.1:
            raise ValueError('Inconsistent pause wall-clock budget')
        state['elapsed_s']=elapsed
        changed=config['task_source_hashes'][task]!=task_hashes[task]
        if changed and state['evals_used']:
            raise ValueError('A changed benchmark with prior evaluations needs explicit revalidation')
        if changed:state.update(has_pass=False,best_speedup=0,best_submission=-1)
        if not (checkpoint/'workspace').is_dir() or not (checkpoint/'messages.json').is_file():
            raise ValueError('Incomplete pause workspace/messages')
        result[task]=dict(checkpoint_directory=str(checkpoint),judge_state=state,
                          benchmark_changed=changed,previous_source_hash=config['task_source_hashes'][task],
                          stats=transcript_stats(checkpoint))
    return result


def restore_workspace(checkpoint,task_out,task_md):
    source=Path(checkpoint);dest=Path(task_out)
    shutil.copytree(source/'workspace',dest/'workspace',dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('.sandbox_root','.torch_ext','__pycache__'))
    (dest/'workspace/TASK.md').write_text(task_md)
    for name in ('transcript.jsonl','messages.json','kda_budget_state.json','submissions.jsonl'):
        if (source/name).is_file():shutil.copy2(source/name,dest/name)
    if (source/'submissions').is_dir():
        shutil.copytree(source/'submissions',dest/'submissions',dirs_exist_ok=True)
    hook=dest/'kda_budget_state.json'
    if hook.exists():
        state=json.loads(hook.read_text());state['last_poll']=0;hook.write_text(json.dumps(state))
