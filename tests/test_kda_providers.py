"""Provider routing, native thinking round trips and billing fallback checks."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

anthropic=pytest.importorskip('anthropic')
httpx2=pytest.importorskip('httpx2')

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from kda_eval.providers import OfficialClaudeWithFallback,native_messages,aws_messages,credit_exhausted,official_cost


def error(status,message):
    request=httpx2.Request('POST','https://api.anthropic.com/v1/messages')
    return anthropic.APIStatusError(message,response=httpx2.Response(status,request=request),
        body={'error':{'type':'invalid_request_error','message':message}})


class Stream:
    def __init__(self,response=None,failure=None):self.response=response;self.failure=failure
    def __enter__(self):
        if self.failure:raise self.failure
        return self
    def __exit__(self,*args):pass
    def __iter__(self):return iter([])
    def get_final_message(self):return self.response


def clients(tmp_path,failure=None):
    env=tmp_path/'.env';env.write_text('CLAUDE_API_KEY=private-test-value\n')
    calls=[];aws_calls=[]
    blocks=[{'type':'thinking','thinking':'','signature':'unchanged-signature'},
            {'type':'tool_use','id':'tool_1','name':'bash','input':{'command':'true'}}]
    response=SimpleNamespace(content=[SimpleNamespace(model_dump=lambda exclude_none=True,b=b:copy.deepcopy(b)) for b in blocks],
        usage=SimpleNamespace(model_dump=lambda **kwargs:dict(input_tokens=20,output_tokens=10,
            cache_creation_input_tokens=100,cache_read_input_tokens=200)),stop_reason='tool_use')
    def stream(**kwargs):calls.append(kwargs);return Stream(response,failure)
    official=SimpleNamespace(messages=SimpleNamespace(stream=stream))
    def complete(**kwargs):
        aws_calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:dict(prompt_tokens=1,completion_tokens=1)))
    aws=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    router=OfficialClaudeWithFallback(aws,env,tmp_path/'usage.json',official_client=official)
    return router,calls,aws_calls,blocks,official,aws,env


def request(router,messages=None):
    return router.create(model='claude-opus-5',messages=messages or [{'role':'user','content':'task'}],
        tools=[{'type':'function','function':{'name':'bash','description':'shell','parameters':{'type':'object'}}}],
        max_completion_tokens=65536,reasoning_effort='low',timeout=30)


def test_native_low_effort_cap_caching_usage_and_signed_history(tmp_path):
    router,calls,aws_calls,blocks,*_=clients(tmp_path)
    response=request(router)
    assert not aws_calls
    assert calls[0]['max_tokens']==65536
    assert calls[0]['output_config']=={'effort':'low'}
    assert calls[0]['thinking']=={'type':'adaptive'}
    assert calls[0]['cache_control']=={'type':'ephemeral'}
    assert response.usage.model_dump()['prompt_tokens']==320
    message=response.choices[0].message
    assert message.tool_calls[0].function.name=='bash'
    history=[{'role':'user','content':'task'},{'role':'assistant','content':'',
        'tool_calls':[x.model_dump() for x in message.tool_calls],'_anthropic_content':message.anthropic_content},
        {'role':'tool','tool_call_id':'tool_1','content':'ok'}]
    request(router,history)
    assert calls[1]['messages'][1]['content']==blocks
    assert calls[1]['messages'][2]['content'][0]['type']=='tool_result'
    state=json.loads((tmp_path/'usage.json').read_text())
    assert state['official_requests']==2
    assert state['estimated_official_cost_usd']==pytest.approx(2*official_cost(dict(input_tokens=20,
        output_tokens=10,cache_creation_input_tokens=100,cache_read_input_tokens=200)))
    assert 'private-test-value' not in (tmp_path/'usage.json').read_text()
    assert 'private-test-value' not in (tmp_path/'usage.jsonl').read_text()


def test_credit_exhaustion_retries_same_conversation_on_aws_and_latches_between_tasks(tmp_path):
    router,calls,aws_calls,blocks,official,aws,env=clients(tmp_path,error(400,'Your credit balance is too low'))
    history=[{'role':'user','content':'task'},{'role':'assistant','content':'x','_anthropic_content':blocks}]
    request(router,history)
    assert len(calls)==1 and len(aws_calls)==1
    assert aws_calls[0]['messages']==aws_messages(history)
    assert aws_calls[0]['max_completion_tokens']==65536
    assert router.last_provider=='aws'
    another=OfficialClaudeWithFallback(aws,env,tmp_path/'usage.json',official_client=official)
    request(another)
    assert len(calls)==1 and len(aws_calls)==2
    state=json.loads((tmp_path/'usage.json').read_text())
    assert state['official_disabled'] and state['active_provider']=='aws'


def test_normal_rate_limit_does_not_mark_credit_exhausted(tmp_path):
    from openai import RateLimitError
    router,calls,aws_calls,*_=clients(tmp_path,error(429,'Input tokens per minute rate limit exceeded'))
    with pytest.raises(RateLimitError):request(router)
    assert not aws_calls
    assert not json.loads((tmp_path/'usage.json').read_text())['official_disabled']


def test_invalid_request_is_not_disguised_as_billing_failure(tmp_path):
    from openai import BadRequestError
    router,_,aws_calls,*_=clients(tmp_path,error(400,'messages: invalid tool result'))
    with pytest.raises(BadRequestError):request(router)
    assert not aws_calls


def test_credit_error_inside_http200_stream_also_switches(tmp_path):
    failure=anthropic.APIError('credit balance too low',
        request=httpx2.Request('POST','https://api.anthropic.com/v1/messages'),
        body={'error':{'type':'billing_error'}})
    router,_,aws_calls,*_=clients(tmp_path,failure)
    request(router)
    assert len(aws_calls)==1 and router.last_provider=='aws'


def test_new_official_key_clears_credit_latch(tmp_path):
    router,calls,aws_calls,_,official,aws,env=clients(tmp_path,error(402,'billing error'))
    request(router)
    env.write_text('CLAUDE_API_KEY=replacement-private-value\n')
    # The second client must attempt official again, even though the same usage
    # state previously disabled the old key. Its fake error can fall back again.
    replacement=OfficialClaudeWithFallback(aws,env,tmp_path/'usage.json',official_client=official)
    request(replacement)
    assert len(calls)==2 and len(aws_calls)==2


@pytest.mark.parametrize('status,text,expected',[(402,'payment issue',True),
    (400,'Workspace spend limit reached',True),(429,'Monthly spend cap reached',True),
    (429,'requests per minute limit',False),(500,'server error',False)])
def test_credit_classification(status,text,expected):
    assert credit_exhausted(error(status,text)) is expected


def test_legacy_tool_results_are_merged_and_input_history_not_modified():
    history=[{'role':'system','content':'system'}, {'role':'user','content':'task'},
        {'role':'assistant','content':'','tool_calls':[
            {'id':'one','function':{'name':'bash','arguments':'{"command":"true"}'}},
            {'id':'two','function':{'name':'bash','arguments':'{"command":"false"}'}}]},
        {'role':'tool','tool_call_id':'one','content':'a'},
        {'role':'tool','tool_call_id':'two','content':'b'}]
    previous=copy.deepcopy(history);system,messages=native_messages(history)
    assert system=='system' and len(messages[-1]['content'])==2
    assert history==previous


@pytest.mark.parametrize('route,official_key_name,expected',[
    ('aws','','openai'),
    ('aws','CUSTOM_CLAUDE_KEY','official_only'),
    ('anthropic_with_aws_fallback','CUSTOM_CLAUDE_KEY','fallback'),
])
def test_merged_cli_preserves_all_provider_modes(tmp_path,monkeypatch,route,official_key_name,expected):
    import eval_kda
    import openai
    from kda_eval import claude_client

    for name in ['OPENAI_API_KEY','OPENAI_BASE_URL','CUSTOM_CLAUDE_KEY']:
        monkeypatch.delenv(name,raising=False)
    env=tmp_path/'.env'
    # Official-only mode must work without OpenAI-compatible credentials.
    env.write_text('CUSTOM_CLAUDE_KEY=test-official-value\n'+('' if expected=='official_only' else
        'OPENAI_API_KEY=test-compatible-value\nOPENAI_BASE_URL=https://example.invalid/v1\n'))
    args=SimpleNamespace(api_provider=route,claude_key_env=official_key_name,
        claude_effort='low',reasoning_effort='',env_file=str(env),api_key_env='OPENAI_API_KEY',
        api_base_env='OPENAI_BASE_URL',api_base='',run_dir=str(tmp_path),
        provider_state_file=str(tmp_path/'usage.json'),model='claude-opus-5')
    aws=SimpleNamespace();official=SimpleNamespace(effort='low');calls=[]
    def compatible_client(**kwargs):calls.append(kwargs);return aws
    monkeypatch.setattr(openai,'OpenAI',compatible_client)
    monkeypatch.setattr(claude_client,'shared_client',lambda *a,**kw:official)
    client=eval_kda.make_api_client(args,'class2/hopper/test')
    if expected=='official_only':
        assert client is official and not calls
        assert client.last_provider=='anthropic'
        assert eval_kda.load_official_only_key(args)=='test-official-value'
    else:
        assert len(calls)==1 and calls[0]['base_url']=='https://example.invalid/v1'
        assert eval_kda.load_official_only_key(args)==''
        if expected=='fallback':
            assert isinstance(client,OfficialClaudeWithFallback)
            assert client.aws is aws and client.key_env=='CUSTOM_CLAUDE_KEY'
            assert client.effort=='low'
        else:
            assert client is aws
