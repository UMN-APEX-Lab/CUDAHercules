"""Native Claude Messages API with durable credit-exhaustion fallback to AWS.

Only the trusted controller imports this module. Credentials never enter model
history, usage files, or sandbox environments. Normal API keys cannot read the
credit balance; successful usage and explicit billing failures are monitored.
"""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time


def read_env(path):
    result={}
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if line and not line.startswith('#') and '=' in line:
            key,value=line.split('=',1);result[key.strip()]=value.strip().strip('"\'')
    return result


def credit_exhausted(error):
    body=getattr(error,'body',None) or {}
    text=(json.dumps(body,default=str)+' '+str(error)).lower()
    terms=['credit balance','insufficient credit','insufficient_quota','quota_exceeded',
           'spend limit','spending limit','spend cap','monthly spend','monthly usage limit',
           'billing_error','billing_disabled','payment_required','credits exhausted','budget exceeded']
    return getattr(error,'status_code',None)==402 or any(term in text for term in terms)


def native_messages(messages):
    """Convert legacy AWS history while echoing native signed blocks verbatim."""
    system=[];result=[]
    for message in messages:
        role=message['role'];content=message.get('content') or ''
        if role=='system':
            system.append(content);continue
        if role=='tool':
            role='user';blocks=[dict(type='tool_result',tool_use_id=message['tool_call_id'],content=content)]
        elif role=='assistant' and message.get('_anthropic_content') is not None:
            blocks=copy.deepcopy(message['_anthropic_content'])
        else:
            blocks=[dict(type='text',text=content)] if content else []
            for call in message.get('tool_calls') or []:
                function=call['function']
                blocks.append(dict(type='tool_use',id=call['id'],name=function['name'],input=json.loads(function['arguments'])))
        if not blocks:continue
        if result and result[-1]['role']==role:result[-1]['content'].extend(blocks)
        else:result.append(dict(role=role,content=blocks))
    return '\n\n'.join(system),result


def aws_messages(messages):
    allowed={'role','content','tool_calls','tool_call_id','name'}
    return [{k:copy.deepcopy(v) for k,v in m.items() if k in allowed} for m in messages]


def official_cost(usage):
    """USD estimate for Opus 5 standard pricing; not an account-balance reading.

    https://platform.claude.com/docs/en/models/opus-5/overview
    Cached reads and 5m writes: $0.50/$6.25 per MTok; input/output: $5/$25.
    """
    creation=usage.get('cache_creation') or {}
    hourly=creation.get('ephemeral_1h_input_tokens',0) or 0
    five_min=(usage.get('cache_creation_input_tokens',0) or 0)-hourly
    return ((usage.get('input_tokens',0) or 0)*5+(usage.get('output_tokens',0) or 0)*25+
            max(0,five_min)*6.25+hourly*10+(usage.get('cache_read_input_tokens',0) or 0)*.5)/1e6


class OfficialClaudeWithFallback:
    def __init__(self,aws_client,env_file,state_file,*,model='claude-opus-5',effort='low',task_id='',official_client=None,key_env='CLAUDE_API_KEY'):
        self.aws=aws_client;self.env_file=Path(env_file).expanduser();self.state_file=Path(state_file)
        self.key_env=key_env
        self.model=model;self.effort=effort;self.task_id=task_id
        self.client=official_client;self.client_key_id=None
        self.chat=SimpleNamespace(completions=self)
        self.last_provider='anthropic'
        self.state_file.parent.mkdir(parents=True,exist_ok=True)

    @property
    def preserve_native_history(self):
        return self.last_provider=='anthropic'

    def _state(self,key_id,update=None):
        with self.state_file.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
            if state.get('key_id')!=key_id:
                state=dict(key_id=key_id,active_provider='anthropic',official_disabled=False,
                    official_requests=0,estimated_official_cost_usd=0.,usage={})
            if update:update(state)
            temp=self.state_file.with_suffix(f'.{os.getpid()}.tmp')
            temp.write_text(json.dumps(state,indent=2)+'\n');os.chmod(temp,0o600);temp.replace(self.state_file)
            return state

    def _event(self,event):
        event=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_id=self.task_id,**event)
        with self.state_file.with_suffix('.jsonl').open('a') as output:
            output.write(json.dumps(event,default=str)+'\n')

    def _usage(self,key_id,usage,**extra):
        cost=official_cost(usage)
        def update(state):
            state['official_requests']+=1;state['active_provider']='anthropic'
            state['estimated_official_cost_usd']+=cost
            for key in ['input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens']:
                state['usage'][key]=state['usage'].get(key,0)+(usage.get(key,0) or 0)
            state['last_response_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        state=self._state(key_id,update)
        self._event(dict(type='usage',provider='anthropic',model=self.model,effort=self.effort,
            usage=usage,estimated_cost_usd=cost,estimated_cumulative_cost_usd=state['estimated_official_cost_usd'],**extra))

    def _fallback(self,key_id,reason):
        def update(state):
            state.update(official_disabled=True,active_provider='aws',fallback_reason=reason,
                fallback_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        self._state(key_id,update)
        self._event(dict(type='provider_fallback',provider='aws',reason=reason))

    def _aws(self,kwargs,timeout):
        self.last_provider='aws'
        kwargs=dict(kwargs,messages=aws_messages(kwargs['messages']))
        response=self.aws.chat.completions.create(**kwargs,timeout=timeout)
        # Do not modify the SDK response type; AgentLoop reads this router field.
        usage=response.usage.model_dump() if response.usage else {}
        self._event(dict(type='usage',provider='aws',model=kwargs['model'],effort=kwargs.get('reasoning_effort'),usage=usage))
        return response

    def create(self,*,timeout=900,**kwargs):
        import anthropic
        from openai import (APIConnectionError,APITimeoutError,BadRequestError,
                            InternalServerError,RateLimitError)
        started=time.monotonic();deadline=started+timeout
        key=os.environ.get(self.key_env) or read_env(self.env_file).get(self.key_env,'')
        key_id=hashlib.sha256(key.encode()).hexdigest()[:16]
        state=self._state(key_id)
        if not key or state.get('official_disabled'):
            if not key:self._fallback(key_id,'missing_official_key')
            return self._aws(kwargs,max(.1,deadline-time.monotonic()))
        self.last_provider='anthropic'
        if self.client is None or (self.client_key_id is not None and self.client_key_id!=key_id):
            self.client=anthropic.Anthropic(api_key=key,max_retries=0,timeout=min(120,timeout))
        self.client_key_id=key_id
        system,messages=native_messages(kwargs['messages'])
        tools=[dict(name=t['function']['name'],description=t['function'].get('description',''),
                    input_schema=t['function']['parameters']) for t in kwargs.get('tools') or []]
        payload=dict(model=self.model,max_tokens=kwargs['max_completion_tokens'],system=system,
            messages=messages,thinking={'type':'adaptive'},output_config={'effort':self.effort},
            cache_control={'type':'ephemeral'})
        if tools:payload.update(tools=tools,tool_choice={'type':'auto'})
        def attempt(state):
            state['official_request_attempts']=state.get('official_request_attempts',0)+1
            state['last_request_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._state(key_id,attempt)
        self._event(dict(type='request',provider='anthropic',model=self.model,effort=self.effort,
            max_tokens=payload['max_tokens'],message_count=len(messages)))
        try:
            with self.client.messages.stream(**payload,timeout=min(120,timeout)) as stream:
                for event in stream:
                    if time.monotonic()>=deadline:
                        raise TimeoutError('Official response exceeded remaining request/task budget')
                message=stream.get_final_message()
            usage=message.usage.model_dump(exclude_none=True)
            self._usage(key_id,usage,request_id=getattr(message,'_request_id',None),max_tokens=payload['max_tokens'])
            from openai.types.chat import ChatCompletionMessage
            blocks=[b.model_dump(exclude_none=True) for b in message.content]
            calls=[dict(id=b['id'],type='function',function=dict(name=b['name'],arguments=json.dumps(b['input'])))
                   for b in blocks if b['type']=='tool_use']
            content='\n'.join(b['text'] for b in blocks if b['type']=='text')
            normalized=ChatCompletionMessage(role='assistant',content=content,tool_calls=calls or None,
                anthropic_content=blocks)
            prompt=sum(usage.get(k,0) or 0 for k in ['input_tokens','cache_creation_input_tokens','cache_read_input_tokens'])
            mapped_usage=dict(prompt_tokens=prompt,completion_tokens=usage.get('output_tokens',0),
                total_tokens=prompt+(usage.get('output_tokens',0) or 0),anthropic_usage=usage)
            reason='tool_calls' if calls else 'length' if message.stop_reason=='max_tokens' else 'stop'
            return SimpleNamespace(choices=[SimpleNamespace(message=normalized,finish_reason=reason)],
                usage=SimpleNamespace(model_dump=lambda:mapped_usage))
        except anthropic.APIStatusError as error:
            status=error.status_code
            if credit_exhausted(error) or status in [401,403]:
                reason='official_credit_or_spend_limit_exhausted' if credit_exhausted(error) else 'official_auth_or_permission_error'
                self._fallback(key_id,reason)
                return self._aws(kwargs,max(.1,deadline-time.monotonic()))
            text=str(error).replace(key,'[REDACTED]')
            self._event(dict(type='api_error',provider='anthropic',status=status,error=text[:1000]))
            kind=RateLimitError if status==429 else InternalServerError if status>=500 else BadRequestError
            raise kind(text,response=error.response,body=error.body) from None
        except (anthropic.APITimeoutError,TimeoutError) as error:
            self._event(dict(type='api_error',provider='anthropic',error='request_timeout'))
            request=getattr(error,'request',None)
            if request is None:
                import httpx2
                request=httpx2.Request('POST','https://api.anthropic.com/v1/messages')
            raise APITimeoutError(request=request) from None
        except anthropic.APIConnectionError as error:
            raise APIConnectionError(request=error.request,message='Official Claude connection failed') from None
        except anthropic.APIError as error:
            # Streaming errors may arrive inside HTTP 200 and have no status.
            if credit_exhausted(error):
                self._fallback(key_id,'official_credit_or_spend_limit_exhausted')
                return self._aws(kwargs,max(.1,deadline-time.monotonic()))
            import httpx2
            text=str(error).replace(key,'[REDACTED]')
            self._event(dict(type='api_error',provider='anthropic',error=text[:1000]))
            response=httpx2.Response(500,request=httpx2.Request('POST','https://api.anthropic.com/v1/messages'))
            raise InternalServerError(text,response=response,body=getattr(error,'body',None)) from None
