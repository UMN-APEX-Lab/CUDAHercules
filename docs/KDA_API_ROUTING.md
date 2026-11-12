# KDA official Claude API with an OpenAI-compatible fallback

Launch `scripts/eval_kda.py` with
`--api-provider anthropic_with_openai_fallback --reasoning-effort low` to prefer
the native Claude Messages API. The logical model remains `claude-opus-5`.
`CLAUDE_API_KEY` is read from the trusted `.env` before each request, so replacing
it switches subsequent requests without restarting the run. A request already
in flight finishes using its original provider and key. OpenAI-compatible
endpoints retain the neutral `OPENAI_API_KEY` and `OPENAI_BASE_URL` defaults.
Other variable names can be selected with `--api-key-env` / `--api-base-env`.
Keys are never put in
command arguments, model history, usage logs or sandbox environments.

The existing `--claude-key-env NAME --claude-effort low` official-only mode is
also supported and does not require OpenAI-compatible credentials. When combined with
`--api-provider anthropic_with_openai_fallback`, `NAME` selects the official key
variable for the fallback router; the OpenAI-compatible fallback credentials
are required. `--reasoning-effort` overrides `--claude-effort` in fallback mode.

The official request uses adaptive thinking with `output_config.effort=low`,
streaming, and automatic 5-minute prompt caching. The per-request output cap
remains 65,536 tokens, including thinking tokens. Native assistant blocks,
including signed thinking, are stored and echoed intact across tool calls.
History produced through the OpenAI-compatible endpoint is converted without repeating any tool execution.

Usage is updated after each successful official response:

- `.cache/kda_official_usage.json`: current provider, key fingerprint, request
  counts, token totals, cumulative estimated official cost, and fallback state.
- `.cache/kda_official_usage.jsonl`: per-request/provider events and raw usage.
- Task transcripts: actual provider, output cap, effort and response usage.

Ordinary API keys cannot query the current credit balance. Cost is an estimate
from reported usage, not a remaining-balance reading. The Opus 5 standard rates
used are $5/MTok input, $25/MTok output, $6.25/MTok 5-minute cache creation and
$0.50/MTok cache reads. Interrupted requests may have unreported billable usage.
See [official pricing](https://platform.claude.com/docs/en/models/opus-5/overview)
and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

Credit-balance errors, billing errors and explicit spend-cap errors immediately
switch the same unexecuted request to the OpenAI-compatible endpoint. This is persisted across tasks so an
exhausted official key is not retried for every task. A different key clears the
latch. Authentication or permission failures also use that endpoint; ordinary temporary
rate limits remain retryable rather than being labeled exhausted credit.
The [official error reference](https://platform.claude.com/docs/en/api/errors)
documents the billing and spend-limit signals.

Provider changes retain completed task results, the conversation/workspace,
elapsed task time and evaluation count. A continuation run records the switch;
per-task time/evaluation limits and the reference cache stay the same.
