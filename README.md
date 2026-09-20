# Deep Agents Template

## Native lead harness — current implementation, 20 September 2026

New lead tasks use internal graph `msty_native`: native LangChain `create_agent`,
ToolNode, TodoListMiddleware and Deep Agents Filesystem/Memory/Skills middleware.
Previous graph `msty` remains for old checkpoints, text-only analysts and bounded
workers. This is compatibility, not a second visible Brain. The gateway chooses
by its saved execution version, never by matching message text.

Actual approved Store contents are read through StoreBackend/CompositeBackend.
`/memory/PROJECT.md` is shared across chats; `/skills/` supplies three progressive
descriptors, with bodies read on demand. Memory/skills are model-read-only.
Virtual scratch uses StateBackend, not the Mac disk. Real file/browser/build/
publication actions still require existing local MCP tools and grants. Reviewed
operator Store updates affect new chats without replacing a Python constant.
Missing/unavailable Store has an explicit packaged fallback; corrupt approval
fails closed. Canonical journals/Engram remain the sources, not this projection.

Native filesystem calls enforce this distinction before backend access: reads
stay in `/scratch/`, `/memory/`, `/skills/`, `/large_tool_results/`; model writes
stay in `/scratch/`. Mac, relative and traversal paths return an error without
access. Native write observations explicitly identify virtual-only storage.
Real local artifacts require external MCP callbacks and actual verification.
The task verifier decodes the actual Msty HTTP/MCP TextContent envelope, retaining
the plan identity. Failed checks survive native TODO updates and block unsupported
completion prose, including when the original request contains an action veto.
These guards were added after a real-client acceptance exposed a virtual file
being claimed as local; the failed evidence must remain in the delivery journal.

Every model generation retains a separate reservation. An internal tool batch
is checkpointed before tools execute; `msty_native_continue` resumes automatically
after shared-action reservation and stop/budget validation. Native/local actions share24 actions and the existing
$1 estimated task cap. External callbacks retain exact batch/ID/schema/task checks.
Internal narration is buffered; no internal tool call is sent to Msty for execution.
Native tools have no hidden model calls. Only task-relevant skills/tools are needed;
an ordinary answer does not require planning, a council or a consultant.

The stack is composed with `create_agent`: Deep Agents0.4.11 `create_deep_agent`
unconditionally installs summary/subagent model calls outside the local accounting
gate. Offline probes demonstrated swallowed interrupts and summary replay. Native
summarization and native `task` remain disabled pending separate metered integration;
existing explicit compaction and bounded local workers remain available. This is
not activation of every Deep Agents feature or proof of autonomous weight learning.

Shared knowledge is distinct from chat history. Msty2.9.11 normal chat requests do
not automatically pass stable split IDs. Tool chains have durable identity, while
independent turns without explicit identity still start separate graph threads.
No shared static ID, text-based linking, secret indexing or permission expansion.
Dependencies are pinned and CI uses frozen uv.lock. Deployment and UI acceptance
must be checked separately in the owner's unified change journal.
Hosted Agent Server0.14.2 requires langgraph-sdk>=0.4.4; the client SDK is pinned
to0.4.4 accordingly and covered by the same offline regression. A locally passing
SDK0.3.15 pin was rejected by the hosted build; it was never made active.

Sources: [memory](https://docs.langchain.com/oss/python/deepagents/memory),
[skills](https://docs.langchain.com/oss/python/deepagents/skills),
[middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in).

## Historical startup projection — superseded for new lead tasks

Correction: the old path below verified a Store copy but supplied the packaged
Python constant to the model. It was not dynamic Store-backed memory. The new
native path above reads actual approved Store content. Old checkpoints preserve
their existing behavior for compatibility; previous evidence is not rewritten.

`msty_memory.load_context` uses the Agent Server's native `Runtime.store`, shared
across chats in this single-owner private deployment. One version/hash-pinned
projection is seeded once, then read without an LLM, MCP discovery or filesystem
scan. It contains curated stable project/repository/storage locations, not secrets,
raw chat history, fresh deployment status or new authority. Canonical project
documents and Engram remain the sources; changing this projection requires a
reviewed version update, it is not autonomous free-form learning.

The exact verified projection is automatically in the lead model's system context.
Analysts do not receive it. Store failures or a mismatching item fall back to the
packaged projection within one second and expose `project_memory_delivery` status;
they do not trigger disk discovery or overwrite corrupt data. Requests cannot choose
a namespace or substitute memory content. New chats share knowledge, not messages,
task permissions or publication grants. Source/permission checks still apply before
mutations. `msty_project_resolve` is now conditional, not mandatory onboarding for
every familiar-site question. Saved Msty project and existing-chat prompts must be
updated separately through UI; code alone cannot override their stale instructions.

Built on [native LangGraph Store](https://docs.langchain.com/oss/python/langgraph/stores).
No new database/service, embeddings, scheduler, paid judge or model-weight training.

## Local isolated site executor — 20 September 2026

The compact operating passport documents optional `msty_site_prepare/file/check/
status/cancel/release`. Only supplied MCP schemas are usable; installing source
does not imply native-client reconnection. The cloud graph remains a planner and
tool-call loop: source checkout, offline non-root Docker Node22/pnpm11.7 checks,
and credential-owning GitHub/Vercel delivery are local. No cloud shell or new
orchestrator is added. Local checks bind to exact source content; CI and explicit
original-user release scope are mandatory. Connection-only tasks cannot publish.
The existing parent budget/action limits apply; bounded status waits reduce polls.
No production changes are made merely to validate this integration. Build success
without runtime secrets is not UI or business acceptance. Native integration and
current acceptance are tracked by the owner's unified local change journal.

## Compact operating context and bounded artifact workers — 20 September 2026

The server-owned operating policy and native memory projection supply a compact
project/access map automatically on every lead step. It is not a request to read
the complete project history, a credential dump or weight training. Unchanging
entry points need no repeated discovery; mutable source/deployment state still
requires a task-relevant check. Only actually supplied tool schemas are usable.

Local Admin 1.5.0 adds paged project reads, existing-project resolution and real
artifact-worker start/status/cancel. Workers have isolated declared file outputs,
not arbitrary shell, site publication or independent budgets. The local gateway
issues one-use worker capabilities and accounts lead/consultants/workers against
one parent task cap and aggregate action quota. No council or retired supervisor
is restored. Installation and native client reconnect are separate release gates.

Task-plan and task-verification receipts optionally include strictly validated
`learning` metadata. Local verification records evidence-linked candidate lessons
and attempts projection to the existing Engram. Local file success, NAS delivery,
retrieved hints and observed same-plan recovery remain separate facts; none proves
actual lesson application or causal improvement. Legacy receipts stay supported.

## Optional incremental text transport — 20 September 2026

`text_stream_protocol=msty-text-delta-v1` opts one ordinary response step into
the existing model's native `astream(..., stream_usage=True)`. Without it, the
original single `ainvoke` path is unchanged. This is a rendering flag, not a
model, budget or authority selector: verified tool callbacks may enable it or
clear it with `None` (omission also clears it). Unknown versions fail before
generation. Native compaction remains a separate, nonstreaming counted call.

LangGraph `custom` events before the existing `validated_result` may be:

```json
{"type":"text_delta","version":1,"seq":0,"text":"nonempty provisional text"}
{"type":"text_invalidated","version":1}
```

Sequence numbers are consecutive from zero; cumulative text is limited to
256 KiB UTF-8. Only ordinary text is exposed, never tool arguments, reasoning
blocks or executable tool fragments. All tool/schema/model/consultation limits
and artifact-completion gates still apply to the fully aggregated final message.
Every step with a nonempty artifact task contract is buffered, including failed
or verified contracts, so the model cannot show a premature completion claim
before the native verification gate. Buffered final responses need not produce
any delta events. The current gateway starts an independent user turn in a fresh
thread; an old contract does not turn off streaming for future new tasks. This
does not introduce stable identity across unidentified user turns.
There is no fallback inference or stream retry.

Deltas are **provisional**, not accepted output or proof of completed work.
After a final guard changes their text, rejects model identity, or a stream
error, `text_invalidated` precedes any guarded final. A consumer must reject that
provisional answer and release no actions; appended OpenAI SSE cannot erase text
already shown. Known `length`/filter/refusal termination with unchanged text is
not invalidated: it retains its provider finish reason, measured usage and
incomplete status, with no executable partial tool calls. For a successful
streamed answer, concatenated deltas must equal the final ordinary text.
The local bridge must still validate the final message, usage/model identity and
checkpoint before releasing any tools. Cancellation propagates without retry;
an EOF without a provider finish marker is blocked.

The pinned LangChain adapter aggregates full usage and finish metadata. A narrow
OpenAI chunk-converter override preserves raw stream usage for the existing
fail-unknown accounting validator: SDK-filled zeros must not replace missing
provider counts. Duplicate usage receipts and broken streams are unknown, never
zero-cost successes; output rejected by final guards retains measured usage.
Tests use the real compiled graph and installed chunk conversion with mocked
providers. Their success does not establish deployment, native UI latency or
provider/Gateway support. See `test_msty_stream.py`; sources:
[LangGraph custom writer](https://reference.langchain.com/python/langgraph/config/get_stream_writer),
[LangChain OpenAI streaming](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).

## Bounded task criteria and metered compaction — 20 September 2026

This source increment requires the matching gateway and native Admin MCP release.
Offline tests are not evidence of deployment or native UI acceptance. The gateway
supplies an immutable task UUID and pricing/profile/output manifest; the cloud
asserts it against its server-selected model and counts **every** bound input.
Luna user image attachments use the bounded image admission described below;
unsupported image forms still fail before generation. Combined admission remains
180000 tokens, and no public/local-network access is added.

When real `msty_task_plan` / `msty_task_verify` tools are attached, an actual-issued
plan and its typed MCP observation establish bounded, **model-proposed** artifact
criteria. Before a plain final with pending checks, the code emits the existing
native verification tool call without an extra model generation. Explicit current
user stop/wait/plan-only/explain-only phrases, disabled tools and the 24-action cap
veto it. The phrase veto is conservative, not a complete intent classifier.
Tool-output prose cannot trigger this veto. A passed receipt with matched plan ID,
source hashes and every declared check allows `verified_against_observations`;
this is neither signed execution proof nor verified business-task delivery.
Further actions invalidate earlier verification. Unknown/mismatching observations
remain blocked; no verdict is inferred from an assistant's prose.

With `compaction_protocol=msty-compaction-v1`, a lead context reaching 120000
admission tokens may summarize one contiguous range of old complete text-only tool bundles
(16–400 KB), keeping the newest two bundles, pending pairs and **all** user/system
messages verbatim. Up to eight hash-bound summaries are persisted in the same
checkpoint. The canonical source messages are never replaced; checkpoint state
and `msty_compaction.source_messages` provide source readback. This does not add a
model-visible retrieval tool. The model-visible summary is explicitly unverified
historical memory and never completion evidence. Source-reference/size validation
cannot prove semantic completeness of a lossy summary.

Each summary is one separately published, metered generation (`msty_stage=compaction`)
followed by a native `msty_compaction` interrupt. The gateway settles that run and
resumes once with no new input for the ordinary generation, under the same task
budget. At most one summary plus one normal generation occur in a client HTTP leg;
each remains a separate cloud run and accounting record. Summary output is capped
at `min(2048, original_output_limit)`. Malformed/partial summaries retain usage,
preserve originals and block without a repair generation. If the projection is
still over 180000, it blocks instead of silently cutting owner instructions.

Compaction is disabled for analysts and legacy requests. It does not solve the
2 MB incoming transport bound, create stable identity across unidentified new
user turns, or make long chats unlimited. Old pending pre-manifest callbacks need
a fresh user turn under the new gateway; they are not silently upgraded.
Quick answers remain one generation with no plan or compaction pass. Native
LangGraph MIT checkpoint/interrupt is reused; stock summary middleware is not
used because its hidden model calls/history replacement violate this boundary.
See `test_msty_compaction.py` and `test_msty_task.py` for offline acceptance.

## Economical Msty Brain — 20 September 2026

### Unified inference transport

Operator setting `MSTY_LLM_GATEWAY_ENABLED=1` routes both existing profiles through
LangSmith Gateway. Luna uses `/openai/v1/chat/completions` with its native model ID;
DeepSeek uses `/v1/chat/completions` and the saved `custom/Msty%20DeepSeek%20Flash`
configuration. The latter must match server config ID
`ae7376e7-fea6-43cb-b50c-97db118c8c47` in
`MSTY_LLM_GATEWAY_DEEPSEEK_CONFIG_ID`. The Gateway credential is supplied only by
the deployment secret `LANGSMITH_GATEWAY_API_KEY`; provider keys reside in the
workspace's Provider Secrets. No secret belongs in Git, prompts or tool arguments.

Fixed `X-Gateway-App: sanare-msty` scopes the additional $5/day spend policy and
60 requests/minute policy. These are not a universal account/invoice ceiling:
unrelated apps and external paid tools remain outside their scope. Existing local
task reservations, usage accounting and stop checks remain authoritative too.
No automatic retry, model fallback, redirects or direct-provider fallback occurs
on Gateway failure. `0` is an explicit operator rollback, never a model action.
Unknown model identity or usage is rejected; an uncertain charge stays reserved.
Gateway does not plan tasks, execute MCP, train weights or trigger a council.

References: [native model access](https://docs.langchain.com/langsmith/llm-gateway-direct-model-access),
[header policies](https://docs.langchain.com/langsmith/llm-gateway-header-policies).
Offline coverage: `tests/unit_tests/test_msty_gateway.py`. Deployment and native
acceptance evidence live in the owning project's change journal, not this source
increment. Engine, sandboxes and subscription preferences are unchanged.

The `msty` graph now defaults to server profile `luna` (`gpt-5.6-luna`,
reasoning `none`). There is no compulsory prompt rewriter, council or hidden
second model. Each cloud run performs at most one billed generation; the explicit
compaction protocol above may require two separately counted runs in one client leg.
`MSTY_MODEL_PROFILE=sonnet` is an explicit operator rollback, not an automatic
expensive fallback. The old `MSTY_MODEL` variable is not a free-form selector.

An optional **text analyst**, `deepseek-flash` with thinking disabled, is invoked
only through the actual local `msty_brain_consult` MCP tool. This makes a separate
bounded request to the same budgeted `team.brain` endpoint with
`brain_task_role=analyst`. The role cannot receive tools, arbitrary model IDs or
endpoints: input is plain text up to 96 KB, output up to 2048 tokens. The MCP
facade additionally limits the explicit brief/evidence to 16000 characters and
rejects common secret patterns (not a complete DLP guarantee). The consultant
does not read paths, execute changes, browse or recursively delegate.

The primary Luna reads sources and performs authorized work through Msty's
existing MCP tools. It may ask for analysis/review when justified, then checks
the opinion against actual sources and observes the result of its own actions.
At most two issued consultations are counted in the native task checkpoint;
the ordinary 24-action limit still applies. Role changes inside a pending
callback are rejected. This is not a new background worker system and does not
reactivate retired teams, Paperclip jobs or `brain_supervisor`.

```text
Msty → existing gateway/stop/budget → LangGraph → Luna
  ↑          local MCP results / native resume       ↓
  └─ files, browser, memory, optional DeepSeek analysis
```

Validated results carry canonical model identity and measured usage; the local
ledger prices each request using a server-owned immutable profile snapshot.
Missing usage or identity is unknown, not zero. Token estimates exclude hosting
fees and are not provider invoices. No shared budget limit is changed here.

Context admission uses `msty-model-count-v1` (legacy flag accepted for rollout).
For Luna it uses the official tiktoken model mapping plus a documented safety
allowance: an admission estimate, not an exact provider count or billable usage.
DeepSeek uses a conservative text charge; budget-bound requests always run
admission counting. Sonnet retains its official token counter.

Luna user `image_url` blocks retain their full original URLs/base64 and detail
when sent to Chat Completions. Only the local counting copy substitutes their
payload with a marker. Per-image admission uses OpenAI's server-enforced patch
limits: `low` 256, `high` 2500, `auto`/`original` 30000, multiplied by 1.2 and
rounded up, plus one documented rounding token and 128 framing allowance.
This image component is an upper bound, not a dimension guess; the combined
text/schema/image value is still an **admission estimate**, not an exact count or
provider bill. Up to 32 images are accepted subject to the combined 180000 cap;
several small `auto` images can be conservatively rejected because their actual
dimensions are not fetched. The receipt says `tiktoken-image-envelope-v1`.
Unknown detail, malformed image blocks, DeepSeek images, and image blocks in
tool results fail closed. Current Chat Completions tool content supports text,
so a screenshot returned directly as a tool image is **not** enabled by this
change; a user image attachment is supported. No image download, extra API call,
expensive fallback or Responses migration is introduced. The Responses exact
input-token endpoint counts a different request format and is not claimed as an
exact counter for this route. Binary tool bundles are never serialized into a
text compaction prompt. Source: [OpenAI image-token rules](https://developers.openai.com/api/docs/guides/images-vision),
[Responses token counting](https://developers.openai.com/api/docs/guides/token-counting).
See `test_msty_image_envelope.py` for offline preservation and zero-generation
rejection checks; offline mocks do not establish native UI or paid API success.

No source text, rules or unknown blocks are silently discarded. The 180000
admission limit and 2 MB transport limit remain; memory is not unlimited.

The official MIT `langchain-openai==1.1.11` adapter and existing LangGraph native
interrupt/resume are reused. Sources: [LangChain adapter](https://github.com/langchain-ai/langchain/tree/master/libs/partners/openai),
[Luna API](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[DeepSeek thinking configuration](https://api-docs.deepseek.com/guides/thinking_mode/).
Deployment and live acceptance are recorded separately in the LLM project's
unified change journal. Offline passing tests alone do not prove deployment,
business completion or broad model quality. Earlier dated sections below describe
the inherited Sonnet implementation where their model/counting details differ.

## Msty Brain restoration — 19 September 2026

The `msty` graph is the main Msty route (`team.brain`), distinct from the local
`team.architect`. Msty executes local MCP tools and returns their real results;
the cloud graph does not independently access the Mac filesystem.

Tool policy is generated from each request's actual function schemas. With no
tools attached, the model must report unavailable execution, never fabricate
tool calls or file contents. Unsupported/malformed structured calls are rejected.
Textual hallucinations are not mechanically eliminated: live verification is
still required. Retired supervisor/council tools are not required by the policy.

Msty project LLM supplies its scoped instructions, Sanare Core and Knowledge
Stacks. Existing old conversations do not necessarily inherit changed project
defaults; use a fresh conversation for acceptance. Check actual tool results,
not just HTTP success or generated prose.

## Task continuity policy — 20 September 2026

`MSTY_TASK_CONTINUITY_V1` is included in the server-owned policy for every model
step, including existing Msty conversations and tool-result callbacks. A task
authorizes its ordinary necessary in-scope steps; the model must not ask for a
new command after each partial result. A minimal route is not a one-action limit.
Explicit stop, explanation/plan-only requests, unavailable tools, access and
budget/action limits still take priority. Unknown write outcomes require readback,
not blind retry. Once the requested outcome is verified, the task ends.

This continuity policy alone is a behavioral instruction, **not a deterministic completion controller**.
The newer bounded artifact-check gate is described above, separately from this
historical prompt-only change. No regex-based regeneration, permission expansion
or hidden background loop is added. Each cloud run returns one model
step for Msty's native tool loop. A text-only response remains a protocol-valid
answer, not proof of task completion. Tests in `test_msty_continuity.py` check
policy delivery and preserved user stop/history/tool choice, not model accuracy.
Real behavioral canaries and the old conversation's saved prompt need separate
verification. Old saved prompts are changed only through Msty GUI, not by editing
its obfuscated config or live database.

## Native checkpointed tool lifecycle — 20 September 2026

Requests using `execution_protocol=msty-local-tools-v1` use the installed
LangGraph 1.1.2 `interrupt` / `Command(resume=...)`, with Agent Server persistence.
After a schema-valid tool batch, a separate `wait_external` node records the
task, step, batch, guarded-result SHA256 and cumulative issued-action count.
That node performs no model call or external operation before interrupting.
The gateway publishes only a matching guarded result plus typed interrupt;
Msty continues to execute its own local MCP tools. A result callback resumes
the exact checkpoint/interrupt with the saved client-to-model call-ID mapping.
It does not resubmit the original request as a new graph run from START.

Resume validates task/batch, tools, output cap, original user turn and every
expected call/result. The checkpoint retains its original instructions and
history; only this batch's tool observations are appended. Root/RAG settings
changed while waiting are not silently substituted; they apply on a new user
turn. Canonical `b1_` client IDs are preserved in subsequent history. Provider
`length`/refusal responses cannot create executable pending calls. The 24-action
cap is cumulative in checkpoint state, not recomputed from shortened history.

Statuses `waiting_tools`, `answered`, `incomplete` and `blocked` describe the
execution protocol, **not verified completion of the user's business task**.
Tool-result text is still a client observation, not a signed action receipt.
Local SessionStore claims prevent duplicate publication/resume; uncertain
remote outcomes stay blocked for readback rather than being blindly retried.
Legacy requests with protocol `None` retain the old one-step behavior for
backwards-compatible rollout, not as fallback after a failed native resume.
The compatible gateway enforces the existing fresh global-stop projection
before dispatch and before publication; budgets, models and limits are unchanged.

Identity: the existing gateway uses explicit scoped chat IDs when supplied;
without one, initial requests remain isolated and callbacks reconnect through
their saved call IDs. This does not establish persistent identity across new
user turns that arrive without a chat ID. No unlimited context, autonomous
background task runner or weight training is implied.

Offline tests use real LangGraph checkpoints with fake inference. Recreating a
graph with the same in-memory saver tests resume semantics, not OS-crash recovery.
Live Agent Server/Msty verification and measured costs belong in the delivery
receipt. The [synthetic evaluator](tools/MSTY_EVAL.md) is an additional release
regression and optional LangSmith dataset, not a claim of general model quality.

Ready-made implementation selected (MIT, no additional dependencies):
[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence),
[interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts),
[source/license](https://github.com/langchain-ai/langgraph/blob/main/LICENSE).
Full DeepAgents migration and stock summarization middleware were not adopted:
they add unrelated execution semantics and do not by themselves provide safe
Msty history identity or failure-preserving compaction.

## Validated publication contract — 20 September 2026

The `msty` graph publishes one `custom` event after the model step and guards:
`{"type":"validated_result","message":<AIMessage.model_dump()>}`. The same
message remains in `State.result`; existing input/state fields are unchanged.
Controlled context-admission blockers use this contract too, without generation.
The bridge must consume `custom` + `values`, **never publish native `messages`
events**: those are unvalidated provider chunks and can precede rejection.
Text is intentionally buffered until the guarded final message is available.
Deploy the graph and compatible bridge together before accepting this contract.

Every structured call is checked against its current function's complete JSON
Schema using pinned `jsonschema` / `referencing`, including nested constraints,
required fields and local references. A declared supported draft is honored;
schemas without a dialect use draft 2020-12, and missing `parameters` means `{}`.
Built-in installed format checks are enabled; unknown format names remain
annotations as specified by JSON Schema. Unknown drafts, malformed schemas,
duplicate function names and unresolved references fail closed. An empty
reference registry prevents external HTTP/file retrieval; local `$defs` work.

Any unknown, malformed or schema-invalid call rejects the entire call batch,
replaces unverified prose with a visible blocker, and clears raw call fields.
Measured usage and response metadata are retained; there is no second LLM repair
call. Valid batches are returned unchanged for Msty to execute. Schema validity
is not authorization, action success or proof of textual truth: the local tool
executor still enforces access and checks actual results.

For a text-only final after the action cap, retain the current tool schemas and
history but send `tool_choice: {"type":"none"}`. Historical Anthropic
`tool_use`/`tool_result` blocks still require tool definitions. The graph also
normalizes OpenAI's string `"none"` to that dictionary (the installed LangChain
adapter otherwise treats it as a tool name), and rejects any new structured
call under explicit `none` before publication, even with schema-valid arguments.

Offline regression (no provider requests):
`uv run pytest tests/unit_tests/test_msty.py tests/unit_tests/test_msty_guards.py`.
The stream test exercises the actual `msty.graph.astream`, including a synthetic
stream that emits unsafe chunks before the guard; only the guarded message is
present in the custom publication channel.

## Optional five-minute static-prefix cache — 20 September 2026

`MSTY_STATIC_CACHE=0` disables this optimization; the default is on.
Only the first, gateway-owned `POLICY` plus current tool names receives
`cache_control={"type":"ephemeral","ttl":"5m"}`. Anthropic's prefix includes
preceding tool schemas. Msty project instructions, history and results receive
no new marker. Existing client cache controls are untouched, with no extra
breakpoint. Text/order, model, budgets, guards and zero-retry behavior stay the
same. Token preflight sees the same complete system blocks as generation.
No local content store, automatic history caching or warmup call is introduced.

For Sonnet 4.6, published USD/MTok rates are $3 uncached input, $3.75 five-minute
writes, $0.30 reads and $15 output. Cache expiry or prefix changes can therefore
make an isolated request more expensive. The minimum prefix is 1,024 tokens;
we neither guess tokens from characters nor pad prompts. Hits require actual
provider usage evidence; tests do not establish savings. See [pricing and
cache limits](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#pricing).

Privacy: this opts into provider-side ephemeral KV/hash storage, not a local
database or training. The five-minute lifetime refreshes on reuse; deletion
after expiry is not instantaneous. Cache isolation is workspace-scoped, not
per chat. No one-hour TTL is added. ZDR eligibility does not establish this
account's retention agreement. See [provider data-retention details](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#data-retention).

Accounting must be updated before deploying this version: SDK 1.3.5's
`usage_metadata.input_tokens` includes uncached input, reads and writes;
`total_tokens` adds output. Its `input_token_details` (singular) contains
`cache_read` and either generic `cache_creation` or `ephemeral_5m_input_tokens`
plus `ephemeral_1h_input_tokens`. When TTL detail is populated the SDK resets
generic creation to zero; do not charge it twice. Tests exercise that real SDK
conversion using synthetic usage only. Live cache hits and any measured benefit
belong in the deployment receipt, not inferred from unit tests.

## Context admission — 20 September 2026

The local bridge limits transport to 2 MB after bounded directory-tree previews;
it does not cut system instructions, conversation, ordinary file reads or tools.
The former 240 KB cap was a byte heuristic, not the model's context window.
For larger requests the bridge requires `context_budget: anthropic-count-v1`.
It clears this field and `context_budget_check` on every new request so persisted
thread state cannot reuse an earlier admission receipt.

Before generation, `msty` counts the **actual** policy + messages + tool schemas
with Anthropic's official token-counting API if serialized input exceeds 200 KB
or the flag is set. Admission is at most 180,000 input tokens, leaving room for
the existing 8,192 output cap and counting variance. Count failure/overflow
returns a visible blocker and does not call generation. Small requests do not
pay the counting round trip. No summarization, automatic retry, model swap or
budget increase is introduced.

Deploy this graph before reloading the updated bridge. The bridge validates a
fresh version1 admission receipt before forwarding tool calls on large requests.
Only byte/token counts, status and request IDs enter gateway diagnostics, never
raw prompts or tool contents. Token counting is not a provider invoice or a
guarantee of unlimited history. References: [token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting),
[transport limits](https://platform.claude.com/docs/en/api/errors).

Deployment template for a deep agent built with `create_deep_agent(...)`.

## What this template gives you

- A deployable deep agent graph at `src/deep_agent/graph.py`.
- Explicit workflow prompt (plan, delegate, critique, finalize).
- Two predefined sub-agents (`researcher`, `critic`).
- Human-in-the-loop interrupts on `execute` and `write_file`.
- A `uv`-managed local workflow with a small `Makefile` wrapper and starter tests.

## Prerequisites

- An API key for your model provider (Anthropic by default)
- A [LangSmith](https://smith.langchain.com/) account (Plus plan or higher) to deploy

## Quickstart

1. Sync the project and configure environment:

```bash
uv sync
cp .env.example .env
```

2. Start the dev server:

```bash
uv run langgraph dev
```

3. Deploy to LangSmith:

```bash
uv run langgraph deploy
```

See the [CLI docs](https://docs.langchain.com/langsmith/cli#deploy) for deploy options.

To set up CI instead, push this repo to GitHub and configure your deployment through the LangSmith UI.

## Tests and lint

```bash
make test
make integration-tests
make lint
make format
```

Integration tests are skipped unless `ANTHROPIC_API_KEY` is set.

## Reference docs

- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents quickstart: https://docs.langchain.com/oss/python/deepagents/quickstart
- LangSmith CLI: https://docs.langchain.com/langsmith/cli
