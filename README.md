# Deep Agents Template

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
