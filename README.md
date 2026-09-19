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
