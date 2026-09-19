# Sanare Brain improvement contract

LangGraph does not silently retrain model weights. Improvement is an auditable
loop over traces, evaluations, and versioned graph changes:

1. LangSmith records deployment runs and latency/error traces.
2. A user correction or failed evaluation becomes a candidate rule with its
   source, scope, expected behavior, and a regression example.
3. A candidate is promoted only after the example passes and the change does not
   regress the direct-answer path.
4. Promoted changes are committed to this repository and deployed as a new
   revision. Rejected candidates remain evidence, not active instructions.

Production prompts and tools must never rewrite themselves from one conversation.
Secrets, raw private files, and full chat transcripts are not learning artifacts.

## Working release check

Run `python tools/verify_release.py` using the project's installed environment.
It runs the offline regression suite and checks that source files did not change
during validation. Its JSON result contains the verified source fingerprint.
GitHub Actions runs the same check on push and pull requests, without secrets or
paid model calls. This is a regression check, not proof of model quality.
The existing LangSmith build-on-push setting is independent of GitHub Actions:
do not claim a failed check technically blocks automatic deployment.

For each correction, add a regression example before changing the rule. Keep
evidence and decisions in the existing LLM project change journal. Test candidate
behavior with a synthetic task, compare errors/tool correctness and cost, and
only then publish. Reverting the exact Git commit provides a reversible release.

## Msty execution

The `msty` graph performs one bounded model step and returns native tool calls.
Msty executes the attached tools and sends their results back. It does not use
the cloud sandbox or hide local tool approvals inside a cloud interrupt.
Explicit chat/project IDs allow stable LangGraph threads. Clients that omit
those IDs use their complete supplied conversation as state; the bridge never
guesses identity from text. Cross-project retrieval remains the existing Msty
Knowledge Stack / project MCP responsibility.

Minimum routing evaluations:

- A one-word or simple factual request returns directly without subagents.
- A complex research request may use one researcher.
- A high-risk or explicitly requested review may use one critic.
- A council is used for explicit requests or consequential unresolved alternatives;
  the owner authorized adaptive delegation on 2026-09-19. Explain the reason first.
- A claimed external action must be backed by a real tool result.

## Adaptive supervisor (2026-09-19)

Use the existing Msty Admin MCP tools `msty_brain_delegate`, `msty_brain_job`,
and `msty_brain_lessons`. No additional framework or subscription is introduced.
This follows the tool-calling supervisor pattern recommended by
https://github.com/langchain-ai/langgraph-supervisor-py (MIT), not the older
mandatory multi-agent loop. One lead retains actual Msty execution tools.

Delegation is a consultation: worker selects DeepSeek Flash/Luna/GLM/Gemini/Qwen;
parallel gives DeepSeek implementation and Luna architecture/risk analysis;
review selects Luna or DeepSeek from a different family than the declared author;
council invokes the installed council via the existing budgeted gateway.
Two live delegation jobs maximum, six per task ID, no blind automatic retries.
Local-only is restricted to Qwen single-worker; no cloud fallback is requested.
Do not rename tasks to bypass limits. Job completion is not business completion.
Arbitrary external bots are not registered or claimed operational.

For substantive project work retrieve lessons. After a verified correction,
record an evidence-linked candidate in the existing primary change journal.
Candidates are reference material, not executable policy or permissions.
Before reuse check current applicability; before changing code add a regression
and independent review where justified. No weight training or automatic promotion
is claimed. Local MCP outcomes, not model narratives, are the evidence.
