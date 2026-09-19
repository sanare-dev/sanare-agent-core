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

Minimum routing evaluations:

- A one-word or simple factual request returns directly without subagents.
- A complex research request may use one researcher.
- A high-risk or explicitly requested review may use one critic.
- A council is used only when the user explicitly asks for a council or vote.
- A claimed external action must be backed by a real tool result.
