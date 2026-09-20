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

## Current routing and evidence (2026-09-20)

The main `msty` graph uses Luna; the separate demonstration `agent` graph is not
its supervisor. The retired `msty_brain_delegate`/job/council path MUST NOT be
restored from old instructions. `msty_brain_consult` is the optional bounded
DeepSeek Flash text analyst: maximum two consultations in the current native
task chain, no filesystem/browser/subagents of its own. Msty owns real MCP actions.
The local architect remains a separate, explicitly selected route.

Minimum acceptance scenarios:

- A simple one-word request returns directly, with zero tools or consultations.
- A multi-step synthetic task reads actual fixtures, creates an artifact and
  reads it back; a printed plan or fabricated tool narrative fails acceptance.
- A justified consultation preserves restrictions and receives only necessary
  non-secret evidence; its advice alone never proves execution.
- A blocked tool, budget exhaustion or a failed check cannot become success.
- Native Msty delivery is checked separately from a successful API request.
- Long-context compaction must preserve system/user instructions and unresolved
  tool pairs, record its own usage, and never substitute a summary for evidence.

No automatic council, expensive fallback, mandatory prompt rewriter, arbitrary
external bots or self-publishing agent is part of this route. Task-chain limits
do not by themselves identify every future user turn as one business project.

For substantive project work retrieve lessons. After a verified correction,
record an evidence-linked candidate using `msty_brain_lessons` in the existing
primary change journal. New receipt evidence must have an exact `project_slug`
(or exact `project`) match, a structured non-empty `verification` field, and an
intact recorded SHA-256. Failed checks are valid evidence for a proposed lesson;
they are NOT relabeled passed. Retrieval rechecks the source and quarantines
changed, missing or legacy-unbound evidence without erasing it.
Candidates are reference material, not executable policy or permissions.
Before reuse check current applicability; before changing code add a regression
and independent review where justified. No weight training or automatic promotion
is claimed. Local MCP outcomes, not model narratives, are the evidence. Structural
receipt checks and model tool observations do not establish semantic truth.

## Promotion gate

Use the existing release process, not a second self-update service:

1. Capture the failure as a redacted, project-bound primary receipt and candidate.
2. Add a reproducing offline regression; keep the negative case as well as the fix.
3. Run `tools/verify_release.py` and the affected gateway/MCP regressions.
4. Run only justified bounded synthetic live acceptance, within its explicit cap.
   Measure task result, tool correctness, latency and token-based cost separately.
5. Deploy the exact reviewed commit only after CI succeeds; record revision,
   rollback target and native Msty readback in the primary journal.

LangSmith tracing is already used; Engine, paid judges, Gateway and cloud Sandboxes
are separate optional products, not prerequisites for this loop. Engine analysis
requires its own approved budget. Synthetic protocol dataset delivery alone is
not a quality experiment. A lesson is not an instruction or new authorization.
