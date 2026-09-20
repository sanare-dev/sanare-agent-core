# Synthetic protocol evaluation — v1

This is a small adapter to the **installed official LangSmith SDK**, not a new
framework, model judge, agent, Engine, task runner or background process. No new
dependency. GitHub-first review selected the maintained MIT SDK and its ordinary
code-evaluator contract; the larger starter kit would add unrelated UI evaluator
and automation setup. The existing release runner is not copied or replaced.

Sources reviewed 2026-09-20:

- https://github.com/langchain-ai/langsmith-sdk/tree/main/python
- https://github.com/langchain-ai/langsmith-sdk/blob/main/LICENSE (MIT)
- https://docs.langchain.com/langsmith/code-evaluator-sdk
- https://docs.langchain.com/langsmith/manage-datasets-programmatically
- https://github.com/langchain-ai/langsmith-starter-kit (not installed)

## Offline use

`python tools/msty_eval.py` validates the fixed synthetic dataset and prints its
version/hash plus the evaluator code hash. It does not evaluate a model.
`tests/unit_tests/test_msty_eval.py` is picked up by existing
`python tools/verify_release.py`; no second release gate is needed. Evaluator
and dataset hashes are included in the new test's reviewed integrity contract,
because the existing release fingerprint otherwise excludes `tools/`.

Cases: simple answer, no tools, invalid schema, explicit stop/tool_choice none,
multi-step call/result structure, cross-thread rejection, truncated/length answer.
The stop case is a tool-choice/protocol test, not proof of global emergency-stop
enforcement. Existing graph/bridge regressions still test their implementations;
in particular cross-thread/length fixture evaluation is not a new live bridge test.

`protocol_evaluator(inputs, outputs, reference_outputs)` follows LangSmith's code
evaluator signature. It can later be supplied to `evaluate`/`aevaluate` by a
separately authorized caller; this tool does **not** run those APIs, upload runs,
configure cloud evaluators or launch live inference.

The target observation envelope has `schema_version=msty.protocol.observation.v1`,
`evidence_kind`, `steps`, `guard_blocked`, and optionally `final_text`. Each step
records `thread_id`, `finish_reason`, `tool_calls` (`id/name/args`) and callback
`tool_results` (`tool_call_id/name`). Cross-thread rejection also records
`attempted_thread_ids` and `cross_thread_rejected`.

Metrics distinguish `synthetic_mock` from `native_capture_unverified`. The score
means **structured protocol consistency**, never task completion or real tool
execution. Prose saying “done”, tool-result text and caller-supplied provenance
are not independent receipts. Native action verification is always reported
`not_performed`; model behavior quality is always `not_measured`. The adapter
does not authenticate or ingest native receipts. No native artifacts are imported.

## Optional synthetic-only delivery

External sync is **not run by tests or release verification**. A separate explicit
command is required, with an already configured `LANGSMITH_API_KEY`, matching
`LANGSMITH_WORKSPACE_ID`, and the exact reviewed hash printed by the manifest:

```sh
python tools/msty_eval.py sync --workspace-id EXISTING_WORKSPACE_UUID \
  --confirm-dataset-sha256 REVIEWED_DATASET_SHA256
```

Do not put a key on the command line. The adapter does not read `.env`, provider
key files, Msty SQLite/Tokens or project histories. Only the pinned fixture file
can be uploaded: there is no arbitrary input-file/run-import argument. Endpoint
allowlist is the existing official US/EU LangSmith API, with an explicit workspace
header. The official async SDK is lazy-created only after local checks; no retry,
prompt cache or tracing is configured by this adapter. Existing SDK credentials
are used only for that LangSmith destination; errors do not print SDK bodies.
In the installed SDK, `max_retries=1` means **one total HTTP attempt**, not one
retry; zero prevents any request. A real-SDK mock-transport regression verifies
that behavior instead of relying only on a fake client.

Dataset name contains version and content hash. Example IDs are deterministic per
workspace/version/hash/case. Existing exact examples are read back and skipped;
conflicting content or a name collision fails closed, never overwrites/deletes.
After an uncertain request, inspect/read back the same IDs before retrying. A
successful sync means seven synthetic examples verified remotely, **not** model
accuracy or action success. No experiment, feedback, judge or model call is made.
Platform storage charges are not measured and are not asserted to be zero.

The dataset/hash and evaluator/hash require review when changed. All production
deployment, stop/budget checks, live model tests and change-journal recording stay
with the existing controlled delivery process. This file grants no new authority.
