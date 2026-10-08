# Native Graph: catalogue after an external batch

The external pending batch is validated against its original schemas, task ID,
batch ID, tool IDs and arguments. `msty_execution.validate_resume` accepts a
separate gateway-staged catalogue only after validating the whole pending batch.

The native middleware must carry the **validated** `resumed.tools` into its next
checkpoint update and model step. It must not substitute `response.input.tools`
or reset the task, action counters, native counters, output cap or owner scope.

The live component request `ade31ff9-3bc7-4c2e-88e6-a17522f52875` on Desk
2026.10.08-20 / 4c7bd04d7ce3 completed `skills_list` but answered without the
required `skills_get`. Its exact owned cloud checkpoint retained only
`skills_list`; the bridge had already pinned both tools. The delivered Graph
source 8881eae6 validated the new catalogue but omitted `tools` from native
middleware continuation fields. This correction forwards that validated field.

`tests/unit_tests/test_native_catalog_refresh.py` exercises the real native
graph with checkpoint recreation and a scripted model boundary. It checks a
new tool and an updated schema reach the next model and persisted state, while
task ID, counters and output cap remain unchanged. A forged catalogue hash
must fail before the second model step. Network access is disabled in tests.

Offline results prove this continuation boundary. They do not prove an active
cloud deployment, image generation, browser operation or whole-task completion.
The failed live request and its one-shot intent are retained; they are not reset
or retried. A new live request requires reviewed delivery, fresh stop/budget and
idle gates, and explicit authority. Source application and cloud deployment
follow the existing Claude-only route; no protection or budget is relaxed.
