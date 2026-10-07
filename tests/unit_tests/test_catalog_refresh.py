"""Real checkpoint/resume, fake inference; no external actions."""

import json
import asyncio
from copy import deepcopy
import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from deep_agent import msty_execution as e

TASK = "11111111-1111-4111-8111-111111111111"
OLD = [
    {
        "type": "function",
        "function": {
            "name": "read_fixture",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
NEW = OLD + [
    {
        "type": "function",
        "function": {
            "name": "new_fixture",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def state():
    s = {
        "messages": [{"role": "user", "content": "Build and verify fixture."}],
        "tools": deepcopy(OLD),
        "max_tokens": 64,
        "tool_choice": "auto",
        "result": {
            "content": "Inspect",
            "tool_calls": [{"id": "m1", "name": "read_fixture", "args": {}}],
        },
        "context_budget": None,
        "context_budget_check": None,
        "execution_protocol": e.PROTOCOL,
        "execution_task_id": TASK,
        "execution": {},
    }
    s["execution"] = e.execution_after(s, {"result": s["result"]})
    return s


def resume(s, refresh=True):
    incoming = {
        k: deepcopy(s[k])
        for k in (
            "messages",
            "tools",
            "max_tokens",
            "tool_choice",
            "context_budget",
            "execution_protocol",
            "execution_task_id",
        )
    }
    incoming.update(result={}, context_budget_check=None)
    calls = []
    results = []
    mapping = []
    for i, c in enumerate(s["execution"]["pending"]["calls"]):
        cid = "b1_test_" + str(s["execution"]["step"]) + "_" + str(i)
        mapping.append({"client_id": cid, "model_id": c["id"]})
        calls.append(
            {
                "id": cid,
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
            }
        )
        results.append(
            {"role": "tool", "tool_call_id": cid, "content": "Observed fixture result"}
        )
    incoming["messages"] += [
        {"role": "assistant", "content": s["result"]["content"], "tool_calls": calls},
        *results,
    ]
    r = {
        "version": 1,
        "task_id": TASK,
        "batch_id": s["execution"]["pending"]["batch_id"],
        "tool_id_map": mapping,
        "input": incoming,
    }
    if refresh:
        r["catalog_refresh"] = {
            "version": 1,
            "task_id": TASK,
            "batch_id": r["batch_id"],
            "previous_sha256": e.canonical_digest(OLD),
            "next_sha256": e.canonical_digest(NEW),
            "tools": deepcopy(NEW),
        }
    return r


@pytest.mark.parametrize("catalog_size", [1, 234])
def test_full_checkpoint_refresh_new_tool_and_same_task(monkeypatch, catalog_size):
    old = OLD + [
        {
            "type": "function",
            "function": {
                "name": "fixture_" + str(i),
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(catalog_size - 1)
    ]
    monkeypatch.setitem(globals(), "OLD", old)
    monkeypatch.setitem(globals(), "NEW", old + [NEW[-1]])
    e.msty_models.check_tools(OLD)
    e.msty_models.check_tools(NEW)

    async def run():
        seen = []

        def respond(s):
            seen.append([x["function"]["name"] for x in s["tools"]])
            if len(seen) == 1:
                out = {
                    "content": "inspect",
                    "tool_calls": [{"id": "m1", "name": "read_fixture", "args": {}}],
                }
            elif len(seen) == 2:
                assert seen[-1] == [x["function"]["name"] for x in NEW]
                out = {
                    "content": "verify new tool",
                    "tool_calls": [{"id": "m2", "name": "new_fixture", "args": {}}],
                }
            else:
                out = {"content": "Fixture verified", "tool_calls": []}
            return {
                **s,
                "result": out,
                "execution": e.execution_after(s, {"result": out}),
            }

        g = StateGraph(dict)
        g.add_node("respond", respond)
        g.add_node("wait_external", e.wait_external)
        g.add_edge(START, "respond")
        g.add_conditional_edges(
            "respond", e.next_node, {"wait_external": "wait_external", "__end__": END}
        )
        g.add_edge("wait_external", "respond")
        check = InMemorySaver()
        graph = g.compile(checkpointer=check)
        cfg = {"configurable": {"thread_id": "refresh"}}
        initial = state()
        initial["execution"] = {}
        initial["result"] = {}
        first = await graph.ainvoke(initial, cfg)
        graph = g.compile(checkpointer=check)  # real restart/resume
        second = await graph.ainvoke(Command(resume=resume(first)), cfg)
        final = await graph.ainvoke(Command(resume=resume(second, False)), cfg)
        assert len(seen) == 3 and final["execution"]["task_id"] == TASK
        assert (
            final["execution"]["actions_issued"] == 2
            and final["execution"]["step"] == 3
        )
        assert (
            final["execution"]["status"] == "answered"
            and final["execution"]["pending"] is None
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "case",
    [
        "missing_result",
        "wrong_owner",
        "changed_args",
        "changed_tools",
        "wrong_task",
        "wrong_batch",
        "wrong_previous",
        "wrong_next",
        "duplicate_tool",
        "over_cap",
        "wrong_role",
        "higher_limit",
    ],
)
def test_reject_invalid_refresh_before_next_inference(case):
    s = state()
    r = resume(s)
    if case == "missing_result":
        r["input"]["messages"].pop()
    if case == "wrong_owner":
        r["input"]["messages"][0]["content"] = "Different task"
    if case == "changed_args":
        r["input"]["messages"][-2]["tool_calls"][0]["function"]["arguments"] = (
            '{"other":1}'
        )
    if case == "changed_tools":
        r["input"]["tools"] = NEW
    if case == "wrong_task":
        r["catalog_refresh"]["task_id"] = "22222222-2222-4222-8222-222222222222"
    if case == "wrong_batch":
        r["catalog_refresh"]["batch_id"] = "22222222-2222-4222-8222-222222222222"
    if case == "wrong_previous":
        r["catalog_refresh"]["previous_sha256"] = "0" * 64
    if case == "wrong_next":
        r["catalog_refresh"]["next_sha256"] = "0" * 64
    if case == "duplicate_tool":
        r["catalog_refresh"]["tools"] = OLD + OLD
    if case == "over_cap":
        r["catalog_refresh"]["tools"] = [
            {
                "type": "function",
                "function": {"name": "fixture_" + str(i), "parameters": {}},
            }
            for i in range(257)
        ]
    if case == "wrong_role":
        s["brain_task_role"] = "analyst"
        r["input"]["brain_task_role"] = "analyst"
    if case == "higher_limit":
        r["input"]["max_tokens"] = 65
    with pytest.raises(e.ExecutionProtocolError):
        e.validate_resume(s, r)
    assert (
        s["execution"]["actions_issued"] == 1 and s["execution"]["pending"] is not None
    )


def test_no_refresh_unchanged_old_contract():
    s = state()
    out = e.validate_resume(s, resume(s, False))
    assert out["tools"] == OLD and out["execution"]["task_id"] == TASK
    assert (
        out["execution"]["actions_issued"] == 1 and out["execution"]["pending"] is None
    )


def test_refresh_uses_existing_admission_boundary():
    s = state()
    r = resume(s)
    tools = [
        {
            "type": "function",
            "function": {"name": "fixture_" + str(i), "parameters": {}},
        }
        for i in range(256)
    ]
    assert e.msty_models.MAX_TOOLS == 256
    e.msty_models.check_tools(tools)
    r["catalog_refresh"].update(tools=tools, next_sha256=e.canonical_digest(tools))
    assert len(e.validate_resume(s, r)["tools"]) == 256
    tools[0]["function"]["parameters"]["bad"] = float("nan")
    with pytest.raises(e.ExecutionProtocolError):
        e.validate_resume(s, r)
