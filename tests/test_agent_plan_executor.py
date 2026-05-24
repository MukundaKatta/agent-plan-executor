"""Tests for agent-plan-executor."""

from __future__ import annotations

import pytest

from agent_plan_executor import (
    CyclicDependencyError,
    PlanExecutor,
    PlanStep,
    PlanStepStatus,
    UnknownStepError,
)

# ---------------------------------------------------------------------------
# PlanStepStatus
# ---------------------------------------------------------------------------


def test_status_values():
    assert PlanStepStatus.PENDING.value == "pending"
    assert PlanStepStatus.RUNNING.value == "running"
    assert PlanStepStatus.DONE.value == "done"
    assert PlanStepStatus.FAILED.value == "failed"
    assert PlanStepStatus.SKIPPED.value == "skipped"


def test_status_is_str():
    assert isinstance(PlanStepStatus.DONE, str)


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------


def test_step_defaults():
    s = PlanStep("s1", "my_tool")
    assert s.status == PlanStepStatus.PENDING
    assert s.args == {}
    assert s.depends_on == []
    assert s.result is None
    assert s.error is None


def test_step_is_terminal_pending():
    s = PlanStep("s1", "t")
    assert not s.is_terminal


def test_step_is_terminal_running():
    s = PlanStep("s1", "t", status=PlanStepStatus.RUNNING)
    assert not s.is_terminal


def test_step_is_terminal_done():
    s = PlanStep("s1", "t", status=PlanStepStatus.DONE)
    assert s.is_terminal


def test_step_is_terminal_failed():
    s = PlanStep("s1", "t", status=PlanStepStatus.FAILED)
    assert s.is_terminal


def test_step_is_terminal_skipped():
    s = PlanStep("s1", "t", status=PlanStepStatus.SKIPPED)
    assert s.is_terminal


def test_step_to_dict():
    s = PlanStep("fetch", "http_get", args={"url": "https://example.com"})
    d = s.to_dict()
    assert d["step_id"] == "fetch"
    assert d["tool_name"] == "http_get"
    assert d["args"]["url"] == "https://example.com"
    assert d["status"] == "pending"
    assert d["result"] is None
    assert d["error"] is None


def test_step_repr():
    s = PlanStep("fetch", "http_get")
    r = repr(s)
    assert "fetch" in r
    assert "http_get" in r


# ---------------------------------------------------------------------------
# PlanExecutor — registration
# ---------------------------------------------------------------------------


def test_add_step_basic():
    ex = PlanExecutor()
    s = PlanStep("a", "tool_a")
    ex.add_step(s)
    assert ex.get_step("a") is s


def test_add_step_via_constructor():
    steps = [PlanStep("a", "t"), PlanStep("b", "t", depends_on=["a"])]
    ex = PlanExecutor(steps)
    assert len(ex.steps) == 2


def test_add_duplicate_raises():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    with pytest.raises(ValueError, match="already registered"):
        ex.add_step(PlanStep("a", "t"))


def test_add_unknown_dep_raises():
    ex = PlanExecutor()
    with pytest.raises(UnknownStepError):
        ex.add_step(PlanStep("b", "t", depends_on=["a"]))


def test_remove_step():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.remove_step("a")
    assert ex.get_step("a") is None


def test_remove_missing_step_raises():
    ex = PlanExecutor()
    with pytest.raises(UnknownStepError):
        ex.remove_step("nonexistent")


def test_remove_step_with_dependent_raises():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t", depends_on=["a"]))
    with pytest.raises(ValueError, match="depend on it"):
        ex.remove_step("a")


def test_steps_insertion_order():
    ex = PlanExecutor()
    ex.add_step(PlanStep("c", "t"))
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t"))
    assert [s.step_id for s in ex.steps] == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# PlanExecutor — execution_order (topological sort)
# ---------------------------------------------------------------------------


def test_execution_order_single():
    ex = PlanExecutor([PlanStep("a", "t")])
    assert ex.execution_order() == ["a"]


def test_execution_order_chain():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t", depends_on=["a"]))
    ex.add_step(PlanStep("c", "t", depends_on=["b"]))
    order = ex.execution_order()
    assert order.index("a") < order.index("b") < order.index("c")


def test_execution_order_diamond():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t", depends_on=["a"]))
    ex.add_step(PlanStep("c", "t", depends_on=["a"]))
    ex.add_step(PlanStep("d", "t", depends_on=["b", "c"]))
    order = ex.execution_order()
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_execution_order_parallel_roots():
    ex = PlanExecutor()
    ex.add_step(PlanStep("x", "t"))
    ex.add_step(PlanStep("y", "t"))
    ex.add_step(PlanStep("z", "t"))
    order = ex.execution_order()
    assert set(order) == {"x", "y", "z"}


def test_execution_order_cycle_raises():
    # Manually inject a cycle after construction
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t", depends_on=["a"]))
    # Force a back-edge to create a cycle
    ex._steps["a"].depends_on = ["b"]
    with pytest.raises(CyclicDependencyError):
        ex.execution_order()


# ---------------------------------------------------------------------------
# PlanExecutor — ready_steps
# ---------------------------------------------------------------------------


def test_ready_steps_no_deps():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t"))
    assert set(ex.ready_steps()) == {"a", "b"}


def test_ready_steps_with_pending_dep():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t", depends_on=["a"]))
    assert ex.ready_steps() == ["a"]


def test_ready_steps_after_dep_done():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t", depends_on=["a"]))
    ex.mark_done("a")
    assert ex.ready_steps() == ["b"]


def test_ready_steps_skips_non_pending():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.mark_running("a")
    assert ex.ready_steps() == []


# ---------------------------------------------------------------------------
# PlanExecutor — transitions
# ---------------------------------------------------------------------------


def test_mark_running():
    ex = PlanExecutor([PlanStep("a", "t")])
    step = ex.mark_running("a")
    assert step.status == PlanStepStatus.RUNNING


def test_mark_done_with_result():
    ex = PlanExecutor([PlanStep("a", "t")])
    step = ex.mark_done("a", result={"key": "val"})
    assert step.status == PlanStepStatus.DONE
    assert step.result == {"key": "val"}


def test_mark_failed_with_error():
    ex = PlanExecutor([PlanStep("a", "t")])
    step = ex.mark_failed("a", error="timeout")
    assert step.status == PlanStepStatus.FAILED
    assert step.error == "timeout"


def test_mark_skipped():
    ex = PlanExecutor([PlanStep("a", "t")])
    step = ex.mark_skipped("a")
    assert step.status == PlanStepStatus.SKIPPED


def test_transition_unknown_step_raises():
    ex = PlanExecutor()
    with pytest.raises(UnknownStepError):
        ex.mark_done("nope")


def test_reset_clears_all():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t"))
    ex.mark_done("a", result=42)
    ex.mark_failed("b", error="oops")
    ex.reset()
    for step in ex.steps:
        assert step.status == PlanStepStatus.PENDING
        assert step.result is None
        assert step.error is None


# ---------------------------------------------------------------------------
# PlanExecutor — is_complete / is_blocked
# ---------------------------------------------------------------------------


def test_is_complete_empty():
    ex = PlanExecutor()
    assert ex.is_complete()


def test_is_complete_all_done():
    ex = PlanExecutor([PlanStep("a", "t"), PlanStep("b", "t")])
    ex.mark_done("a")
    ex.mark_done("b")
    assert ex.is_complete()


def test_is_complete_with_pending():
    ex = PlanExecutor([PlanStep("a", "t"), PlanStep("b", "t")])
    ex.mark_done("a")
    assert not ex.is_complete()


def test_is_complete_failed_counts():
    ex = PlanExecutor([PlanStep("a", "t")])
    ex.mark_failed("a")
    assert ex.is_complete()


def test_is_complete_skipped_counts():
    ex = PlanExecutor([PlanStep("a", "t")])
    ex.mark_skipped("a")
    assert ex.is_complete()


def test_is_blocked_false_when_complete():
    ex = PlanExecutor([PlanStep("a", "t")])
    ex.mark_done("a")
    assert not ex.is_blocked()


def test_is_blocked_false_when_ready_steps():
    ex = PlanExecutor([PlanStep("a", "t")])
    assert not ex.is_blocked()


def test_is_blocked_true_when_dep_failed():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t", depends_on=["a"]))
    ex.mark_failed("a")
    assert ex.is_blocked()


def test_is_blocked_false_when_running():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t", depends_on=["a"]))
    ex.mark_running("a")
    assert not ex.is_blocked()


# ---------------------------------------------------------------------------
# PlanExecutor — summary / to_dict / repr
# ---------------------------------------------------------------------------


def test_summary_counts():
    ex = PlanExecutor()
    ex.add_step(PlanStep("a", "t"))
    ex.add_step(PlanStep("b", "t"))
    ex.add_step(PlanStep("c", "t"))
    ex.mark_done("a")
    ex.mark_failed("b")
    s = ex.summary()
    assert s["total"] == 3
    assert s["counts"]["done"] == 1
    assert s["counts"]["failed"] == 1
    assert s["counts"]["pending"] == 1


def test_to_dict_keys():
    ex = PlanExecutor([PlanStep("a", "t")])
    d = ex.to_dict()
    assert "steps" in d
    assert "summary" in d
    assert d["steps"][0]["step_id"] == "a"


def test_repr():
    ex = PlanExecutor([PlanStep("a", "t"), PlanStep("b", "t")])
    ex.mark_done("a")
    r = repr(ex)
    assert "PlanExecutor" in r
    assert "2" in r
