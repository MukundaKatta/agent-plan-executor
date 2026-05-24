"""Dependency-aware DAG executor for agent tool plans.

Define a plan as a set of :class:`PlanStep` objects, each with an optional
``depends_on`` list.  :class:`PlanExecutor` resolves a topological execution
order, surfaces steps whose dependencies are satisfied, and records results.

Example::

    from agent_plan_executor import PlanExecutor, PlanStep

    executor = PlanExecutor()
    executor.add_step(PlanStep("fetch", tool_name="http_get"))
    executor.add_step(PlanStep("parse", tool_name="json_parse", depends_on=["fetch"]))
    executor.add_step(PlanStep("store", tool_name="db_write", depends_on=["parse"]))
    executor.add_step(PlanStep("notify", tool_name="send_email", depends_on=["store"]))

    print(executor.execution_order())   # ['fetch', 'parse', 'store', 'notify']
    print(executor.ready_steps())       # ['fetch']

    executor.mark_done("fetch", result={"status": 200})
    print(executor.ready_steps())       # ['parse']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PlanStepStatus(str, Enum):
    """Lifecycle state of a single plan step."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


_TERMINAL_STATUSES = {
    PlanStepStatus.DONE,
    PlanStepStatus.FAILED,
    PlanStepStatus.SKIPPED,
}


class CyclicDependencyError(ValueError):
    """Raised when the dependency graph contains a cycle."""


class UnknownStepError(KeyError):
    """Raised when referencing a step_id that is not registered."""


@dataclass
class PlanStep:
    """A single step in an execution plan.

    Attributes:
        step_id:    Unique identifier for this step.
        tool_name:  Name of the tool/function to invoke.
        args:       Arguments to pass to the tool.
        depends_on: IDs of steps that must complete before this one runs.
        status:     Current :class:`PlanStepStatus`.
        result:     Value returned by the tool, set on completion.
        error:      Error message, set on failure.
    """

    step_id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: Any = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        """``True`` when the step has reached a final state."""
        return self.status in _TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "depends_on": list(self.depends_on),
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"PlanStep(step_id={self.step_id!r},"
            f" tool={self.tool_name!r},"
            f" status={self.status.value!r})"
        )


class PlanExecutor:
    """Execute a dependency-ordered plan of tool calls.

    Steps are organised as a DAG.  :meth:`execution_order` returns a
    topological sort; :meth:`ready_steps` returns the subset whose
    dependencies are all ``DONE``.

    Args:
        steps: Optional initial list of :class:`PlanStep` objects.

    Raises:
        :class:`CyclicDependencyError`: if the dependency graph is cyclic.
        :class:`UnknownStepError`: if a dependency references an unknown step.
    """

    def __init__(self, steps: list[PlanStep] | None = None) -> None:
        self._steps: dict[str, PlanStep] = {}
        self._order: list[str] = []  # insertion order
        for step in steps or []:
            self.add_step(step)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_step(self, step: PlanStep) -> None:
        """Register a step.

        Raises:
            ValueError: if *step_id* is already registered.
            :class:`UnknownStepError`: if any dependency is not registered.
        """
        if step.step_id in self._steps:
            raise ValueError(f"Step {step.step_id!r} is already registered")
        for dep in step.depends_on:
            if dep not in self._steps:
                raise UnknownStepError(
                    f"Dependency {dep!r} not registered"
                    f" (add it before {step.step_id!r})"
                )
        self._steps[step.step_id] = step
        self._order.append(step.step_id)

    def remove_step(self, step_id: str) -> None:
        """Remove a registered step.

        Raises:
            :class:`UnknownStepError`: if *step_id* is not registered.
            ValueError: if another registered step depends on *step_id*.
        """
        self._require(step_id)
        dependents = [
            s.step_id
            for s in self._steps.values()
            if step_id in s.depends_on
        ]
        if dependents:
            raise ValueError(
                f"Cannot remove {step_id!r}: steps {dependents} depend on it"
            )
        del self._steps[step_id]
        self._order.remove(step_id)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def steps(self) -> list[PlanStep]:
        """All steps in registration order."""
        return [self._steps[sid] for sid in self._order]

    def get_step(self, step_id: str) -> PlanStep | None:
        """Return the step, or ``None`` if not found."""
        return self._steps.get(step_id)

    def execution_order(self) -> list[str]:
        """Return step IDs in a valid topological execution order.

        Raises:
            :class:`CyclicDependencyError`: if the graph has a cycle.
        """
        return _topological_sort(self._steps)

    def ready_steps(self) -> list[str]:
        """Return step IDs that are PENDING and whose deps are all DONE."""
        result = []
        for sid in self._order:
            step = self._steps[sid]
            if step.status != PlanStepStatus.PENDING:
                continue
            if all(
                self._steps[dep].status == PlanStepStatus.DONE
                for dep in step.depends_on
            ):
                result.append(sid)
        return result

    def is_complete(self) -> bool:
        """``True`` when every step is in a terminal state."""
        return all(s.is_terminal for s in self._steps.values())

    def is_blocked(self) -> bool:
        """``True`` when non-terminal steps exist but none are ready or running.

        This usually means a failed dependency has blocked downstream steps.
        """
        if self.is_complete():
            return False
        running = any(
            s.status == PlanStepStatus.RUNNING for s in self._steps.values()
        )
        if running:
            return False
        return len(self.ready_steps()) == 0

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def mark_running(self, step_id: str) -> PlanStep:
        """Mark a step as RUNNING.

        Raises:
            :class:`UnknownStepError`: if *step_id* not found.
        """
        step = self._require(step_id)
        step.status = PlanStepStatus.RUNNING
        return step

    def mark_done(self, step_id: str, result: Any = None) -> PlanStep:
        """Mark a step as DONE with an optional result.

        Raises:
            :class:`UnknownStepError`: if *step_id* not found.
        """
        step = self._require(step_id)
        step.status = PlanStepStatus.DONE
        step.result = result
        return step

    def mark_failed(self, step_id: str, error: str = "") -> PlanStep:
        """Mark a step as FAILED with an optional error message.

        Raises:
            :class:`UnknownStepError`: if *step_id* not found.
        """
        step = self._require(step_id)
        step.status = PlanStepStatus.FAILED
        step.error = error
        return step

    def mark_skipped(self, step_id: str) -> PlanStep:
        """Mark a step as SKIPPED.

        Raises:
            :class:`UnknownStepError`: if *step_id* not found.
        """
        step = self._require(step_id)
        step.status = PlanStepStatus.SKIPPED
        return step

    def reset(self) -> None:
        """Reset all steps to PENDING, clearing results and errors."""
        for step in self._steps.values():
            step.status = PlanStepStatus.PENDING
            step.result = None
            step.error = None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of overall plan state."""
        counts: dict[str, int] = {s.value: 0 for s in PlanStepStatus}
        for step in self._steps.values():
            counts[step.status.value] += 1
        return {
            "total": len(self._steps),
            "is_complete": self.is_complete(),
            "is_blocked": self.is_blocked(),
            "counts": counts,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable full plan representation."""
        return {
            "steps": [self._steps[sid].to_dict() for sid in self._order],
            "summary": self.summary(),
        }

    def __repr__(self) -> str:
        n = len(self._steps)
        done = sum(1 for s in self._steps.values() if s.status == PlanStepStatus.DONE)
        return f"PlanExecutor(steps={n}, done={done})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require(self, step_id: str) -> PlanStep:
        if step_id not in self._steps:
            raise UnknownStepError(f"Step {step_id!r} not found")
        return self._steps[step_id]


def _topological_sort(steps: dict[str, PlanStep]) -> list[str]:
    """Kahn's algorithm — returns step IDs in topological order.

    Raises:
        :class:`CyclicDependencyError`: if the graph has a cycle.
    """
    # Build in-degree map
    in_degree: dict[str, int] = {sid: 0 for sid in steps}
    for step in steps.values():
        for dep in step.depends_on:
            in_degree[step.step_id] += 1

    # Queue starts with all zero-in-degree nodes (insertion-order stable)
    queue: list[str] = [sid for sid in steps if in_degree[sid] == 0]
    result: list[str] = []

    while queue:
        sid = queue.pop(0)
        result.append(sid)
        # Reduce in-degree of dependents
        for other in steps.values():
            if sid in other.depends_on:
                in_degree[other.step_id] -= 1
                if in_degree[other.step_id] == 0:
                    queue.append(other.step_id)

    if len(result) != len(steps):
        raise CyclicDependencyError(
            "Dependency graph contains a cycle; topological sort is impossible"
        )
    return result
