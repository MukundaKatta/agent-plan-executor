# agent-plan-executor

Dependency-aware DAG executor for agent tool plans. Zero dependencies.

## Install

```bash
pip install agent-plan-executor
```

## Quick start

```python
from agent_plan_executor import PlanExecutor, PlanStep

executor = PlanExecutor()
executor.add_step(PlanStep("fetch",  tool_name="http_get"))
executor.add_step(PlanStep("parse",  tool_name="json_parse", depends_on=["fetch"]))
executor.add_step(PlanStep("store",  tool_name="db_write",   depends_on=["parse"]))
executor.add_step(PlanStep("notify", tool_name="send_email", depends_on=["store"]))

print(executor.execution_order())  # ['fetch', 'parse', 'store', 'notify']
print(executor.ready_steps())      # ['fetch']

executor.mark_running("fetch")
executor.mark_done("fetch", result={"status": 200, "body": "..."})

print(executor.ready_steps())      # ['parse']
```

## API

### `PlanStep`

| Attribute | Description |
|---|---|
| `step_id` | Unique step identifier |
| `tool_name` | Name of the tool/function to invoke |
| `args` | Arguments dict for the tool |
| `depends_on` | List of step IDs that must be DONE first |
| `status` | Current `PlanStepStatus` |
| `result` | Tool result (set on completion) |
| `error` | Error message (set on failure) |

### `PlanExecutor`

| Method | Description |
|---|---|
| `add_step(step)` | Register a step (deps must be registered first) |
| `remove_step(step_id)` | Unregister a step (fails if another step depends on it) |
| `execution_order()` | Topological sort of all step IDs |
| `ready_steps()` | Step IDs whose deps are all DONE and status is PENDING |
| `mark_running(step_id)` | Transition step to RUNNING |
| `mark_done(step_id, result)` | Transition step to DONE |
| `mark_failed(step_id, error)` | Transition step to FAILED |
| `mark_skipped(step_id)` | Transition step to SKIPPED |
| `reset()` | Reset all steps to PENDING |
| `is_complete()` | `True` when all steps are terminal |
| `is_blocked()` | `True` when non-terminal steps exist but none are ready |
| `summary()` | Status counts and completion flags |

### Errors

| Exception | Raised when |
|---|---|
| `CyclicDependencyError` | Dependency graph has a cycle |
| `UnknownStepError` | Referencing an unregistered step ID |

## License

MIT
