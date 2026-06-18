from clearml import PipelineController

# Baseline tasks to clone (must exist, must not be in 'failed' state)
RL_BASE_PROJECT = "PROJECT/rl"
RL_BASE_TASK = "rl_baseline"
BC_BASE_PROJECT = "PROJECT/bc"
BC_BASE_TASK = "bc_baseline"

# Metric used to compare RL runs
METRIC_TITLE = "Train/mean_reward"
METRIC_SERIES = "series"

PIPELINE_PROJECT = "PROJECT/pipeline"
QUEUE = "geonosis"
N_RL_RUNS = 10

TASK_OVERRIDES = {
    "container.image": "ghcr.io/agh-ceai/aegis_gym:v0.0.2",
    "container.arguments": "--env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1",
    # "script.branch": "main",
    # "script.version_num": "",
}


def select_best_rl(metric_title: str, metric_series: str, **rl_task_kwargs):
    """Pick the RL run with the highest metric value and expose its IDs."""
    from clearml import Task

    task_ids = [
        rl_task_kwargs[k]
        for k in sorted(rl_task_kwargs, key=lambda s: int(s.rsplit("_", 1)[-1]))
    ]

    best_task_id, best_value = None, float("-inf")
    for tid in task_ids:
        task = Task.get_task(task_id=tid)
        metrics = task.get_last_scalar_metrics()
        value = metrics.get(metric_title, {}).get(metric_series, {}).get("last")
        print(f"task={tid} {metric_title}/{metric_series}={value}")
        if value is not None and value > best_value:
            best_task_id, best_value = tid, value

    if best_task_id is None:
        raise RuntimeError(f"No RL run reports metric {metric_title}/{metric_series}")

    best_task = Task.get_task(task_id=best_task_id)
    output_models = list(best_task.models.get("output", []))
    best_model_id = ""
    if output_models:
        named_best = [m for m in output_models if "best" in m.name.lower()]
        chosen = named_best[-1] if named_best else output_models[-1]
        best_model_id = chosen.id
        print(f"checkpoint: name={chosen.name} id={best_model_id}")

    Task.current_task().set_parameters(
        {"Args/best_task_id": best_task_id, "Args/best_model_id": best_model_id}
    )

    print(f"best run: task={best_task_id} value={best_value}")
    return best_task_id, best_model_id


pipe = PipelineController(
    name="rl_to_bc",
    project=PIPELINE_PROJECT,
    version="0.1.0",
    add_pipeline_tags=True,
)
pipe.set_default_execution_queue(QUEUE)

pipe.add_parameter(name="rl_max_iterations", default=10)
pipe.add_parameter(name="bc_max_iterations", default=5)
pipe.add_parameter(name="num_envs", default=256)

rl_steps = []
for i in range(N_RL_RUNS):
    step = f"train_rl_{i}"
    rl_steps.append(step)
    pipe.add_step(
        name=step,
        base_task_project=RL_BASE_PROJECT,
        base_task_name=RL_BASE_TASK,
        parameter_override={
            "Args/exp_name": f"pipe_rl_run{i}",
            "Args/max_iterations": "${pipeline.rl_max_iterations}",
            "Args/num_envs": "${pipeline.num_envs}",
        },
        task_overrides=dict(TASK_OVERRIDES),
        monitor_metrics=[(METRIC_TITLE, METRIC_SERIES)],
    )

# Select the best RL run
pipe.add_function_step(
    name="select_best",
    parents=rl_steps,
    function=select_best_rl,
    function_kwargs=dict(
        metric_title=METRIC_TITLE,
        metric_series=METRIC_SERIES,
        **{f"rl_task_id_{i}": f"${{train_rl_{i}.id}}" for i in range(N_RL_RUNS)},
    ),
    function_return=["best_task_id", "best_model_id"],
)

# BC training on the best teacher (clone of the BC baseline task)
pipe.add_step(
    name="train_bc",
    parents=["select_best"],
    base_task_project=BC_BASE_PROJECT,
    base_task_name=BC_BASE_TASK,
    parameter_override={
        "Args/load_rl_task_id": "${select_best.parameters.Args/best_task_id}",
        "Args/load_rl_model_id": "${select_best.parameters.Args/best_model_id}",
        "Args/exp_name": "pipe_bc",
        "Args/max_iterations": "${pipeline.bc_max_iterations}",
        "Args/num_envs": "${pipeline.num_envs}",
    },
    task_overrides=dict(TASK_OVERRIDES),
)

if __name__ == "__main__":
    pipe.start_locally()
    print("pipeline finished")
