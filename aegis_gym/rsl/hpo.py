from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    UniformParameterRange,
    UniformIntegerParameterRange,
    DiscreteParameterRange,
)
from clearml.automation.optuna import OptimizerOptuna


# USER CONFIG

# ID of the base task to optimize (must already exist in ClearML)
BASE_TASK_ID = "ID"

OBJECTIVE_TITLE = "Train/mean_reward"
OBJECTIVE_SERIES = "series"
OBJECTIVE_SIGN = "max"

# Optimization strategy: "optuna", "bohb", "random", "grid"
OPTIMIZER_TYPE = "optuna"

# ClearML queue where experiments will be executed
EXECUTION_QUEUE = "geonosis"

# Maximum number of experiments running at the same time
MAX_CONCURRENT_TASKS = 1

# Total number of experiments to run
# Set to None to run indefinitely
TOTAL_MAX_JOBS = 10

# Maximum number of training iterations per experiment
# Overrides the base task configuration
MAX_ITERATIONS_PER_JOB = 1000

# How often (in minutes) the optimizer checks job status
POOL_PERIOD_MIN = 0.2

# How often (in minutes) to print progress logs
# Only affects logging, not optimization behavior
REPORT_PERIOD_MIN = 1


# HYPERPARAMETERS

# Define hyperparameters to optimize
HYPER_PARAMETERS = [
    UniformParameterRange(
        "alg_cfg/learning_rate",
        min_value=1e-5,
        max_value=3e-4,
    ),
    UniformIntegerParameterRange(
        "alg_cfg/num_learning_epochs",
        min_value=3,
        max_value=8,
        step_size=1,
    ),
    DiscreteParameterRange("policy_cfg/activation", values=["relu", "elu"]),
]

# OPTIMIZER SELECTION

if OPTIMIZER_TYPE == "optuna":
    optimizer_class = OptimizerOptuna
elif OPTIMIZER_TYPE == "bohb":
    from clearml.automation.hpbandster import OptimizerBOHB

    optimizer_class = OptimizerBOHB
elif OPTIMIZER_TYPE == "random":
    from clearml.automation import RandomSearch

    optimizer_class = RandomSearch
elif OPTIMIZER_TYPE == "grid":
    from clearml.automation import GridSearch

    optimizer_class = GridSearch
else:
    raise ValueError(f"Unknown OPTIMIZER_TYPE: {OPTIMIZER_TYPE}")

# CALLBACK


def job_complete_callback(
    job_id, objective_value, objective_iteration, job_parameters, top_performance_job_id
):
    print("Job completed:", job_id)
    print("score:", objective_value)
    print("params:", job_parameters)

    if job_id == top_performance_job_id:
        print("NEW BEST MODEL")


# MAIN

task = Task.init(
    project_name="HPO",
    task_name="hyperparameter_optimization",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)

optimizer = HyperParameterOptimizer(
    base_task_id=BASE_TASK_ID,
    hyper_parameters=HYPER_PARAMETERS,
    objective_metric_title=OBJECTIVE_TITLE,
    objective_metric_series=OBJECTIVE_SERIES,
    objective_metric_sign=OBJECTIVE_SIGN,
    optimizer_class=optimizer_class,
    execution_queue=EXECUTION_QUEUE,
    max_number_of_concurrent_tasks=MAX_CONCURRENT_TASKS,
    total_max_jobs=TOTAL_MAX_JOBS,
    max_iteration_per_job=MAX_ITERATIONS_PER_JOB,
    pool_period_min=POOL_PERIOD_MIN,
)

optimizer.set_report_period(REPORT_PERIOD_MIN)

optimizer.start(job_complete_callback=job_complete_callback)

optimizer.wait()

# Number of best experiments to return (sorted by objective metric)
top_exp = optimizer.get_top_experiments(top_k=3)

print("Top experiments:")
print([t.id for t in top_exp])

optimizer.stop()

print("HPO finished")
