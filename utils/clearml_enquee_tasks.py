#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "clearml",
# ]
# ///

"""
Enquee a batch of tasks to a given ClearML queue.
"""

import argparse
import json
from typing import Iterable, Any
from clearml import Task
from clearml.automation import ClearmlJob


def enquee_tasks(
    base_task_id: str,
    tasks_num: int,
    queue_name: str,
    tags: Iterable[str] = None,
    parameters: dict[str, Any] = None,
) -> None:
    """
    Enqueue a batch of tasks to a given ClearML queue.

    This function creates and launches multiple ClearML jobs based on a base task,
    allowing for batch processing of tasks with optional parameter overrides and tags.

    Args:
        base_task_id (str): The ID of the base task to clone for each job.
        tasks_num (int): The number of tasks to enqueue.
        queue_name (str): The name of the ClearML queue to enqueue the tasks to.
        tags (Iterable[str], optional): A list of tags to add to each task. Defaults to an empty list.
        parameters (dict[str, Any], optional): A dictionary of parameter overrides for each task. Defaults to an empty dict.
    """

    tags = tags or []
    parameters = parameters or {}

    base_task = Task.get_task(task_id=base_task_id)
    for _ in range(tasks_num):
        job = ClearmlJob(
            base_task_id=base_task.id,
            parameter_override=parameters,
            tags=tuple(set(tags)),
        )
        job.launch(queue_name=queue_name)


if __name__ == "__main__":
    """
    Command-line interface for enqueuing tasks to a ClearML queue.

    Usage: python enquee_tasks.py <base_task_id> <tasks_num> <queue_name> [--tags TAG1 TAG2] [--parameters JSON_STRING]
    """
    parser = argparse.ArgumentParser(
        description="Enqueue a batch of tasks to a given ClearML queue."
    )
    parser.add_argument(
        "base_task_id", help="The ID of the base task to clone for each job."
    )
    parser.add_argument("tasks_num", type=int, help="The number of tasks to enqueue.")
    parser.add_argument(
        "queue_name", help="The name of the ClearML queue to enqueue the tasks to."
    )
    parser.add_argument("--tags", nargs="*", help="Optional tags to add to each task.")
    parser.add_argument(
        "--parameters", help="Optional parameter overrides as a JSON string."
    )

    args = parser.parse_args()

    parameters = json.loads(args.parameters) if args.parameters else {}

    enquee_tasks(
        base_task_id=args.base_task_id,
        tasks_num=args.tasks_num,
        queue_name=args.queue_name,
        tags=args.tags,
        parameters=parameters,
    )
