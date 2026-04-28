#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "clearml",
# ]
# ///

"""
Enqueue a batch of tasks to a given ClearML queue.
"""

import argparse
import json
import logging
import sys
import time
from typing import Iterable, Any

from clearml import Task
from clearml.automation import ClearmlJob

from helpers.logging import timed, setup_logging

TASKS_NUM_SAFEGUARD_LIMIT = 100


@timed
def main():
    """
    Command-line interface for enqueuing tasks to a ClearML queue.

    Usage: python enqueue_tasks.py <base_task_id> <tasks_num> <queue_name> [--tags TAG1 TAG2] [--parameters JSON_STRING]
    """
    log = logging.getLogger(f"{__name__}")

    args = build_parser().parse_args()
    parameters = json.loads(args.parameters) if args.parameters else {}

    if not args.tasks_num < TASKS_NUM_SAFEGUARD_LIMIT:
        log.warning(
            f"You are trying to enqueue {args.tasks_num} tasks. "
            f"The safeguard limit is {TASKS_NUM_SAFEGUARD_LIMIT} tasks. "
            "If you REALLY want to eqnue that much, please add the `--allow-large-batch` flag."
        )
        time.sleep(1)
        log.info("Exiting.")
        return

    log.info(
        "Proceeding to enquee:\n"
        f"\tBase task: \t{args.base_task_id}"
        f"\tNum of tasks: \t{args.tasks_num}"
        f"\tQueue name: \t{args.queue_name}"
        f"\tTags: \t{args.tags}"
        f"\tExtra parameters: \t{parameters}"
    )
    sleep_countdown(5)

    enqueue_tasks(
        base_task_id=args.base_task_id,
        tasks_num=args.tasks_num,
        queue_name=args.queue_name,
        tags=args.tags,
        parameters=parameters,
    )
    log.info(f"{args.tasks_num} tasks enqeued to {args.queue_name} queue.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Enqueue a batch of tasks to a given ClearML queue."
    )
    p.add_argument(
        "--base-task-id", help="The ID of the base task to clone for each job."
    )
    p.add_argument("--tasks-num", type=int, help="The number of tasks to enqueue.")
    p.add_argument(
        "--queue-name", help="The name of the ClearML queue to enqueue the tasks to."
    )
    p.add_argument("--tags", nargs="*", help="Optional tags to add to each task.")
    p.add_argument(
        "--parameters", help="Optional parameter overrides as a JSON string."
    )
    p.add_argument(
        "--allow-large-batch",
        action="store_true",
        default=False,
        help="Disable safeguard mechanism for sending a large batch of runs.",
    )
    return p


def enqueue_tasks(
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


def sleep_countdown(t_sleep: int) -> None:
    logger = logging.getLogger()

    # Countdown loop with logging
    for i in range(t_sleep, -1, -1):
        logger.info(f"{i} seconds remaining... ")
        sys.stdout.write("\r")
        sys.stdout.flush()
        time.sleep(1)

    # Completion message
    logger.info("Proceeding...\n")
    sys.stdout.flush()


if __name__ == "__main__":
    setup_logging()
    log = logging.getLogger(f"{__name__}")
    main()
    log.info("Exiting")
