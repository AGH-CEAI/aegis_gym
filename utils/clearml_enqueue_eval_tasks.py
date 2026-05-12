#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "clearml",
#   "tqdm",
# ]
# ///

import argparse
import logging
import sys
import time
from typing import Iterable

from clearml import Task
from clearml.automation import ClearmlJob
from tqdm import tqdm

from helpers.logging import timed, setup_logging

TASKS_NUM_SAFEGUARD_LIMIT = 100
TEMPLATE_PREFIX = "TEMPLATE_"
LOAD_BC_TASK_ID_KEY = "Args/load_bc_task_id"


@timed
def main():
    log = logging.getLogger(f"{__name__}")

    args = build_parser().parse_args()

    source_tasks = get_source_tasks(args.source_project, args.source_tags)
    if not source_tasks:
        log.error(f"No completed tasks found in '{args.source_project}'. Exiting.")
        sys.exit(1)

    if not len(source_tasks) < TASKS_NUM_SAFEGUARD_LIMIT:
        if not args.allow_large_batch:
            log.warning(
                f"You are trying to enqueue {len(source_tasks)} tasks. "
                f"The safeguard limit is {TASKS_NUM_SAFEGUARD_LIMIT} tasks. "
                "If you REALLY want to enqueue that much, please add the `--allow-large-batch` flag."
            )
            time.sleep(1)
            return

    eval_template = Task.get_task(task_id=args.eval_template_id)
    eval_project = eval_template.get_project_name()

    log.info(
        "Proceeding to enqueue:\n"
        f"\n> {'Eval template:':<16} {eval_template.task_id}"
        f"\n> {'Template name:':<16} {eval_template.name}"
        f"\n> {'Source project:':<16} {args.source_project}"
        f"\n> {'Eval project:':<16} {eval_project}"
        f"\n> {'Tasks found:':<16} {len(source_tasks)}"
        f"\n> {'Queue name:':<16} {args.queue_name}"
        f"\n> {'Tags:':<16} {args.tags}\n"
    )
    sleep_countdown(8)

    log.info("Adding tags to source tasks")

    enqueue_eval_tasks(
        eval_template=eval_template,
        source_tasks=source_tasks,
        eval_project=eval_project,
        queue_name=args.queue_name,
        tags=args.tags,
    )
    log.info(f"{len(source_tasks)} eval tasks enqueued to {args.queue_name} queue")


def get_source_tasks(project_name: str, tags: list[str]) -> list[Task]:
    log = logging.getLogger(f"{__name__}")

    task_filter = {
        "status": ["completed", "published"],
        "type": ["training", "testing", "inference", "application"],
        "system_tags": ["-archived"],
    }

    exclude_tags = ["-summary", "-template"]
    tags_filter = []
    if tags:
        tags_filter += ["__$all"] + list(tags)
    tags_filter += exclude_tags

    log.info(f"Fetching tasks from '{project_name}' (tags: {list(tags) or 'any'}) ...")
    tasks = Task.get_tasks(
        project_name=project_name,
        tags=tags_filter,
        task_filter=task_filter,
    )
    log.info(f"Found {len(tasks)} task(s)")
    return tasks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="For each completed training task in a source project, clone an eval template and enqueue."
    )
    p.add_argument(
        "--eval-template-id",
        required=True,
        help="ID of the eval template task to clone for each job",
    )
    p.add_argument(
        "--source-project",
        required=True,
        help="ClearML project containing the training runs to evaluate",
    )
    p.add_argument(
        "--queue-name",
        required=True,
        help="ClearML queue to enqueue the eval tasks to",
    )
    p.add_argument(
        "--source-tags",
        nargs="*",
        default=[],
        help="Filter source training tasks by tags",
    )
    p.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Tags to add to each created eval task",
    )
    p.add_argument(
        "--allow-large-batch",
        action="store_true",
        default=False,
        help="Disable safeguard mechanism for sending a large batch of runs",
    )
    p.add_argument(
        "--cleanup-previous-tags",
        action="store_true",
        default=False,
        help="Enable automatic remove of all `eval:XXX` tags from selected tasks.",
    )
    return p


def enqueue_eval_tasks(
    eval_template: Task,
    source_tasks: list[Task],
    eval_project: str,
    queue_name: str,
    cleanup_previous_tags=False,
    tag_for_source: str = "eval",
    tags: Iterable[str] = None,
) -> None:
    tags = tags or [tag_for_source]

    for train_task in tqdm(source_tasks, desc="Scheduling", unit="task"):
        job = ClearmlJob(
            base_task_id=eval_template.id,
            parameter_override={LOAD_BC_TASK_ID_KEY: train_task.id},
            task_overrides={
                "name": f"eval_{train_task.name}",
                "tags": list(set(tags)),
            },
        )
        job.task.move_to_project(new_project_name=eval_project)

        if cleanup_previous_tags:
            cleanup_task_tags(train_task, tag_for_source)
        train_task.add_tags([f"{tag_for_source}:{job.task.task_id}"])
        job.launch(queue_name=queue_name)


def cleanup_task_tags(t: Task, t_tag: str) -> tuple[int, int]:
    """Returns (tasks_cleaned, tags_removed) counts."""
    current_tags = list(t.get_tags() or [])
    filtered_tags = [tag for tag in current_tags if not tag.startswith(f"{t_tag}:")]

    if filtered_tags != current_tags:
        removed = [tag for tag in current_tags if tag.startswith(f"{t_tag}:")]
        t.set_tags(filtered_tags)
        return 1, len(removed)
    return 0, 0


def sleep_countdown(t_sleep: int) -> None:
    logger = logging.getLogger()

    for i in range(t_sleep, -1, -1):
        logger.info(f"{i} seconds remaining... ")
        sys.stdout.write("\r")
        sys.stdout.flush()
        time.sleep(1)

    logger.info("Proceeding...\n")
    sys.stdout.flush()


if __name__ == "__main__":
    setup_logging()
    log = logging.getLogger(f"{__name__}")
    try:
        main()
    except KeyboardInterrupt:
        pass
    log.info("Exiting")
