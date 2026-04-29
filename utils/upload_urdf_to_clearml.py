#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "clearml",
# ]
# ///
"""
Upload robot simulator assets to ClearML Dataset.
Handles URDF + STL/DAE models as a versioned dataset.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from clearml import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_robot_assets(
    robot_folder_path: str,
    dataset_name: str = "robot_simulator_assets",
    dataset_project: str = "DeepRL",
    parent: str = None,
    output_storage: Optional[str] = None,  # None = use default ClearML file server
    description: Optional[str] = None,
):
    """
    Upload robot simulator folder to ClearML Dataset.

    Args:
        robot_folder_path: Path to folder containing URDF + models (STL/DAE)
        dataset_name: Name of the dataset (version auto-incremented)
        dataset_project: Project name in ClearML
        output_storage: Target storage (None=default fileserver, or "/path", "s3://bucket", etc.)
        description: Optional dataset description
    """

    folder_path = Path(robot_folder_path).resolve()

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {robot_folder_path}")

    logger.info(f"> Creating dataset: {dataset_project}/{dataset_name}")
    if parent:
        logger.info(f"> Parent DatasetID: {parent}")
    logger.info(f"> Source folder: {folder_path}")
    logger.info(f"> Files to upload: {len(list(folder_path.rglob('*')))}")

    # Create dataset (automatically increments version if exists)
    parent_list = [parent] if parent else None
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        parent_datasets=parent_list,
        description=description or "Robot simulator assets",
    )

    logger.info(f"> Dataset created: {dataset.id}")

    # Add all files from folder (preserves structure)
    logger.info("> Adding files to dataset...")
    num_files = dataset.add_files(path=str(folder_path), recursive=True, verbose=True)

    logger.info(f"> Added {num_files} files")

    # Upload to storage
    logger.info("> Uploading to ClearML server...")
    dataset.upload(
        show_progress=True,
        output_url=output_storage,  # Uses default if None
    )

    logger.info("> Upload complete")

    # Finalize (lock version, prevents further modifications)
    logger.info("> Finalizing dataset...")
    dataset.finalize()

    logger.info("\n> SUCCESS!")
    logger.info(f"> Dataset ID: {dataset.id}")
    logger.info(f"> Project: {dataset_project}")
    logger.info(f"> Name: {dataset_name}")
    logger.info(f"> Use in code: Dataset.get(dataset_id='{dataset.id}')")

    return dataset.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload robot assets to ClearML")
    parser.add_argument("folder", help="Path to robot model folder")
    parser.add_argument("--name", default="AegisURDFModel", help="Dataset name")
    parser.add_argument("--project", default="AEGIS_GRASP", help="ClearML project")
    parser.add_argument(
        "--parent",
        default=None,
        help="Already existing dataset ID which will be the parent of this version.",
    )
    parser.add_argument(
        "--storage", default=None, help="Storage URL (s3://, /path, etc.)"
    )
    parser.add_argument("--desc", default=None, help="Dataset description")

    args = parser.parse_args()

    upload_robot_assets(
        robot_folder_path=args.folder,
        dataset_name=args.name,
        dataset_project=args.project,
        parent=args.parent,
        output_storage=args.storage,
        description=args.desc,
    )
