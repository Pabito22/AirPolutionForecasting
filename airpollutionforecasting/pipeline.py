"""Orchestrator for each pipeline steps."""

from typing import Literal
from pathlib import Path
import pandas

from airpollutionforecasting.data.loader import DataLoader
from airpollutionforecasting.data.data_validator import DataValidator
from airpollutionforecasting.data.splitter import DataSplitter
from airpollutionforecasting.config import DatasetConfig, FeatureConfig
from airpollutionforecasting.features.feature_creator import FeatureCreator
from airpollutionforecasting.utils import load_data_from_path

from loguru import logger


def run_pipeline(
    stop_step: Literal["load", "validate", "split", "feature"] = "split",
) -> None | pandas.DataFrame | list[Path]:
    """Run pipeline till the stop_step

    Args:
     stop_step: Allowed values["load", "validate", "split"]
        load - loads the data
        validate - validates if the data is clean
        split - splits data into train, validation and test sets.

    Returns result of the last stop.
    """
    logger.info(f"Running pipeline until {stop_step}...")

    data_config = DatasetConfig.load_config()
    feature_config = FeatureConfig.load_config()

    # 1. Load
    loader = DataLoader(data_config=data_config)
    loaded_data = loader.load()

    if stop_step == "load":
        return loaded_data

    # 2. Validate
    validator = DataValidator(data_config=data_config)
    validator.validate(loaded_data=loaded_data)

    if stop_step == "validate":
        return None

    # 3. Split
    splitter = DataSplitter(data_config=data_config)
    split_files = splitter.split(loaded_data=loaded_data)

    logger.info(f"Data successfully split: {split_files}")

    if stop_step == "split":
        return split_files

    # 4. Feature creation

    logger.info("Creating features based on the config..")

    # load the data produced by split step
    train_data = load_data_from_path(data_path=split_files[0], datetime_cols=["time"])
    val_data = load_data_from_path(data_path=split_files[1], datetime_cols=["time"])
    test_data = load_data_from_path(data_path=split_files[2], datetime_cols=["time"])

    # create the features as in configuration
    feature_creator = FeatureCreator(
        data_config=data_config, feature_config=feature_config
    )

    feature_files: list[Path] = [
        feature_creator.create(train_data, "train"),
        feature_creator.create(val_data, "val"),
        feature_creator.create(test_data, "test"),
    ]

    # TODO: make other steps utilize the new saving files functionality
    # TODO: test your solution

    logger.info(f"Features successfully created: {feature_files}")

    if stop_step == "feature":
        return feature_files


if __name__ == "__main__":
    run_pipeline()
