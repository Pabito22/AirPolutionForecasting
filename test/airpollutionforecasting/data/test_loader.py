"""Test data loader."""

import pandas
from loguru import logger

from airpollutionforecasting.config import DatasetConfig
from airpollutionforecasting.data import DataLoader, DataValidator


def test_data_loader():
    config = DatasetConfig.from_yaml()
    validator = DataValidator(dataset_config=config)
    loader = DataLoader(config)

    loaded_data = loader.load()

    logger.info("Columns: {}", loaded_data.columns.tolist())
    logger.info("Shape: {}", loaded_data.shape)
    logger.info("First rows:\n{}", loaded_data.head())

    assert not loaded_data.empty
    assert "time" in loaded_data.columns
    assert pandas.api.types.is_datetime64_any_dtype(loaded_data["time"])

    validator.check_issues(loaded_data)
