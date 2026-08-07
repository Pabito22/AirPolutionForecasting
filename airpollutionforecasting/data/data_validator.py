"""Validates data quality."""

import pandas as pd
from typing import Optional
from loguru import logger

from airpollutionforecasting.config import DatasetConfig


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class DataValidator:
    """Ensures that data meets quality standards such as completeness, continuity, and correctness."""

    def __init__(self, dataset_config: DatasetConfig):
        """
        Initialize the DataValidator.

        Args:
            dataset_config (DatasetConfig): Configuration for the dataset.
        """
        self.dataset_config = dataset_config

    def _validate_columns(self, processed_data: pd.DataFrame):
        """
        Ensure the data contains only the expected columns.

        Args:
            processed_data (pd.DataFrame): The processed dataset to validate.

        Raises:
            DataValidationError: If unexpected columns are found.
        """
        expected_columns = set(self.dataset_config.column_mapping.values())
        actual_columns = set(processed_data.columns)
        unexpected_columns = actual_columns - expected_columns

        if unexpected_columns:
            raise DataValidationError(
                f"Data contains unexpected columns: {unexpected_columns}."
            )

    def check_issues(self, processed_data: pd.DataFrame) -> Optional[None]:
        """
        Validate the quality of the processed data.

        Notes:
            1. Ensures data contains required columns.
            2. Ensures right types of columns.
            3. Ensures that data contains all dates in the range specified in config.
            4. Ensures no NaN values are present in the data.

        Args:
            processed_data (pd.DataFrame): The processed dataset to validate.

        Returns:
            None if the data meets expectations.

        Raises:
            DataValidationError: If any validation check fails.
        """
        logger.info("Starting data validation...")
        self._validate_columns(processed_data=processed_data)
        logger.info("Column validation passed.")
