"""Validates data quality."""

import pandas
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

    def _validate_columns(self, loaded_data: pandas.DataFrame) -> None:
        """Ensure data contains expected columns and types."""
        expected_columns = set(self.dataset_config.column_mapping.values())
        actual_columns = set(loaded_data.columns)

        missing_columns = expected_columns - actual_columns
        unexpected_columns = actual_columns - expected_columns

        if missing_columns:
            raise DataValidationError(
                f"Data is missing expected columns: {missing_columns}."
            )

        if unexpected_columns:
            raise DataValidationError(
                f"Data contains unexpected columns: {unexpected_columns}."
            )

        for column, expected in self.dataset_config.column_types.items():
            actual = loaded_data[column]

            valid = (
                pandas.api.types.is_datetime64_any_dtype(actual)
                if expected == "datetime"
                else str(actual.dtype) == expected
            )

            if not valid:
                raise DataValidationError(
                    f"{column} should be {expected}, not {actual.dtype}."
                )

    def _date_range_validation(self, loaded_data: pandas.DataFrame) -> None:
        """Ensure data covers the configured date range."""
        datetime_column = self.dataset_config.column_mapping.get("time", "datetime")

        if loaded_data.empty or datetime_column not in loaded_data.columns:
            raise DataValidationError(
                f"Data must contain a non-empty '{datetime_column}' column."
            )

        actual_min = loaded_data[datetime_column].min()
        actual_max = loaded_data[datetime_column].max()

        if (
            actual_min != self.dataset_config.min_date
            or actual_max != self.dataset_config.max_date
        ):
            raise DataValidationError(
                f"Expected date range {self.dataset_config.min_date} to "
                f"{self.dataset_config.max_date}, got {actual_min} to {actual_max}."
            )

    def _validate_no_missing_values(self, loaded_data: pandas.DataFrame) -> None:
        """Ensure that no missing values remain in the processed data."""
        missing_columns = loaded_data.columns[loaded_data.isna().any()].tolist()
        if missing_columns:
            raise DataValidationError(
                f"Data contains missing values in columns: {missing_columns}."
            )

    def validate(self, loaded_data: pandas.DataFrame) -> Optional[None]:
        """
        Validate the quality of the processed data.

        Notes:
            1. Ensures data contains required columns.
            2. Ensures right types of columns.
            3. Ensures that data contains all dates in the range specified in config.
            4. Ensures no NaN values are present in the data.

        Args:
            loaded_data (pandas.DataFrame): The processed dataset to validate.

        Returns:
            None if the data meets expectations.

        Raises:
            DataValidationError: If any validation check fails.
        """
        logger.info("Starting data validation...")
        self._validate_columns(loaded_data=loaded_data)
        logger.info("Column validation passed.")
        self._date_range_validation(loaded_data=loaded_data)
        self._validate_no_missing_values(loaded_data=loaded_data)
