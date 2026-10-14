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

    def _validate_columns(self, processed_data: pandas.DataFrame):
        """
        Ensure the data contains only the expected columns and types.

        Args:
            processed_data (pandas.DataFrame): The processed dataset to validate.

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

        # Check wether the type of the column is as expected
        for column, expected in self.dataset_config.column_types.items():
            actual = processed_data[column]
            valid = (
                pandas.api.types.is_datetime64_any_dtype(actual)
                if expected == "datetime"
                else str(actual.dtype) == expected
            )
            if not valid:
                raise DataValidationError(
                    f"{column} should be {expected}, not {actual.dtype}"
                )

    def _date_range_validation(self, processed_data: pandas.DataFrame) -> None:
        """Ensures that data contains all dates in the range specified in config."""
        datetime_column = self.dataset_config.column_mapping.get("time", "datetime")
        if processed_data.empty or datetime_column not in processed_data.columns:
            raise DataValidationError(
                f"Data must contain a non-empty '{datetime_column}' column."
            )

        actual_min = processed_data[datetime_column].min()
        actual_max = processed_data[datetime_column].max()
        expected_min = self.dataset_config.min_date
        expected_max = self.dataset_config.max_date
        if actual_min != expected_min or actual_max != expected_max:
            raise DataValidationError(
                f"Expected date range {expected_min} to {expected_max}, "
                f"got {actual_min} to {actual_max}."
            )

    def _validate_no_missing_values(self, processed_data: pandas.DataFrame) -> None:
        """Ensure that no missing values remain in the processed data."""
        missing_columns = processed_data.columns[processed_data.isna().any()].tolist()
        if missing_columns:
            raise DataValidationError(
                f"Data contains missing values in columns: {missing_columns}."
            )

    def check_issues(self, processed_data: pandas.DataFrame) -> Optional[None]:
        """
        Validate the quality of the processed data.

        Notes:
            1. Ensures data contains required columns.
            2. Ensures right types of columns.
            3. Ensures that data contains all dates in the range specified in config.
            4. Ensures no NaN values are present in the data.

        Args:
            processed_data (pandas.DataFrame): The processed dataset to validate.

        Returns:
            None if the data meets expectations.

        Raises:
            DataValidationError: If any validation check fails.
        """
        logger.info("Starting data validation...")
        self._validate_columns(processed_data=processed_data)
        logger.info("Column validation passed.")
        self._date_range_validation(processed_data=processed_data)
        self._validate_no_missing_values(processed_data=processed_data)
