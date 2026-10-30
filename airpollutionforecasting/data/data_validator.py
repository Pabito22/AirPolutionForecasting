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

    def __init__(self, data_config: DatasetConfig):
        """
        Initialize the DataValidator.

        Args:
            dataset_config (DatasetConfig): Configuration for the dataset.
        """
        self.dataset_config = data_config

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
    def _validate_no_duplicates(self, loaded_data: pandas.DataFrame) -> None:
        """Ensure that no duplicated rows exist."""
        duplicated = loaded_data.duplicated()

        if duplicated.any():
            raise DataValidationError(
                f"Data contains {duplicated.sum()} duplicated rows."
            )

    def _validate_chronological_order(
        self, loaded_data: pandas.DataFrame) -> None:
        """Ensure data is sorted chronologically."""
        datetime_column = self.dataset_config.column_mapping.get("time", "datetime")

        if not loaded_data[datetime_column].is_monotonic_increasing:
            raise DataValidationError(
                f"Data is not sorted chronologically by '{datetime_column}'."
            )

    def _validate_time_continuity(
        self, loaded_data: pandas.DataFrame
    ) -> None:
        """Ensure there are no missing hourly timestamps."""
        datetime_column = self.dataset_config.column_mapping.get("time", "datetime")

        time_diffs = loaded_data[datetime_column].diff().dropna()

        invalid_diffs = time_diffs != pandas.Timedelta(hours=1)

        if invalid_diffs.any():
            raise DataValidationError(
                f"Found {invalid_diffs.sum()} missing or irregular hourly intervals."
            )

    def _validate_unique_timestamps(self, loaded_data: pandas.DataFrame) -> None:
        """Ensure every timestamp occurs only once."""
        datetime_column = self.dataset_config.column_mapping.get("time", "datetime")

        if loaded_data[datetime_column].duplicated().any():
            raise DataValidationError("Data contains duplicated timestamps.")

    
    def validate(self, loaded_data: pandas.DataFrame) -> None:
        """
        Validate the quality and consistency of the processed dataset.

        Validation checks:
            1. Required columns are present and no unexpected columns exist.
            2. All columns have the expected data types.
            3. The dataset covers the configured minimum and maximum dates.
            4. No missing (NaN) values are present.
            5. No duplicated rows are present.
            6. Timestamps are unique.
            7. Data is sorted in chronological order.
            8. Timestamps are continuous, with exactly one-hour intervals.

        Args:
            loaded_data (pandas.DataFrame): The processed dataset to validate.

        Returns:
            None: If all validation checks pass.

        Raises:
            DataValidationError: If any validation check fails.
        """
        logger.info("Starting data validation...")

        logger.info("Checking required columns and data types...")
        self._validate_columns(loaded_data)
        logger.success("Column and data type validation passed.")

        logger.info("Checking configured date range...")
        self._date_range_validation(loaded_data)
        logger.success("Date range validation passed.")

        logger.info("Checking for missing values...")
        self._validate_no_missing_values(loaded_data)
        logger.success("Missing values validation passed.")

        logger.info("Checking for duplicated rows...")
        self._validate_no_duplicates(loaded_data)
        logger.success("Duplicate rows validation passed.")

        logger.info("Checking for duplicated timestamps...")
        self._validate_unique_timestamps(loaded_data)
        logger.success("Timestamp uniqueness validation passed.")

        logger.info("Checking chronological ordering...")
        self._validate_chronological_order(loaded_data)
        logger.success("Chronological ordering validation passed.")

        logger.info("Checking hourly timestamp continuity...")
        self._validate_time_continuity(loaded_data)
        logger.success("Timestamp continuity validation passed.")

        logger.success("All data validation checks passed.")