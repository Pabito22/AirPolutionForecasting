"""Utilities for airpollution forecasting project."""

from pathlib import Path
import pandas


def load_data_from_path(
    data_path: Path, datetime_cols: None | list[str] = None
) -> pandas.DataFrame:
    """Load the data from the given path and return it as pandas.DataFrame.

    Args:
    datetime_cols: columns that should be converted to

    Notes
    Used for loading data beteen steps executed in the pipeline."""

    data = pandas.read_csv(filepath_or_buffer=data_path)
    if datetime_cols is not None:
        for col in datetime_cols:
            if col not in data.columns:
                raise ValueError(
                    f"Provided column {col} not existing in the provided file {data_path}."
                )
            data[col] = pandas.to_datetime(data[col])

    return data
