"""Load the data from the cleaned CSV file."""

from pathlib import Path

import pandas

from airpollutionforecasting.config import PROJ_ROOT, DatasetConfig


class DataLoader:
    """Load the data from the cleaned CSV file."""

    def __init__(self, data_config: DatasetConfig):
        """Initialize loader."""
        self.config = data_config

    def load(self) -> pandas.DataFrame:
        """Load, normalize, type, and date-filter the configured dataset."""
        data_path = Path(self.config.path)
        if not data_path.is_absolute():
            data_path = PROJ_ROOT / data_path

        data = pandas.read_csv(data_path)

        missing_columns = set(self.config.column_mapping) - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset is missing configured columns: {sorted(missing_columns)}"
            )

        # Normalize source column names.
        data = data.rename(columns=self.config.column_mapping)

        # Convert configured column types.
        for column, dtype in self.config.column_types.items():
            if column not in data.columns:
                raise ValueError(f"Configured type references missing column: {column}")

            if dtype == "datetime":
                data[column] = pandas.to_datetime(data[column], errors="raise")
            elif dtype in {"int", "int64"}:
                data[column] = data[column].astype("int64")
            elif dtype in {"float", "float64"}:
                data[column] = data[column].astype("float64")
            elif dtype in {"str", "string"}:
                data[column] = data[column].astype("string")
            elif dtype == "bool":
                data[column] = data[column].astype("bool")
            else:
                raise ValueError(f"Unsupported configured dtype: {dtype}")

        # Use the normalized datetime column name.
        datetime_column = self.config.column_mapping.get("time", "datetime")

        if datetime_column not in data.columns:
            raise ValueError(
                f"Dataset has no configured datetime column: {datetime_column}"
            )

        # Filter the configured date range and keep the time column.
        data = data.loc[
            data[datetime_column].between(
                self.config.min_date,
                self.config.max_date,
            )
        ].copy()

        # Sort chronologically; keep time as a regular column.
        data = data.sort_values(datetime_column).reset_index(drop=True)
        
        return data
