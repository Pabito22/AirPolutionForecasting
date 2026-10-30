"""Configuration file for PM10 forecasting project."""

from pathlib import Path
import yaml

from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Load the YAML configuration
CONFIG_PATH = PROJ_ROOT / "config" / "base.yaml"


@dataclass
class DatasetConfig:
    """Dataset configuration.

    Attributes:
        min_date (datetime): Minimum date that is expected to exist and to be taken from the dataset.
        max_date (datetime): Maximum date that is expected to exist and to be taken from the dataset.
        path (str): Path to the dataset file.
        column_mapping(dict[str,str]): Mapping of column names from the dataset to the ones provided in
        configuraion.
        column_types (dict[str, str]): Mapping of column name to its type.
        data_split_dates (list[datetime]): List of dates, that specify the start and end of train, validation
            and test datasets: [train_start, train_end, val_start, val_end, test_start, test_end]
    """

    min_date: datetime
    max_date: datetime
    path: str
    column_mapping: dict[str, str]
    column_types: dict[str, str]
    data_split_dates: list[datetime]

    @classmethod
    def load_config(cls, config_path: Path = CONFIG_PATH) -> "DatasetConfig":
        """Create a dataset configuration from the YAML file."""
        with config_path.open("r", encoding="utf-8") as conf_file:
            yaml_config: dict[str, Any] = yaml.safe_load(conf_file)

        try:
            dataset_config = yaml_config["dataset"]
            return cls(
                min_date=datetime.fromisoformat(dataset_config["min_date"]),
                max_date=datetime.fromisoformat(dataset_config["max_date"]),
                path=dataset_config["path"],
                column_mapping=dataset_config["column_mapping"],
                column_types=dataset_config["column_types"],
                data_split_dates = [datetime.fromisoformat(val) for val in dataset_config["data_split_dates"]],
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Invalid or incomplete dataset configuration in YAML."
            ) from error
