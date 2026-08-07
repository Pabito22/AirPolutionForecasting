"""Configuration file for PM10 forecasting project."""

from pathlib import Path
import yaml

from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
from datetime import datetime


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
    """

    min_date: datetime
    max_date: datetime
    path: str
    column_mapping: dict[str, str]
    column_types: dict[str, str]

    def load_congfig(self):
        """Read the yaml configuration."""
        with open(CONFIG_PATH, "r") as conf_file:
            yaml_config = yaml.safe_load(conf_file)

            for key, value in yaml_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"No {key, value} is definded in config.py!")
