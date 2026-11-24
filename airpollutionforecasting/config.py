"""Configuration file for PM10 forecasting project."""

import hashlib
from pathlib import Path
import yaml

from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass, field
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

# Place to keep the logs of configurations used
LOGS_DIR = PROJ_ROOT / "logs"


@dataclass
class SavedConfigMixin:
    """Adds hash-based persistence to a configuration dataclass.

    After ``register_config`` is called, the configuration section is stored in
    ``<logs_dir>/<section>/<hash>.yaml`` and the hash and path are kept on the
    instance, so downstream steps can name their output files after them.

    Attributes:
        config_hash (str | None): Short SHA-256 hash of the canonical section.
        config_path (Path | None): Location of the saved configuration file.
    """

    # kw_only lets subclasses keep their own non-default fields (e.g. DatasetConfig).
    config_hash: str | None = field(default=None, kw_only=True, compare=False)
    config_path: Path | None = field(default=None, kw_only=True, compare=False)

    def register_config(
        self,
        section_name: str,
        raw_section: dict[str, Any],
        logs_dir: Path = LOGS_DIR,
        hash_length: int = 12,
    ) -> None:
        """Hash the configuration section and save it under logs_dir/section_name.

        Args:
            section_name (str): Name of the YAML section, e.g. "feature".
            raw_section (dict[str, Any]): The section exactly as read from YAML.
            logs_dir (Path): Root directory for saved configurations.
            hash_length (int): Number of hex characters of the hash to keep.
        """
        canonical = yaml.safe_dump(
            raw_section, sort_keys=True, default_flow_style=False
        )
        self.config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
            :hash_length
        ]

        path = logs_dir / section_name / f"{self.config_hash}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():  # same config -> same file, nothing to rewrite
            path.write_text(canonical, encoding="utf-8")
        self.config_path = path


@dataclass
class DatasetConfig(SavedConfigMixin):
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
        numeric_cols (list[str]): A list of numerical columns used in the oryginal dataset.
    """

    min_date: datetime
    max_date: datetime
    path: str
    column_mapping: dict[str, str]
    column_types: dict[str, str]
    data_split_dates: list[datetime]
    numeric_cols: list["str"]

    @classmethod
    def load_config(cls, config_path: Path = CONFIG_PATH) -> "DatasetConfig":
        """Create a dataset configuration from the YAML file."""
        with config_path.open("r", encoding="utf-8") as conf_file:
            yaml_config: dict[str, Any] = yaml.safe_load(conf_file)

        try:
            dataset_config = yaml_config["dataset"]
            instance = cls(
                min_date=datetime.fromisoformat(dataset_config["min_date"]),
                max_date=datetime.fromisoformat(dataset_config["max_date"]),
                path=dataset_config["path"],
                column_mapping=dataset_config["column_mapping"],
                column_types=dataset_config["column_types"],
                data_split_dates=[
                    datetime.fromisoformat(val)
                    for val in dataset_config["data_split_dates"]
                ],
                numeric_cols=[
                    k for k in dataset_config["column_types"].keys() if k != "time"
                ],
            )
            instance.register_config("dataset", dataset_config)
            return instance
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Invalid or incomplete dataset configuration in YAML."
            ) from error


@dataclass
class FeatureConfig(SavedConfigMixin):
    """Configuration of the features used for modelling.

    Each attribute corresponds to one type of feature creator. An empty
    attribute means that feature type is not used.

    Attributes:
        keep_cols (list[str]): Columns that already exist in the loaded data and
            are passed unchanged to the next steps of the pipeline. Any column not
            listed here is dropped from modelling, unless it is used by another
            feature type. Only columns present in
            ``DatasetConfig.column_mapping.values()`` are accepted.
        cos_sin_cols (dict[str, float]): Periodic columns to be encoded as a
            sine and cosine pair, mapped to their period (denominator), e.g.
            ``{"hour": 24, "weekday": 7}``. For each column ``x`` with
            denominator ``T`` two features are created:

                sin_x = sin(2 * pi * x / T)
                cos_x = cos(2 * pi * x / T)

            The new feature names are the column name prefixed with ``sin_`` or
            ``cos_``. Only columns present in
            ``DatasetConfig.column_mapping.values()`` are accepted.

    Example:
        >>> FeatureConfig(keep_cols=["pm10"], cos_sin_cols={"hour": 24})
    """

    keep_cols: list[str] = field(default_factory=list)
    cos_sin_cols: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load_config(cls, config_path: Path = CONFIG_PATH) -> "FeatureConfig":
        with config_path.open("r", encoding="utf-8") as conf_file:
            yaml_config: dict[str, Any] = yaml.safe_load(conf_file)
        try:
            feature_config = yaml_config["feature"]
            dataset_config = yaml_config["dataset"]

            keep_cols = feature_config.get("keep_cols", [])
            cos_sin_cols = feature_config.get("cos_sin_cols", {})
            allowed_cols = set(dataset_config["column_mapping"].values())

            cols_to_check = [*keep_cols, *cos_sin_cols.keys()]
            invalid_cols = [c for c in cols_to_check if c not in allowed_cols]
            if invalid_cols:
                raise ValueError(
                    f"Columns {invalid_cols} not present in "
                    "dataset.column_mapping values!"
                )
            instance = cls(keep_cols=keep_cols, cos_sin_cols=cos_sin_cols)
            instance.register_config("feature", feature_config)
            return instance
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(
                "Invalid or incomplete feature configuration in YAML."
            ) from error
