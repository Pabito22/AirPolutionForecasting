"""Orchestrator for feature creation."""

import pandas
from pathlib import Path
from loguru import logger
from typing import Literal

from airpollutionforecasting.config import DatasetConfig, FeatureConfig
from airpollutionforecasting.features.base_feature_creator import BaseFeatureCreator
from airpollutionforecasting.features.keep_cols_creator import KeepColsCreator
from airpollutionforecasting.features.sin_cos_creator import SinCosCreator
from airpollutionforecasting.config import PROJ_ROOT


class FeatureCreator:
    """Class for feature creation orechestration"""

    # Maps a feature type to its creator class.
    _feature_classes: dict[str, type[BaseFeatureCreator]] = {
        "keep_cols": KeepColsCreator,
        "cos_sin_cols": SinCosCreator,
    }

    def __init__(self, data_config: DatasetConfig, feature_config: FeatureConfig):
        """Initialize FeatureCreator.

        Args:
        data_config: Dataset Configuration.
        feature_config: Feature configuration."""

        self.data_config = data_config
        self.feature_config = feature_config

    def create(
        self, data: pandas.DataFrame, split: Literal["train", "val", "test"]
    ) -> Path:
        """Create features as specified in the config and save them to a CSV file.

        If a file for this split and this pair of configurations already
        exists, nothing is recomputed and the existing path is returned.

        Args:
            data (pandas.DataFrame): Loaded and validated data from which the
                features for modelling are extracted.
            split (Literal["train", "val", "test"]): Data split the data belongs
                to. Used in the file name.

        Returns:
            Path: Location of the CSV file containing the created features.

        Raises:
            ValueError: If a configuration is not registered (has no hash) or
                no features are specified in the configuration.

        Notes:
            Files are saved in ``<PROJ_ROOT>/data/feature/`` and named::

                <split>_feature_<feature_hash>_data_<data_hash>.csv

            where ``<split>`` is "train", "val" or "test", ``<feature_hash>`` is
            the hash of the feature configuration and ``<data_hash>`` the hash
            of the dataset configuration. The matching configurations are
            stored in ``logs/feature_config/<feature_hash>.yaml`` and
            ``logs/dataset_config/<data_hash>.yaml``. The same configurations
            therefore always map to the same file name, and different ones never
            overwrite each other.
        """
        if (
            self.feature_config.config_hash is None
            or self.data_config.config_hash is None
        ):
            raise ValueError(
                "Both configurations must be registered (hashed) before creating features."
            )

        output_dir = PROJ_ROOT / "data" / "feature"
        file_name = (
            f"{split}_feature_{self.feature_config.config_hash}"
            f"_data_{self.data_config.config_hash}.csv"
        )
        path = output_dir / file_name

        if path.exists():
            logger.info("Features already exist, reusing {}", path)
            return path

        frames = []
        for name, creator_cls in self._feature_classes.items():
            if self._is_enabled(name):
                logger.info("Creating '{}' features...", name)
                frames.append(creator_cls(self.feature_config).create_features(data))

        if not frames:
            raise ValueError("No features specified in the configuration!")

        feature_frame = pandas.concat(frames, axis=1)

        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".csv.tmp")
        feature_frame.to_csv(tmp_path, index=False)
        tmp_path.replace(path)

        logger.info("Saved features to {}", path)
        return path

    def _is_enabled(self, name: str) -> bool:
        """Check whether a feature type is configured."""
        return len(getattr(self.feature_config, name, [])) > 0
