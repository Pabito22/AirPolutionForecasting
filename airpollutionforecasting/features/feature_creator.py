"""Orchestrator for feature creation."""

import pandas
from pathlib import Path
from loguru import logger
from airpollutionforecasting.config import DatasetConfig, FeatureConfig
from airpollutionforecasting.features.base_feature_creator import BaseFeatureCreator
from airpollutionforecasting.features.keep_cols_creator import KeepColsCreator


class FeatureCreator:
    """Class for feature creation orechestration"""

    # Maps a feature type to its creator class.
    _feature_classes: dict[str, type[BaseFeatureCreator]] = {
        "keep_cols": KeepColsCreator,
    }

    def __init__(self, data_config: DatasetConfig, feature_config: FeatureConfig):
        """Initialize FeatureCreator.

        Args:
        data_config: Dataset Configuration.
        feature_config: Feature configuration."""

        self.data_config = data_config
        self.feature_config = feature_config

    def create(self, data: pandas.DataFrame) -> Path:
        """Creates features as specyfied in the congig.

        Args:
        data: Data (loaded and validated) based on which features for modelling will be extracted.
        Returns:
            Paths to the file containg data with newly created features.
        Notes:
        Each file that is created by this class
        contains data that is created based on the provided configurationn files.
        If the specified configuration appeard before and the corresponfing
         files still exist in the corresponding paths, this class will return
         the corresponding paths.
        """
        frames = []
        for name, creator_cls in self._feature_classes.items():
            if self._is_enabled(name):
                logger.info("Creating '{}' features...", name)
                frames.append(creator_cls(self.feature_config).create_features(data))

        if not frames:
            raise ValueError("No features specified in the configuration!")

        feature_frame = pandas.concat(frames, axis=1)
        return Path("Silly path.")

    def _is_enabled(self, name: str) -> bool:
        """Check whether a feature type is configured."""
        return len(getattr(self.feature_config, name, [])) > 0
