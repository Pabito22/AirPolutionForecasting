"""Orchestrator for feature creation."""

import pandas
from pathlib import Path
from loguru import logger
from typing import Literal

from airpollutionforecasting.config import DatasetConfig, FeatureConfig
from airpollutionforecasting.features.base_feature_creator import BaseFeatureCreator
from airpollutionforecasting.features.keep_cols_creator import KeepColsCreator
from airpollutionforecasting.features.sin_cos_creator import SinCosCreator


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
        """Creates features as specyfied in the congig and saves in a .csv file.

        Args:
        data: Data (loaded and validated) based on which features for modelling will be extracted.
        split: What data split is it "val" "test" or "train". Used for saving files.
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

        # TODO: Only take care of the paths, go back here later as u don't really know what features
        # you need!
        # for each run, save the configuration and create filename based on that configuration
        # for example pass the configuration name to some function, and it will save it somewhere
        # only one name can exist per existing configuration
        return Path("Silly path.")

    def _is_enabled(self, name: str) -> bool:
        """Check whether a feature type is configured."""
        return len(getattr(self.feature_config, name, [])) > 0
