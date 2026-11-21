"""Orchestrator for feature creation."""

import pandas
from pathlib import Path
from airpollutionforecasting.config import DatasetConfig, FeatureConfig


class FeatureCreator:
    """Class for feature creation orechestration"""

    # Possible features classes.
    _feature_classes = ["keep_cols"]

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
        frame = data.copy()
        return Path("SillyPath")
