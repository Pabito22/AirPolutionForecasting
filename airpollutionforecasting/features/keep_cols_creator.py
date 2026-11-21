"""Class designed for the keep_cols type of features (1:1)."""

import pandas as pd
from loguru import logger

from airpollutionforecasting.config import FeatureConfig
from airpollutionforecasting.features.base_feature_creator import BaseFeatureCreator


class KeepColsCreator(BaseFeatureCreator):
    """Class designed for the keep_cols type of features (1:1)."""

    def __init__(self, feature_config: FeatureConfig):
        """Initialize KeepColsCreator.

        Args:
            feature_config (FeatureConfig): Feature configuration.
        """
        super().__init__(feature_config)

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on configuration.

        This class keeps the data as it is. It is mainly used to decide
        which columns to keep.

        Args:
            data (pd.DataFrame): Data from which features will be extracted.

        Returns:
            pd.DataFrame: Frame containing only the columns listed in keep_cols.

        Raises:
            ValueError: If any column from keep_cols is missing in the data.
        """
        keep_cols = self.feature_config.keep_cols
        logger.info("Creating features based on keep_cols = {}...", keep_cols)

        missing = [col for col in keep_cols if col not in data.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} specified in configuration are missing from the data!"
            )
        return data[keep_cols].copy()
