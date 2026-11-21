"""BaseFeatureClass."""

import pandas
from abc import ABC, abstractmethod

from airpollutionforecasting.config import FeatureConfig


class BaseFeatureClass(ABC):
    """Base for different classes of features."""

    def __init__(self, feature_config: FeatureConfig):
        """Initialize BaseFeatureClass.

        Args:
            feature_config (FeatureConfig): Feature configuration.
        """
        self.feature_config = feature_config

    @abstractmethod
    def create_features(
        self,
        data: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Create features based on the configuration and the provided data.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Data with the created features.
        """
        pass
