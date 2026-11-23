"""Creator of cyclical (sin/cos) features."""

import numpy
import pandas
from loguru import logger

from airpollutionforecasting.features.base_feature_creator import BaseFeatureCreator


class SinCosCreator(BaseFeatureCreator):
    """Encode periodic columns as sine and cosine pairs.

    Cyclical variables (hour of day, day of week, month, wind direction...) are
    problematic for models when kept as raw numbers: hour 23 and hour 0 are
    neighbours in reality, but numerically far apart. Projecting the value onto
    a circle fixes this, so that values close in time stay close in feature space.

    For every column ``x`` with period ``T`` (the denominator), two features are made:

        sin_x = sin(2 * pi * x / T)
        cos_x = cos(2 * pi * x / T)

    Both are needed: sin alone maps two different points of the cycle to the
    same value, and together they identify every point on the circle uniquely.

    The mapping comes from ``feature_config.cos_sin_cols``, a dict of
    ``{column_name: denominator}``. Typical denominators are 24 for the hour of
    the day, 7 for the day of the week, 12 for the month and 360 for degrees.

    Example:
        >>> config = FeatureConfig(cos_sin_cols={"hour": 24})
        >>> creator = SinCosCreator(config)
        >>> creator.create_features(df)  # returns columns: sin_hour, cos_hour
    """

    def create_features(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Create sine and cosine features for each configured periodic column.

        Args:
            data (pandas.DataFrame): Data containing the periodic columns listed in
                ``feature_config.cos_sin_cols``.

        Returns:
            pandas.DataFrame: A new frame, with the same index as ``data``, containing
                only the new columns ``sin_<col>`` and ``cos_<col>`` for each
                configured column. The original columns are not included.

        Raises:
            ValueError: If any configured column is missing in the data, or any
                denominator is not a positive number.
        """
        cos_sin_dict = self.feature_config.cos_sin_cols
        logger.info("Creating sin/cos features for columns = {}...", list(cos_sin_dict))

        missing = [col for col in cos_sin_dict if col not in data.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} specified in configuration are missing from the data!"
            )

        bad_denominators = {c: d for c, d in cos_sin_dict.items() if d <= 0}
        if bad_denominators:
            raise ValueError(f"Denominators must be positive, got: {bad_denominators}")

        new_features = {}
        for col, denominator in cos_sin_dict.items():
            angle = 2 * numpy.pi * data[col] / denominator
            new_features[f"sin_{col}"] = numpy.sin(angle)
            new_features[f"cos_{col}"] = numpy.cos(angle)

        return pandas.DataFrame(new_features, index=data.index)
