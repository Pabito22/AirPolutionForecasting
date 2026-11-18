"""Calculate autocorrelation and covariates correlation of time series."""

import pandas
from statsmodels.tsa.stattools import acf, pacf
import numpy
from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_acf


class TSCorrelationChecker:
    """Calculate autocorrelation and covariates correlation of time series."""

    def __init__(self, data: pandas.DataFrame, target_col: str):
        """Initialize TSCorrelationChecker.

        Args:
            data: Data on which autocorrelation and correlation with
                covariates will be checked.
            target_col: Target column name (for autocorrelation).

        Raises:
            KeyError: If target_col is not in data.
        """
        if target_col not in data.columns:
            raise KeyError(f"Column '{target_col}' not found in data.")
        self.data = data
        self.target_col = target_col

    def calculate_pacf(
        self,
        nlags: int = 168,
        method: str = "ywm",
        alpha: float = 0.05,
    ) -> pandas.DataFrame:
        """Calculate partial autocorrelation of the target column.

        Args:
            nlags: Number of lags to compute (168 = one week for hourly data).
            method: Estimation method passed to statsmodels' `pacf`
                ("ywm", "ols", "ldb", "burg", ...).
            alpha: Significance level for the confidence bounds.

        Returns:
            DataFrame indexed by lag (1..nlags) with columns:
                pacf: Partial autocorrelation value.
                lower: Lower significance bound (around zero).
                upper: Upper significance bound (around zero).
                significant: True if |pacf| exceeds the bound.

        Raises:
            ValueError: If the series contains NaNs or is too short for nlags.
        """
        series = self.data[self.target_col]

        if series.isna().any():
            raise ValueError(
                f"Column '{self.target_col}' contains NaNs; "
                "fill or drop them before calculating PACF."
            )
        if nlags >= len(series) // 2:
            raise ValueError(
                f"nlags must be < nobs // 2 ({len(series) // 2}), got {nlags}."
            )

        values, confint = pacf(
            series.to_numpy(), nlags=nlags, method=method, alpha=alpha
        )

        # statsmodels returns confint centred on each pacf value;
        # convert to bounds centred on zero (as in plot_pacf).
        half_width = confint[:, 1] - values
        lower, upper = -half_width, half_width

        result = pandas.DataFrame(
            {
                "pacf": values,
                "lower": lower,
                "upper": upper,
                "significant": numpy.abs(values) > half_width,
            },
            index=pandas.RangeIndex(nlags + 1, name="lag"),
        )
        return result.iloc[1:]  # drop lag 0 (always 1)

    def plot_pacf_graph(
        self,
        nlags: int = 168,
        method: str = "ywm",
        alpha: float = 0.05,
        seasonal_period: int | None = 24,
        ax: pyplot.Axes | None = None,
        figsize: tuple[int, int] = (14, 4),
    ) -> pyplot.Axes:
        """Plot partial autocorrelation of the target column.

        Args:
            nlags: Number of lags to plot (168 = one week for hourly data).
            method: Estimation method passed to `calculate_pacf`.
            alpha: Significance level for the confidence band.
            seasonal_period: If set, draws a faint vertical line at every
                multiple of this lag (24 = daily seasonality for hourly
                data). Use None to disable.
            ax: Existing matplotlib axes to draw on. A new figure is
                created if None.
            figsize: Figure size used only when `ax` is None.

        Returns:
            The matplotlib Axes with the plot.
        """
        pacf_df = self.calculate_pacf(nlags=nlags, method=method, alpha=alpha)

        if ax is None:
            _, ax = pyplot.subplots(figsize=figsize)

        # Confidence band around zero
        ax.fill_between(
            pacf_df.index,
            pacf_df["lower"],
            pacf_df["upper"],
            color="tab:blue",
            alpha=0.2,
            label=f"{int((1 - alpha) * 100)}% confidence band",
        )

        # Stems: highlight significant lags
        colors = [
            "tab:red" if is_significant else "tab:blue"
            for is_significant in pacf_df["significant"]
        ]
        ax.vlines(pacf_df.index, 0, pacf_df["pacf"], colors=colors, linewidth=1)
        ax.scatter(pacf_df.index, pacf_df["pacf"], c=colors, s=14, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8)

        # Seasonal guide lines
        if seasonal_period:
            for k in range(seasonal_period, nlags + 1, seasonal_period):
                ax.axvline(k, color="black", alpha=0.25, linestyle="--", linewidth=0.8)
            ax.set_xticks(range(0, nlags + 1, seasonal_period))

        ax.set_xlim(0, nlags + 1)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Partial autocorrelation")
        ax.set_title(f"PACF of '{self.target_col}' ({nlags} lags)")
        ax.legend(loc="upper right")
        return ax

    def calculate_acf(
        self,
        nlags: int = 168,
        adjusted: bool = True,
        alpha: float = 0.05,
    ) -> pandas.DataFrame:
        """Calculate autocorrelation of the target column.

        Args:
            nlags: Number of lags to compute (168 = one week for hourly data).
            adjusted: If True, the autocovariance is divided by (n - k)
                instead of n at each lag k, which reduces bias at long lags.
            alpha: Significance level for the confidence bounds.

        Returns:
            DataFrame indexed by lag (1..nlags) with columns:
                acf: Autocorrelation value.
                lower: Lower significance bound (around zero).
                upper: Upper significance bound (around zero).
                significant: True if |acf| exceeds the bound.

        Raises:
            ValueError: If the series contains NaNs or is too short for nlags.
        """
        series = self.data[self.target_col]

        if series.isna().any():
            raise ValueError(
                f"Column '{self.target_col}' contains NaNs; "
                "fill or drop them before calculating ACF."
            )
        if nlags >= len(series):
            raise ValueError(f"nlags must be < nobs ({len(series)}), got {nlags}.")

        values, confint = acf(
            series.to_numpy(),
            nlags=nlags,
            adjusted=adjusted,
            alpha=alpha,
            fft=True,
        )

        # statsmodels returns Bartlett confidence intervals centred on each
        # acf value; convert to bounds centred on zero (as in plot_acf).
        # Note: these widen with the lag, unlike the constant PACF bounds.
        half_width = confint[:, 1] - values
        lower, upper = -half_width, half_width

        result = pandas.DataFrame(
            {
                "acf": values,
                "lower": lower,
                "upper": upper,
                "significant": numpy.abs(values) > half_width,
            },
            index=pandas.RangeIndex(nlags + 1, name="lag"),
        )
        return result.iloc[1:]  # drop lag 0 (always 1)

    def plot_acf_graph(
        self,
        nlags: int = 168,
        alpha: float = 0.05,
        adjusted: bool = True,
        seasonal_period: int | None = 24,
        ax: pyplot.Axes | None = None,
        figsize: tuple[int, int] = (14, 4),
    ) -> pyplot.Axes:
        """Plot autocorrelation of the target column.

        Args:
            nlags: Number of lags to plot (168 = one week for hourly data).
            alpha: Significance level for the confidence band.
            adjusted: If True, divide autocovariance by (n - k) instead of n.
            seasonal_period: If set, draws a faint vertical line at every
                multiple of this lag (24 = daily seasonality for hourly
                data). Use None to disable.
            ax: Existing matplotlib axes to draw on. A new figure is
                created if None.
            figsize: Figure size used only when `ax` is None.

        Returns:
            The matplotlib Axes with the plot.

        Raises:
            ValueError: If the series contains NaNs or is too short for nlags.
        """
        series = self.data[self.target_col]

        if series.isna().any():
            raise ValueError(
                f"Column '{self.target_col}' contains NaNs; "
                "fill or drop them before plotting ACF."
            )
        if nlags >= len(series):
            raise ValueError(f"nlags must be < nobs ({len(series)}), got {nlags}.")

        if ax is None:
            _, ax = pyplot.subplots(figsize=figsize)

        plot_acf(
            series.to_numpy(),
            lags=nlags,
            alpha=alpha,
            adjusted=adjusted,
            zero=False,  # lag 0 is always 1, so skip it
            auto_ylims=True,  # rescale y-axis to the data (needs statsmodels >= 0.14)
            ax=ax,
        )

        if seasonal_period:
            for k in range(seasonal_period, nlags + 1, seasonal_period):
                ax.axvline(k, color="gray", alpha=0.25, linestyle="--", linewidth=0.8)
            ax.set_xticks(range(0, nlags + 1, seasonal_period))

        ax.set_xlim(0, nlags + 1)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"ACF of '{self.target_col}' ({nlags} lags)")
        return ax
