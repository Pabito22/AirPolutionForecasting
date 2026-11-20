"""Calculate autocorrelation and covariates correlation of time series."""

import pandas
from statsmodels.tsa.stattools import acf, pacf
import numpy
from matplotlib import pyplot
from statistics import NormalDist
from typing import Literal

from statsmodels.graphics.tsaplots import plot_acf


class TSCorrelationChecker:
    """Calculate autocorrelation and covariates correlation of time series."""

    def __init__(self, data: pandas.DataFrame|pandas.Series, target_col: str):
        """Initialize TSCorrelationChecker.

        Args:
            data: Data on which autocorrelation and correlation with
                covariates will be checked.
            target_col: Target column name (for autocorrelation).

        Raises:
            KeyError: If target_col is not in data.
        """
        if isinstance(data, pandas.DataFrame):
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
        if isinstance(self.data, pandas.DataFrame):
            series = self.data[self.target_col]
        else:
            series = self.data

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
        if isinstance(self.data, pandas.DataFrame):
                    series = self.data[self.target_col]
        else:
            series = self.data

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
        if isinstance(self.data, pandas.DataFrame):
                    series = self.data[self.target_col]
        else:
            series = self.data

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

    def calculate_ccf(
        self,
        covariate_cols: list[str] | None = None,
        max_lag: int = 168,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        diff_lag: int | None = None,
    ) -> pandas.DataFrame:
        """Calculate cross-correlation between target and covariates.

        Args:
            covariate_cols: Covariate column names. Defaults to all numeric
                columns except the target.
            max_lag: Maximum lag in both directions (168 = one week for
                hourly data).
            method: Correlation method: "pearson" or "spearman".
            diff_lag: If set, apply differencing at this lag to the target
                and covariates first (e.g. 24 removes the daily cycle, 1
                removes the trend). This reduces spurious correlation
                caused by shared seasonality or trend.

        Returns:
            DataFrame indexed by lag (-max_lag..max_lag) with one column per
            covariate. Positive lag means the covariate leads the target
            (covariate at t - lag vs target at t).

        Raises:
            KeyError: If a covariate column is not in data.
            ValueError: If max_lag is too large for the data.
        """
        if covariate_cols is None:
            covariate_cols = [
                col
                for col in self.data.select_dtypes("number").columns
                if col != self.target_col
            ]
        missing = [col for col in covariate_cols if col not in self.data.columns]
        if missing:
            raise KeyError(f"Columns not found in data: {missing}")
        if max_lag >= len(self.data) // 2:
            raise ValueError(
                f"max_lag must be < nobs // 2 ({len(self.data) // 2}), got {max_lag}."
            )

        frame = self.data[[self.target_col, *covariate_cols]]
        if diff_lag:
            frame = frame.diff(diff_lag).dropna()

        target = frame[self.target_col]
        lags = range(-max_lag, max_lag + 1)
        result = pandas.DataFrame(
            {
                col: [target.corr(frame[col].shift(k), method=method) for k in lags]
                for col in covariate_cols
            },
            index=pandas.RangeIndex(-max_lag, max_lag + 1, name="lag"),
        )
        return result

    def plot_ccf(
        self,
        covariate_cols: list[str] | None = None,
        max_lag: int = 168,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        diff_lag: int | None = None,
        alpha: float = 0.05,
        horizon: int | None = None,
        seasonal_period: int | None = 24,
        figsize_per_plot: tuple[int, int] = (14, 3),
    ) -> numpy.ndarray:
        """Plot cross-correlation between target and each covariate.

        Positive lag means the covariate leads the target (covariate at
        t - lag vs target at t). Lag 0 is the same-time relationship, which
        is what future (known-ahead) covariates provide.

        Args:
            covariate_cols: Covariate column names. Defaults to all numeric
                columns except the target.
            max_lag: Maximum lag in both directions.
            method: Correlation method: "pearson" or "spearman".
            diff_lag: Differencing lag applied before correlating (see
                `calculate_ccf`).
            alpha: Significance level for the approximate confidence band.
            horizon: Forecast horizon in steps. If set, lags below it are
                shaded gray (not available to a past covariate at prediction
                time) and the strongest usable lag is annotated.
            seasonal_period: If set, draws faint vertical lines at every
                multiple of this lag. Use None to disable.
            figsize_per_plot: (width, height) of each subplot.

        Returns:
            Array of matplotlib Axes, one per covariate.
        """
        ccf = self.calculate_ccf(
            covariate_cols=covariate_cols,
            max_lag=max_lag,
            method=method,
            diff_lag=diff_lag,
        )

        # Rough bound: z / sqrt(n). Unreliable for autocorrelated series,
        # use it as a guide only.
        n_obs = len(self.data) - (diff_lag or 0)
        z = NormalDist().inv_cdf(1 - alpha / 2)
        bound = z / numpy.sqrt(n_obs)

        n_plots = ccf.shape[1]
        fig, axes = pyplot.subplots(
            n_plots,
            1,
            figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_plots),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]

        for ax, col in zip(axes, ccf.columns):
            values = ccf[col]

            # Unavailable region for a past covariate
            if horizon:
                ax.axvspan(
                    -max_lag - 1,
                    horizon - 0.5,
                    color="gray",
                    alpha=0.12,
                    label=f"not usable as past covariate (lag < {horizon})",
                )

            # Confidence band around zero
            ax.fill_between(
                values.index,
                -bound,
                bound,
                color="tab:blue",
                alpha=0.2,
                label=f"~{int((1 - alpha) * 100)}% band",
            )

            # Stems: red if outside the band
            colors = ["tab:red" if abs(v) > bound else "tab:blue" for v in values]
            ax.vlines(values.index, 0, values, colors=colors, linewidth=1)
            ax.scatter(values.index, values, c=colors, s=10, zorder=3)
            ax.axhline(0, color="black", linewidth=0.8)

            # Lag 0: same-time relationship (future covariates)
            ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)

            # Seasonal guide lines
            if seasonal_period:
                for k in range(seasonal_period, max_lag + 1, seasonal_period):
                    for sign in (-1, 1):
                        ax.axvline(
                            sign * k,
                            color="black",
                            alpha=0.2,
                            linestyle="--",
                            linewidth=0.8,
                        )

            # Annotate the strongest usable past-covariate lag
            if horizon:
                usable = values.loc[horizon:]
                if not usable.empty:
                    best_lag = int(usable.abs().idxmax())
                    best_val = float(usable.loc[best_lag])
                    ax.annotate(
                        f"lag {best_lag}: {best_val:.2f}",
                        xy=(best_lag, best_val),
                        xytext=(10, 10 if best_val >= 0 else -15),
                        textcoords="offset points",
                        fontsize=9,
                        arrowprops={"arrowstyle": "-", "alpha": 0.6},
                    )

            ax.set_xlim(-max_lag - 1, max_lag + 1)
            ax.set_ylabel("Correlation")
            ax.set_title(f"CCF: '{self.target_col}' vs '{col}'", loc="left")
            ax.legend(loc="upper right", fontsize=8)

        diff_note = f", differenced at lag {diff_lag}" if diff_lag else ""
        axes[-1].set_xlabel(
            "Lag (positive = covariate leads target, 0 = same time, "
            f"negative = target leads covariate{diff_note})"
        )
        fig.tight_layout()
        return axes
