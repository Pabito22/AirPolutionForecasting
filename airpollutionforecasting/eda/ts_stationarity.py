"""Checking stationarity of time series utils."""

import warnings

import pandas
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_report(
    df: pandas.DataFrame | pandas.Series,
    columns: list[str] | None = None,
    alpha: float = 0.05,
) -> pandas.DataFrame:
    """Run ADF and KPSS on each column of df exactly as given.

    ADF:  null = non-stationary. p < alpha -> stationary.
    KPSS: null = stationary.     p >= alpha -> stationary.
    """
    columns = columns or list(df.columns)
    rows = []
    for col in columns:
        x = df[col].dropna().to_numpy()

        adf_p = adfuller(x, autolag="AIC")[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # KPSS p-value outside table range
            kpss_p = kpss(x, regression="c", nlags="auto")[1]

        adf_ok = adf_p < alpha
        kpss_ok = kpss_p >= alpha

        if adf_ok and kpss_ok:
            verdict = "stationary"
        elif not adf_ok and not kpss_ok:
            verdict = "non-stationary"
        else:
            verdict = "conflicting"

        rows.append(
            {
                "column": col,
                "n_obs": len(x),
                "adf_p": adf_p,
                "kpss_p": kpss_p,
                "verdict": verdict,
            }
        )
    return pandas.DataFrame(rows).set_index("column")
