"""Module for plotting utilities."""

from pathlib import Path
from typing import Optional
from typing import Optional, Sequence

from  matplotlib import pyplot 
import pandas 
import seaborn
import typer
import matplotlib.dates as mdates

from airpollutionforecasting.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def draw_histogram(
    data: pandas.Series,
    x_label: Optional[str] = None,
    y_label: str = "Frequency",
    title: Optional[str] = None,
    bins: int | str = "auto",
    kde: bool = False,
    save_path: Optional[Path] = None,
) -> None:
    """
    Draw a clean histogram of a single dataframe column.

    Args:
        data: The column to plot, e.g. df["PM2.5"].
        x_label: X axis label. Defaults to the column's name.
        y_label: Y axis label. Defaults to "Frequency".
        title: Figure title. No title is shown unless specified.
        bins: Number of bins or a binning rule (e.g. "auto", "fd", 30).
        kde: Whether to overlay a smooth density curve.
        save_path: If given, the figure is saved there (e.g. FIGURES_DIR / "pm25.png").

    Returns:
        The matplotlib Figure.
    """
    data = data.dropna()

    seaborn.set_theme(style="whitegrid", context="notebook")

    fig, ax = pyplot.subplots(figsize=(8, 5))

    seaborn.histplot(
        data,
        bins=bins,
        kde=kde,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
        ax=ax,
    )

    ax.set_xlabel(x_label if x_label is not None else str(data.name or "Value"), fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    seaborn.despine(ax=ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

def draw_timeseries(
    data: pandas.DataFrame,
    x: str,
    y: Sequence[str] | str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    legend: bool = True,
    legend_loc: str = "best",
    secondary_y: Optional[Sequence[str]] = None,
    secondary_y_label: Optional[str] = None,
    figsize: tuple[float, float] = (12, 5),
    linewidth: float = 1.6,
    markers: bool = False,
    resample: Optional[str] = None,
    rolling: Optional[int] = None,
    date_format: Optional[str] = None,
    save_path: Optional[Path] = None,
    start: Optional[str | pandas.Timestamp] = None,
    end: Optional[str | pandas.Timestamp] = None,
) -> None:
    """
    Plot one or more columns of a dataframe against a datetime column.

    Args:
        data: Source dataframe.
        x: Name of the datetime column (or the index name if the datetime is the index).
        y: Column name or list of column names to plot on the same axes.
        x_label: X axis label. Defaults to the name of `x`.
        y_label: Y axis label. Defaults to the column name if one column is plotted, else none.
        title: Figure title. No title unless specified.
        labels: Legend names for the series (same length as y). Defaults to column names.
        legend: Show the legend (only shown when there is more than one series, or labels are given).
        legend_loc: Matplotlib legend location, e.g. "upper right", "best".
        secondary_y: Subset of `y` columns to draw on a right-hand y axis (useful for different scales).
        secondary_y_label: Label for the right-hand axis.
        figsize: Figure size in inches.
        linewidth: Line width.
        markers: Draw a marker on each data point.
        resample: Pandas offset alias to resample by mean, e.g. "D", "W", "M".
        rolling: Window size (in samples, after resampling) for a rolling mean.
        date_format: strftime format for x tick labels, e.g. "%Y-%m". Auto if None.
        save_path: If given, the figure is saved there (e.g. FIGURES_DIR / "pm25_ts.png").
        start: Start of the period to plot. Partial dates work, e.g. "2023" or "2023-03".
        end: End of the period (inclusive). Defaults to `start`'s period if only `start` is a partial date, see notes.
    Returns:
        The matplotlib Figure.
    """
    y_cols = [y] if isinstance(y, str) else list(y)
    secondary_y = list(secondary_y) if secondary_y else []

    if labels is not None and len(labels) != len(y_cols):
        raise ValueError("`labels` must have the same length as `y`.")
    label_map = dict(zip(y_cols, labels if labels is not None else y_cols))

    # Build a clean, time-indexed frame
    df = data.copy()
    if x in df.columns:
        df[x] = pandas.to_datetime(df[x])
        df = df.set_index(x)
    elif df.index.name == x:
        df.index = pandas.to_datetime(df.index)
    else:
        raise KeyError(f"'{x}' not found in columns or index name.")
    df = df[y_cols].sort_index()

    if start is not None or end is not None:
        df = df.loc[start:end]
        if df.empty:
            raise ValueError(f"No data between {start} and {end}.")

    if resample:
        df = df.resample(resample).mean()
    if rolling:
        df = df.rolling(rolling, min_periods=1).mean()

    seaborn.set_theme(style="whitegrid", context="notebook")
    palette = seaborn.color_palette("deep", n_colors=len(y_cols))

    fig, ax = pyplot.subplots(figsize=figsize)
    ax2 = ax.twinx() if secondary_y else None

    handles = []
    for col, color in zip(y_cols, palette):
        target = ax2 if col in secondary_y else ax
        (line,) = target.plot(
            df.index,
            df[col],
            label=label_map[col],
            color=color,
            linewidth=linewidth,
            marker="o" if markers else None,
            markersize=3,
        )
        handles.append(line)

    # Axis labels
    ax.set_xlabel(x_label if x_label is not None else x, fontsize=12)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=12)
    elif len(y_cols) == 1:
        ax.set_ylabel(label_map[y_cols[0]], fontsize=12)

    if ax2 is not None:
        ax2.set_ylabel(secondary_y_label or ", ".join(label_map[c] for c in secondary_y), fontsize=12)
        ax2.grid(False)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Date formatting
    if date_format:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    else:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # Single combined legend (covers both axes)
    if legend and (len(y_cols) > 1 or labels is not None):
        ax.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc=legend_loc,
            frameon=True,
            framealpha=0.9,
        )

    seaborn.despine(ax=ax, right=ax2 is None)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    pyplot.close(fig)  
    return fig
    


if __name__ == "__main__":
    app()