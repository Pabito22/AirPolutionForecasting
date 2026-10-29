"""Test wether DataSplitter works as intended, (dates are chronological and as given in the config file)."""

from airpollutionforecasting.data.splitter import DataSplitter

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas 



def test_split_dates(tmp_path, monkeypatch):
    """Test DataSplitter."""
    
    # Arrange: configure the splitter
    dates = [
        datetime(2023, 6, 15, 22),
        datetime(2023, 6, 15, 23),
        datetime(2023, 6, 16, 0),
        datetime(2023, 6, 16, 1),
        datetime(2023, 6, 16, 2),
        datetime(2023, 6, 16, 3),
    ]

    splitter = object.__new__(DataSplitter)
    splitter.config = SimpleNamespace(
        data_split_dates=dates,
        min_date=dates[0],
        max_date=dates[-1],
    )

    monkeypatch.chdir(tmp_path)

    # Arrange: input data, including timestamps at every boundary
    loaded_data = pandas.DataFrame({
        "time": pandas.date_range(
            "2023-06-15 21:00",
            "2023-06-16 04:00",
            freq="h",
        ),
        "value": range(8),
    })

    # Act
    files = splitter.split(loaded_data)

    # Assert: correct number of files
    assert len(files) == 3

    # Assert: exact timestamps in each split
    expected_dates = [
        pandas.date_range(dates[0], dates[1], freq="h"),
        pandas.date_range(dates[2], dates[3], freq="h"),
        pandas.date_range(dates[4], dates[5], freq="h"),
    ]

    actual_dates = [
        pandas.to_datetime(pandas.read_csv(file)["time"])
        for file in files
    ]

    for actual, expected in zip(actual_dates, expected_dates):
        pandas.testing.assert_series_equal(
            actual.reset_index(drop=True),
            pandas.Series(expected, name="time"),
        )

    # Assert: chronological segregation
    assert actual_dates[0].max() < actual_dates[1].min()
    assert actual_dates[1].max() < actual_dates[2].min()