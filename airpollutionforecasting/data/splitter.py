"""Split data into train, validation and test set according to configuration."""

from pathlib import Path
from datetime import timedelta
import pandas

from airpollutionforecasting.config import DatasetConfig

class DataSplitter:
    """Split data into train, validation and test set according to configuration."""

    def __init__(self, data_config: DatasetConfig):
        """Initialize splitter."""
        self.config = data_config

    def split(self, loaded_data: pandas.DataFrame) -> list[str]:
        """Split the data into train, val, test set and returns the location of result.

        Params:
        loaded_data: Data that weant troough DataLoader step.
        
        Notes:
        Firstly checks if the given configuration is correct (chronological, 
        continous, included in the min and max dates present in the configuration).
        Then splits the data as specified in the configuration file and 
        saves them in `data/loaded/[train/validation/test]_startdate_enddate_cleaned.csv`.
        Assumed format from configuration: 
            [train_start, train_end, val_start, val_end, test_start, test_end].
        If the files already exist, location of the splited data will be returned.
        
        Returns:
            data_split_locations (list[str]): [train, val, test] cleaned data locations."""

        # generate file names
        split_dates = self.config.data_split_dates
        split_files :list[str] = [
            f"data/loaded/train_{str(split_dates[0])}_{str(split_dates[1])}_cleaned.csv",
            f"data/loaded/validation_{str(split_dates[2])}_{str(split_dates[3])}_cleaned.csv",
            f"data/loaded/test_{str(split_dates[4])}_{str(split_dates[5])}_cleaned.csv"
            ]

        #if files exist, return generated list of paths
        if all(Path(file).is_file() for file in split_files):
            return split_files


        # check if config correct
        #check for continouity
        for i in range(1,3):
            if split_dates[2*i]-split_dates[2*i-1] != timedelta(hours=1):
                raise ValueError("""Dates from configuration are not continous 
                    (1 hour after another betwen train/val or val/test)""")
        #check for chronoligality 
        for i in range(5):
            if split_dates[i+1] < split_dates[i]:
                raise ValueError("Split dates are not chronological in cofiguration!")
        # check if the dates are between min and max dates specidied in configuration
        if split_dates[0] < self.config.min_date or split_dates[-1] > self.config.max_date:
            raise ValueError("Dates specified in the data_split_dates are not between max and min dates!")
        

        # TODO: split data and save them in generated names

        return split_files
    