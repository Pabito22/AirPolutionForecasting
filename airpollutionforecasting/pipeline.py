"""Orchestrator for each pipeline steps."""

from typing import Literal

from airpollutionforecasting.data.loader import DataLoader
from airpollutionforecasting.data.data_validator import DataValidator
from airpollutionforecasting.data.splitter import DataSplitter
from airpollutionforecasting.config import DatasetConfig

from loguru import logger

def run_pipeline(stop_step: Literal["load", "validate", "split"] = "split")->None:
    """Run pipeline till the stop_step
    
    Params:
     stop_step: Allowed values["load", "validate", "split"]
        load - loads the data
        validate - validates if the data is clean
        split - splits data into train, validation and test sets. 
     """
    logger.info(f"Running pipeline till the step {stop_step}...")

    #configuration
    data_config = DatasetConfig()
    
    # 1. Load
    loader = DataLoader(data_config=data_config)
    loaded_data = loader.load()

    # 2. Validate
    validator = DataValidator(dataset_config=data_config)
    validator.validate(loaded_data=loaded_data)

    # 3. Split
    splitter = DataSplitter
    split_files = splitter.split(loaded_data=loaded_data)

    return 


if __name__ == "__main__":
    run_pipeline()