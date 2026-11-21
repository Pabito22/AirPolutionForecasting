# AirPollutionForecasting

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Project aims to predict future PM10 concetration based on historical and future weather and pollution data.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── cleaned        <- Cleaned, combined data (1 step afterr raw).
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
|
├── eda
│   ├── ts_correlation <- utils for aoutocorrelation and lagged correlation of covariates
│   └── ts_stationarity <- utils for checking stationarity of time series
|
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <-  Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g.
|   ├── README.md      <-  Info about config used for EDA puprposes (notebooks 2-4)
│   ├── 1-PW-Data-set-preparation   <- How data was collected from other project
│   ├── 2-PW-initial-EDA    <-EDA, periods,plots, boxplots, correlations, min max values etc
│   ├── 3-PW-..             <- MSTL, periods, trend
│   └── 4-PW ..            <- Autocorrelation, lagged correlation, stationarity test
│
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         airpollutionforecasting and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── airpollutionforecasting   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes airpollutionforecasting a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── data              <- Scripts to download or generate data
    │   ├── __init__.py
    │   ├── data_validator.py    <- Validates weather data meets all criterria for good quality
    │   └── loader.py            <- Loads the data from the local csv file
    |
    ├── features.py             <- Code to create features for modeling
    │
    ├── pipeline.py             <- orchestrator for each of pipeline stepd (load, validate, split,..,predict,..)
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
