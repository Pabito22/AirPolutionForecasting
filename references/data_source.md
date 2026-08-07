# Data Sources

## Raw Data

### 1. AGH Weather Measurements

**Source:** AGH METEO Serwis: http://meteo.ftj.agh.edu.pl/loginPage

Historical meteorological measurements collected by the Department of Environmental Physics at AGH University. Stored as `data/raw/combined.csv`.

### 2. Open-Meteo Historical Weather Data

**Source:** [Open-Meteo-API](https://open-meteo.com/en/docs/historical-weather-api?hourly=temperature_2m,dew_point_2m,relative_humidity_2m,surface_pressure,pressure_msl,wind_direction_10m,wind_speed_10m,cloud_cover_low&start_date=2016-01-01&end_date=2025-12-31&latitude=50.06143&longitude=19.93658&wind_speed_unit=ms&timezone=Europe%2FBerlin)

Historical weather observations downloaded from the Open-Meteo API and used to fill missing values in the AGH dataset. Stored as `data/raw/open-meteo.csv`.

### 3. GIOŚ PM10 Measurements

**Source:** https://powietrze.gios.gov.pl/pjp/archives

Hourly PM10 concentrations measured by three GIOŚ air quality monitoring stations in Kraków. Stored as `data/raw/other_data_set.csv`.

## Processed Data

The notebook `notebooks/Data set preparation.ipynb` documents the complete preprocessing pipeline used to generate the final modeling dataset, `data/processed/training_dataset.csv`, from the raw data sources listed above.
