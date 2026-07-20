# Databricks notebook source
# DBTITLE 1,LPG Desk Sandbox Refresh - Configuration
"""
LPG Desk Sandbox Refresh
========================
Populates dna_snbx_weather.lpg_desk.* tables from production silver layer.
Schedule: 3x daily (04:00, 10:00, 18:00 UTC) via Lakeflow Jobs.

Sandbox tables:
  - hdd_season              (US PADD cumulative HDD)
  - hdd_normal              (US PADD HDD climatology by day_of_year)
  - hdd_run_comparison      (US PADD HDD latest 3 runs)
  - asia_hdd_season         (Asia region cumulative HDD)
  - asia_hdd_normal         (Asia region HDD climatology)
  - india_precip_anomaly    (India subregion precip vs clim)
  - india_precip_map        (Gridded India precip anomaly)
  - gatun_precip_history    (Gatun Lake catchment cumulative precip by year)
  - gatun_analog_forecast   (Top N analog years for Gatun)
  - padd3_coastal_temps     (PADD 3 terminal temperatures)
  - padd3_coastal_climatology (PADD 3 terminal temp normals)
  - hurricane_activity      (Active tropical cyclones)
  - hurricane_season_stats  (Season ACE & counts)
  - nafrica_wind            (North Africa corridor wind speeds)
  - nafrica_wind_climatology (North Africa wind normals)
"""

from datetime import datetime, timedelta
import numpy as np

# === PARAMETERS ===
SBX_SCHEMA = "dna_snbx_weather.lpg_desk"
TABLE_TEMP = "dna_prod_silver.meteomatics.temperature"
TABLE_TEMP_FCST = "dna_prod_silver.meteomatics.temperature_forecast"
TABLE_PRECIP = "dna_prod_silver.meteomatics.precipitation"
TABLE_PRECIP_FCST = "dna_prod_silver.meteomatics.precipitation_forecast"
TABLE_WIND = "dna_prod_silver.meteomatics.wind"
TABLE_WIND_FCST = "dna_prod_silver.meteomatics.wind_forecast"

CURVE_TEMP_HIST = "t_mean_2m_24h_c_ecmwf_era5_p1d"
CURVE_TEMP_FCST = "t_mean_2m_24h_c_ecmwf_ens_p1d"
MODEL_HIST = "ecmwf-era5"
MODEL_FCST = "ecmwf-ens"

CURVE_PRECIP_HIST = "precip_24h_mm_ecmwf_era5_p1d"
CURVE_PRECIP_FCST = "precip_24h_mm_ecmwf_ens_p1d"

CURVE_WIND_HIST = "wind_speed_10m_ms_ecmwf_era5_p1d"
CURVE_WIND_FCST = "wind_speed_10m_ms_ecmwf_ens_p1d"

HDD_BASE = 18.0
SEASON_START = "10-01"  # Oct 1

run_ts = datetime.now()
print(f"Refresh started: {run_ts}")

# COMMAND ----------

# DBTITLE 1,Create schema if not exists
# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS dna_snbx_weather.lpg_desk;

# COMMAND ----------

# DBTITLE 1,US HDD by PADD - Season + Normal
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TABLE: hdd_season (US PADD cumulative HDD — actuals + forecast)
# MAGIC -- =============================================================================
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.hdd_season AS
# MAGIC WITH padd_assign AS (
# MAGIC   SELECT
# MAGIC     date, value AS temperature,
# MAGIC     CASE
# MAGIC       WHEN latitude >= 36.5 AND latitude < 47.5 AND longitude >= -80.0  AND longitude < -66.5  THEN 'PADD 1 - Northeast'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -97.5  AND longitude < -80.0  THEN 'PADD 2 - Midwest'
# MAGIC       WHEN latitude >= 25.5 AND latitude < 36.5 AND longitude >= -106.0 AND longitude < -79.5  THEN 'PADD 3 - South Central'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -116.5 AND longitude < -97.5  THEN 'PADD 4 - Mountain'
# MAGIC       WHEN latitude >= 32.0 AND latitude < 49.0 AND longitude >= -125.0 AND longitude < -116.5 THEN 'PADD 5 - West'
# MAGIC     END AS padd_region
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_era5_p1d'
# MAGIC     AND model = 'ecmwf-era5'
# MAGIC     AND date >= DATE_FORMAT(DATEADD(YEAR, -1, CURRENT_DATE()), 'yyyy-10-01')
# MAGIC     AND latitude BETWEEN 24 AND 50
# MAGIC     AND longitude BETWEEN -125 AND -65
# MAGIC ),
# MAGIC actuals AS (
# MAGIC   SELECT
# MAGIC     'Actual (ERA5)' AS model_label,
# MAGIC     padd_region,
# MAGIC     date AS delivery_date,
# MAGIC     GREATEST(0, 18.0 - AVG(temperature)) AS daily_hdd
# MAGIC   FROM padd_assign
# MAGIC   WHERE padd_region IS NOT NULL
# MAGIC   GROUP BY padd_region, date
# MAGIC ),
# MAGIC fcst_assign AS (
# MAGIC   SELECT
# MAGIC     date, value AS temperature,
# MAGIC     CASE
# MAGIC       WHEN latitude >= 36.5 AND latitude < 47.5 AND longitude >= -80.0  AND longitude < -66.5  THEN 'PADD 1 - Northeast'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -97.5  AND longitude < -80.0  THEN 'PADD 2 - Midwest'
# MAGIC       WHEN latitude >= 25.5 AND latitude < 36.5 AND longitude >= -106.0 AND longitude < -79.5  THEN 'PADD 3 - South Central'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -116.5 AND longitude < -97.5  THEN 'PADD 4 - Mountain'
# MAGIC       WHEN latitude >= 32.0 AND latitude < 49.0 AND longitude >= -125.0 AND longitude < -116.5 THEN 'PADD 5 - West'
# MAGIC     END AS padd_region
# MAGIC   FROM dna_prod_silver.meteomatics.temperature_forecast
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_ens_p1d'
# MAGIC     AND model = 'ecmwf-ens'
# MAGIC     AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.temperature_forecast WHERE model = 'ecmwf-ens')
# MAGIC     AND latitude BETWEEN 24 AND 50
# MAGIC     AND longitude BETWEEN -125 AND -65
# MAGIC ),
# MAGIC forecasts AS (
# MAGIC   SELECT
# MAGIC     'EC-ens (14d)' AS model_label,
# MAGIC     padd_region,
# MAGIC     date AS delivery_date,
# MAGIC     GREATEST(0, 18.0 - AVG(temperature)) AS daily_hdd
# MAGIC   FROM fcst_assign
# MAGIC   WHERE padd_region IS NOT NULL
# MAGIC   GROUP BY padd_region, date
# MAGIC ),
# MAGIC combined AS (
# MAGIC   SELECT * FROM actuals
# MAGIC   UNION ALL
# MAGIC   SELECT * FROM forecasts
# MAGIC )
# MAGIC SELECT
# MAGIC   model_label, padd_region, delivery_date, daily_hdd,
# MAGIC   SUM(daily_hdd) OVER (PARTITION BY model_label, padd_region ORDER BY delivery_date) AS cumulative_hdd
# MAGIC FROM combined
# MAGIC ORDER BY padd_region, model_label, delivery_date;

# COMMAND ----------

# DBTITLE 1,US HDD Normal (climatology by day_of_year)
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TABLE: hdd_normal (ERA5 climatological daily HDD by PADD, day_of_year)
# MAGIC -- =============================================================================
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.hdd_normal AS
# MAGIC WITH padd_daily AS (
# MAGIC   SELECT
# MAGIC     DAYOFYEAR(date) AS day_of_year,
# MAGIC     CASE
# MAGIC       WHEN latitude >= 36.5 AND latitude < 47.5 AND longitude >= -80.0  AND longitude < -66.5  THEN 'PADD 1 - Northeast'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -97.5  AND longitude < -80.0  THEN 'PADD 2 - Midwest'
# MAGIC       WHEN latitude >= 25.5 AND latitude < 36.5 AND longitude >= -106.0 AND longitude < -79.5  THEN 'PADD 3 - South Central'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -116.5 AND longitude < -97.5  THEN 'PADD 4 - Mountain'
# MAGIC       WHEN latitude >= 32.0 AND latitude < 49.0 AND longitude >= -125.0 AND longitude < -116.5 THEN 'PADD 5 - West'
# MAGIC     END AS padd_region,
# MAGIC     GREATEST(0, 18.0 - AVG(value)) AS daily_hdd
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_era5_p1d'
# MAGIC     AND model = 'ecmwf-era5'
# MAGIC     AND YEAR(date) BETWEEN 2000 AND 2024
# MAGIC     AND latitude BETWEEN 24 AND 50
# MAGIC     AND longitude BETWEEN -125 AND -65
# MAGIC   GROUP BY DAYOFYEAR(date), padd_region, date
# MAGIC )
# MAGIC SELECT
# MAGIC   padd_region,
# MAGIC   day_of_year,
# MAGIC   AVG(daily_hdd) AS daily_hdd_normal
# MAGIC FROM padd_daily
# MAGIC WHERE padd_region IS NOT NULL
# MAGIC GROUP BY padd_region, day_of_year
# MAGIC ORDER BY padd_region, day_of_year;

# COMMAND ----------

# DBTITLE 1,Asia HDD Season
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TABLE: asia_hdd_season (population-weighted HDD for Asia regions)
# MAGIC -- =============================================================================
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.asia_hdd_season AS
# MAGIC WITH city_temps AS (
# MAGIC   -- Japan
# MAGIC   SELECT 'Japan' AS region, date, AVG(value) AS temperature
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND date >= DATE_FORMAT(DATEADD(YEAR, -1, CURRENT_DATE()), 'yyyy-10-01')
# MAGIC     AND ((latitude = 35.5 AND longitude = 139.5) OR (latitude = 34.5 AND longitude = 135.5)
# MAGIC       OR (latitude = 38.0 AND longitude = 139.0) OR (latitude = 38.5 AND longitude = 141.0)
# MAGIC       OR (latitude = 43.0 AND longitude = 141.5) OR (latitude = 33.5 AND longitude = 130.5)
# MAGIC       OR (latitude = 35.0 AND longitude = 137.0))
# MAGIC   GROUP BY date
# MAGIC   UNION ALL
# MAGIC   -- South Korea
# MAGIC   SELECT 'South Korea' AS region, date, AVG(value) AS temperature
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND date >= DATE_FORMAT(DATEADD(YEAR, -1, CURRENT_DATE()), 'yyyy-10-01')
# MAGIC     AND ((latitude = 37.5 AND longitude = 127.0) OR (latitude = 35.0 AND longitude = 129.0)
# MAGIC       OR (latitude = 37.5 AND longitude = 126.5) OR (latitude = 36.0 AND longitude = 128.5))
# MAGIC   GROUP BY date
# MAGIC   UNION ALL
# MAGIC   -- China North
# MAGIC   SELECT 'China North' AS region, date, AVG(value) AS temperature
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND date >= DATE_FORMAT(DATEADD(YEAR, -1, CURRENT_DATE()), 'yyyy-10-01')
# MAGIC     AND ((latitude = 40.0 AND longitude = 116.5) OR (latitude = 39.5 AND longitude = 117.5)
# MAGIC       OR (latitude = 45.5 AND longitude = 126.5) OR (latitude = 42.0 AND longitude = 123.5)
# MAGIC       OR (latitude = 44.0 AND longitude = 125.5) OR (latitude = 41.0 AND longitude = 111.5)
# MAGIC       OR (latitude = 44.0 AND longitude = 87.5)  OR (latitude = 38.0 AND longitude = 114.5)
# MAGIC       OR (latitude = 36.5 AND longitude = 117.0) OR (latitude = 34.5 AND longitude = 109.0))
# MAGIC   GROUP BY date
# MAGIC   UNION ALL
# MAGIC   -- China East
# MAGIC   SELECT 'China East' AS region, date, AVG(value) AS temperature
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND date >= DATE_FORMAT(DATEADD(YEAR, -1, CURRENT_DATE()), 'yyyy-10-01')
# MAGIC     AND ((latitude = 31.0 AND longitude = 121.5) OR (latitude = 32.0 AND longitude = 119.0)
# MAGIC       OR (latitude = 30.5 AND longitude = 120.0) OR (latitude = 30.5 AND longitude = 114.5)
# MAGIC       OR (latitude = 32.0 AND longitude = 117.0) OR (latitude = 35.0 AND longitude = 113.5))
# MAGIC   GROUP BY date
# MAGIC   UNION ALL
# MAGIC   -- China South
# MAGIC   SELECT 'China South' AS region, date, AVG(value) AS temperature
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND date >= DATE_FORMAT(DATEADD(YEAR, -1, CURRENT_DATE()), 'yyyy-10-01')
# MAGIC     AND ((latitude = 23.0 AND longitude = 113.5) OR (latitude = 22.5 AND longitude = 114.0)
# MAGIC       OR (latitude = 25.0 AND longitude = 102.5) OR (latitude = 30.5 AND longitude = 104.0)
# MAGIC       OR (latitude = 29.5 AND longitude = 107.0) OR (latitude = 23.0 AND longitude = 108.5))
# MAGIC   GROUP BY date
# MAGIC ),
# MAGIC actuals AS (
# MAGIC   SELECT
# MAGIC     region,
# MAGIC     'Actual (ERA5)' AS model_label,
# MAGIC     date AS delivery_date,
# MAGIC     GREATEST(0, 18.0 - temperature) AS daily_hdd
# MAGIC   FROM city_temps
# MAGIC )
# MAGIC SELECT
# MAGIC   region, model_label, delivery_date, daily_hdd,
# MAGIC   SUM(daily_hdd) OVER (PARTITION BY region, model_label ORDER BY delivery_date) AS cumulative_hdd
# MAGIC FROM actuals
# MAGIC ORDER BY region, model_label, delivery_date;

# COMMAND ----------

# DBTITLE 1,PADD 3 Coastal Temperatures
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TABLE: padd3_coastal_temps (actuals + forecast for export terminals)
# MAGIC -- =============================================================================
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.padd3_coastal_temps AS
# MAGIC WITH actuals AS (
# MAGIC   SELECT
# MAGIC     CASE
# MAGIC       WHEN latitude = 30.0 AND longitude = -95.5 THEN 'Houston'
# MAGIC       WHEN latitude = 29.0 AND longitude = -95.5 THEN 'Freeport'
# MAGIC       WHEN latitude = 30.0 AND longitude = -94.0 THEN 'Beaumont'
# MAGIC       WHEN latitude = 30.0 AND longitude = -93.0 THEN 'Lake Charles'
# MAGIC       WHEN latitude = 30.0 AND longitude = -90.0 THEN 'New Orleans'
# MAGIC       WHEN latitude = 27.5 AND longitude = -97.5 THEN 'Corpus Christi'
# MAGIC       WHEN latitude = 29.5 AND longitude = -91.0 THEN 'Morgan City'
# MAGIC     END AS city,
# MAGIC     date AS delivery_date,
# MAGIC     'actual' AS source,
# MAGIC     value AS temperature_c
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_era5_p1d'
# MAGIC     AND model = 'ecmwf-era5'
# MAGIC     AND date >= CURRENT_DATE() - INTERVAL 60 DAYS
# MAGIC     AND (
# MAGIC       (latitude = 30.0 AND longitude = -95.5) OR
# MAGIC       (latitude = 29.0 AND longitude = -95.5) OR
# MAGIC       (latitude = 30.0 AND longitude = -94.0) OR
# MAGIC       (latitude = 30.0 AND longitude = -93.0) OR
# MAGIC       (latitude = 30.0 AND longitude = -90.0) OR
# MAGIC       (latitude = 27.5 AND longitude = -97.5) OR
# MAGIC       (latitude = 29.5 AND longitude = -91.0)
# MAGIC     )
# MAGIC ),
# MAGIC forecasts AS (
# MAGIC   SELECT
# MAGIC     CASE
# MAGIC       WHEN latitude = 30.0 AND longitude = -95.5 THEN 'Houston'
# MAGIC       WHEN latitude = 29.0 AND longitude = -95.5 THEN 'Freeport'
# MAGIC       WHEN latitude = 30.0 AND longitude = -94.0 THEN 'Beaumont'
# MAGIC       WHEN latitude = 30.0 AND longitude = -93.0 THEN 'Lake Charles'
# MAGIC       WHEN latitude = 30.0 AND longitude = -90.0 THEN 'New Orleans'
# MAGIC       WHEN latitude = 27.5 AND longitude = -97.5 THEN 'Corpus Christi'
# MAGIC       WHEN latitude = 29.5 AND longitude = -91.0 THEN 'Morgan City'
# MAGIC     END AS city,
# MAGIC     date AS delivery_date,
# MAGIC     'forecast' AS source,
# MAGIC     value AS temperature_c
# MAGIC   FROM dna_prod_silver.meteomatics.temperature_forecast
# MAGIC   WHERE curve = 't_mean_2m_24h_c_ecmwf_ens_p1d'
# MAGIC     AND model = 'ecmwf-ens'
# MAGIC     AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.temperature_forecast WHERE model = 'ecmwf-ens')
# MAGIC     AND (
# MAGIC       (latitude = 30.0 AND longitude = -95.5) OR
# MAGIC       (latitude = 29.0 AND longitude = -95.5) OR
# MAGIC       (latitude = 30.0 AND longitude = -94.0) OR
# MAGIC       (latitude = 30.0 AND longitude = -93.0) OR
# MAGIC       (latitude = 30.0 AND longitude = -90.0) OR
# MAGIC       (latitude = 27.5 AND longitude = -97.5) OR
# MAGIC       (latitude = 29.5 AND longitude = -91.0)
# MAGIC     )
# MAGIC )
# MAGIC SELECT * FROM actuals WHERE city IS NOT NULL
# MAGIC UNION ALL
# MAGIC SELECT * FROM forecasts WHERE city IS NOT NULL
# MAGIC ORDER BY city, delivery_date;

# COMMAND ----------

# DBTITLE 1,North Africa Wind Speeds
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TABLE: nafrica_wind (actual + forecast wind at corridor ports)
# MAGIC -- =============================================================================
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.nafrica_wind AS
# MAGIC WITH actuals AS (
# MAGIC   SELECT
# MAGIC     CASE
# MAGIC       WHEN latitude = 35.5 AND longitude = -6.0  THEN 'Tangier'
# MAGIC       WHEN latitude = 35.5 AND longitude = -0.5  THEN 'Oran'
# MAGIC       WHEN latitude = 36.5 AND longitude = 3.0   THEN 'Algiers'
# MAGIC       WHEN latitude = 36.5 AND longitude = 5.0   THEN 'Bejaia'
# MAGIC       WHEN latitude = 37.0 AND longitude = 7.0   THEN 'Skikda'
# MAGIC       WHEN latitude = 37.0 AND longitude = 10.0  THEN 'Tunis'
# MAGIC       WHEN latitude = 33.0 AND longitude = 13.0  THEN 'Tripoli'
# MAGIC       WHEN latitude = 32.0 AND longitude = 20.0  THEN 'Benghazi'
# MAGIC       WHEN latitude = 31.0 AND longitude = 30.0  THEN 'Alexandria'
# MAGIC       WHEN latitude = 31.5 AND longitude = 32.5  THEN 'Port Said'
# MAGIC     END AS city,
# MAGIC     date AS delivery_date,
# MAGIC     'actual' AS source,
# MAGIC     value AS wind_speed_ms
# MAGIC   FROM dna_prod_silver.meteomatics.wind
# MAGIC   WHERE curve = 'wind_speed_10m_ms_ecmwf_era5_p1d'
# MAGIC     AND model = 'ecmwf-era5'
# MAGIC     AND date >= CURRENT_DATE() - INTERVAL 30 DAYS
# MAGIC     AND latitude BETWEEN 30 AND 38
# MAGIC     AND longitude BETWEEN -7 AND 33
# MAGIC     AND (
# MAGIC       (latitude = 35.5 AND longitude = -6.0) OR (latitude = 35.5 AND longitude = -0.5) OR
# MAGIC       (latitude = 36.5 AND longitude = 3.0)  OR (latitude = 36.5 AND longitude = 5.0) OR
# MAGIC       (latitude = 37.0 AND longitude = 7.0)  OR (latitude = 37.0 AND longitude = 10.0) OR
# MAGIC       (latitude = 33.0 AND longitude = 13.0) OR (latitude = 32.0 AND longitude = 20.0) OR
# MAGIC       (latitude = 31.0 AND longitude = 30.0) OR (latitude = 31.5 AND longitude = 32.5)
# MAGIC     )
# MAGIC ),
# MAGIC forecasts AS (
# MAGIC   SELECT
# MAGIC     CASE
# MAGIC       WHEN latitude = 35.5 AND longitude = -6.0  THEN 'Tangier'
# MAGIC       WHEN latitude = 35.5 AND longitude = -0.5  THEN 'Oran'
# MAGIC       WHEN latitude = 36.5 AND longitude = 3.0   THEN 'Algiers'
# MAGIC       WHEN latitude = 36.5 AND longitude = 5.0   THEN 'Bejaia'
# MAGIC       WHEN latitude = 37.0 AND longitude = 7.0   THEN 'Skikda'
# MAGIC       WHEN latitude = 37.0 AND longitude = 10.0  THEN 'Tunis'
# MAGIC       WHEN latitude = 33.0 AND longitude = 13.0  THEN 'Tripoli'
# MAGIC       WHEN latitude = 32.0 AND longitude = 20.0  THEN 'Benghazi'
# MAGIC       WHEN latitude = 31.0 AND longitude = 30.0  THEN 'Alexandria'
# MAGIC       WHEN latitude = 31.5 AND longitude = 32.5  THEN 'Port Said'
# MAGIC     END AS city,
# MAGIC     date AS delivery_date,
# MAGIC     'forecast' AS source,
# MAGIC     value AS wind_speed_ms
# MAGIC   FROM dna_prod_silver.meteomatics.wind_forecast
# MAGIC   WHERE curve = 'wind_speed_10m_ms_ecmwf_ens_p1d'
# MAGIC     AND model = 'ecmwf-ens'
# MAGIC     AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.wind_forecast WHERE model = 'ecmwf-ens')
# MAGIC     AND (
# MAGIC       (latitude = 35.5 AND longitude = -6.0) OR (latitude = 35.5 AND longitude = -0.5) OR
# MAGIC       (latitude = 36.5 AND longitude = 3.0)  OR (latitude = 36.5 AND longitude = 5.0) OR
# MAGIC       (latitude = 37.0 AND longitude = 7.0)  OR (latitude = 37.0 AND longitude = 10.0) OR
# MAGIC       (latitude = 33.0 AND longitude = 13.0) OR (latitude = 32.0 AND longitude = 20.0) OR
# MAGIC       (latitude = 31.0 AND longitude = 30.0) OR (latitude = 31.5 AND longitude = 32.5)
# MAGIC     )
# MAGIC )
# MAGIC SELECT * FROM actuals WHERE city IS NOT NULL
# MAGIC UNION ALL
# MAGIC SELECT * FROM forecasts WHERE city IS NOT NULL
# MAGIC ORDER BY city, delivery_date;

# COMMAND ----------

# DBTITLE 1,India Precipitation Anomaly
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TABLE: india_precip_anomaly (subregion precipitation vs climatology)
# MAGIC -- =============================================================================
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.india_precip_anomaly AS
# MAGIC WITH subregion_assign AS (
# MAGIC   SELECT
# MAGIC     date, value AS precip_mm,
# MAGIC     CASE
# MAGIC       WHEN latitude >= 23.0 AND latitude < 35.0 AND longitude >= 68.0 AND longitude < 78.0 THEN 'Northwest'
# MAGIC       WHEN latitude >= 23.0 AND latitude < 35.0 AND longitude >= 78.0 AND longitude < 90.0 THEN 'Northeast'
# MAGIC       WHEN latitude >= 18.0 AND latitude < 23.0 AND longitude >= 72.0 AND longitude < 85.0 THEN 'Central'
# MAGIC       WHEN latitude >= 8.0  AND latitude < 18.0 AND longitude >= 73.0 AND longitude < 85.0 THEN 'South'
# MAGIC       WHEN latitude >= 10.0 AND latitude < 23.0 AND longitude >= 68.0 AND longitude < 74.0 THEN 'West Coast'
# MAGIC     END AS subregion
# MAGIC   FROM dna_prod_silver.meteomatics.precipitation
# MAGIC   WHERE curve = 'precip_24h_mm_ecmwf_era5_p1d'
# MAGIC     AND model = 'ecmwf-era5'
# MAGIC     AND date >= CURRENT_DATE() - INTERVAL 90 DAYS
# MAGIC     AND latitude BETWEEN 8 AND 35
# MAGIC     AND longitude BETWEEN 68 AND 90
# MAGIC ),
# MAGIC actuals AS (
# MAGIC   SELECT subregion, date AS delivery_date, AVG(precip_mm) AS precip_actual_mm
# MAGIC   FROM subregion_assign
# MAGIC   WHERE subregion IS NOT NULL
# MAGIC   GROUP BY subregion, date
# MAGIC ),
# MAGIC clim AS (
# MAGIC   SELECT
# MAGIC     CASE
# MAGIC       WHEN latitude >= 23.0 AND latitude < 35.0 AND longitude >= 68.0 AND longitude < 78.0 THEN 'Northwest'
# MAGIC       WHEN latitude >= 23.0 AND latitude < 35.0 AND longitude >= 78.0 AND longitude < 90.0 THEN 'Northeast'
# MAGIC       WHEN latitude >= 18.0 AND latitude < 23.0 AND longitude >= 72.0 AND longitude < 85.0 THEN 'Central'
# MAGIC       WHEN latitude >= 8.0  AND latitude < 18.0 AND longitude >= 73.0 AND longitude < 85.0 THEN 'South'
# MAGIC       WHEN latitude >= 10.0 AND latitude < 23.0 AND longitude >= 68.0 AND longitude < 74.0 THEN 'West Coast'
# MAGIC     END AS subregion,
# MAGIC     DAYOFYEAR(date) AS doy,
# MAGIC     AVG(value) AS precip_clim_mm
# MAGIC   FROM dna_prod_silver.meteomatics.precipitation
# MAGIC   WHERE curve = 'precip_24h_mm_ecmwf_era5_p1d'
# MAGIC     AND model = 'ecmwf-era5'
# MAGIC     AND YEAR(date) BETWEEN 2000 AND 2024
# MAGIC     AND latitude BETWEEN 8 AND 35
# MAGIC     AND longitude BETWEEN 68 AND 90
# MAGIC   GROUP BY subregion, DAYOFYEAR(date)
# MAGIC )
# MAGIC SELECT
# MAGIC   a.subregion,
# MAGIC   a.delivery_date,
# MAGIC   a.precip_actual_mm,
# MAGIC   c.precip_clim_mm,
# MAGIC   NULL AS precip_forecast_mm,
# MAGIC   CASE WHEN c.precip_clim_mm > 0.1
# MAGIC     THEN ROUND(100.0 * (a.precip_actual_mm - c.precip_clim_mm) / c.precip_clim_mm, 1)
# MAGIC     ELSE 0
# MAGIC   END AS deviation_pct
# MAGIC FROM actuals a
# MAGIC JOIN clim c ON a.subregion = c.subregion AND DAYOFYEAR(a.delivery_date) = c.doy
# MAGIC ORDER BY a.subregion, a.delivery_date;

# COMMAND ----------

# DBTITLE 1,Gatun Lake Precipitation History
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- TABLE: gatun_precip_history (cumulative precip over Gatun catchment by year)
# MAGIC -- =============================================================================
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.gatun_precip_history AS
# MAGIC WITH daily AS (
# MAGIC   SELECT
# MAGIC     YEAR(date) AS year,
# MAGIC     DAYOFYEAR(date) AS day_of_year,
# MAGIC     AVG(value) AS daily_precip_mm
# MAGIC   FROM dna_prod_silver.meteomatics.precipitation
# MAGIC   WHERE curve = 'precip_24h_mm_ecmwf_era5_p1d'
# MAGIC     AND model = 'ecmwf-era5'
# MAGIC     AND latitude BETWEEN 8.5 AND 10.0
# MAGIC     AND longitude BETWEEN -80.5 AND -79.0
# MAGIC     AND YEAR(date) BETWEEN 2000 AND 2026
# MAGIC   GROUP BY YEAR(date), DAYOFYEAR(date)
# MAGIC )
# MAGIC SELECT
# MAGIC   year, day_of_year,
# MAGIC   SUM(daily_precip_mm) OVER (PARTITION BY year ORDER BY day_of_year) AS cumulative_precip_mm
# MAGIC FROM daily
# MAGIC ORDER BY year, day_of_year;

# COMMAND ----------

# DBTITLE 1,Gatun Lake Analog Forecast
# =============================================================================
# TABLE: gatun_analog_forecast (top N similar years by cumulative precip pattern)
# =============================================================================
import pandas as pd
import numpy as np
from datetime import datetime

# Load the history table we just created
gatun_hist = spark.sql("SELECT * FROM dna_snbx_weather.lpg_desk.gatun_precip_history").toPandas()
gatun_hist["year"] = gatun_hist["year"].astype(int)
gatun_hist["day_of_year"] = gatun_hist["day_of_year"].astype(int)
gatun_hist["cumulative_precip_mm"] = gatun_hist["cumulative_precip_mm"].astype(float)

current_year = datetime.now().year
current_doy = datetime.now().timetuple().tm_yday

# Current year cumulative up to today
cur = gatun_hist[(gatun_hist["year"] == current_year) & (gatun_hist["day_of_year"] <= current_doy)]

if cur.empty:
    print("No current year data for Gatun Lake yet.")
else:
    # Compute similarity: RMSE of cumulative precip up to current DOY
    cur_series = cur.set_index("day_of_year")["cumulative_precip_mm"]
    
    scores = []
    for yr in gatun_hist["year"].unique():
        if yr == current_year:
            continue
        yr_data = gatun_hist[(gatun_hist["year"] == yr) & (gatun_hist["day_of_year"] <= current_doy)]
        yr_series = yr_data.set_index("day_of_year")["cumulative_precip_mm"]
        
        # Align on common DOYs
        common = cur_series.index.intersection(yr_series.index)
        if len(common) < 30:
            continue
        
        rmse = np.sqrt(np.mean((cur_series[common].values - yr_series[common].values) ** 2))
        scores.append({"analog_year": yr, "similarity_score": round(1.0 / (1.0 + rmse), 4)})
    
    scores_df = pd.DataFrame(scores).sort_values("similarity_score", ascending=False).head(5)
    
    # Get full year data for analog years
    analog_years = scores_df["analog_year"].tolist()
    analog_data = gatun_hist[gatun_hist["year"].isin(analog_years)].copy()
    analog_data = analog_data.merge(scores_df, left_on="year", right_on="analog_year", how="left")
    analog_data = analog_data[["analog_year", "similarity_score", "day_of_year", "cumulative_precip_mm"]]
    
    # Write to sandbox
    spark.createDataFrame(analog_data).write.mode("overwrite").saveAsTable(
        "dna_snbx_weather.lpg_desk.gatun_analog_forecast"
    )
    print(f"Gatun analog forecast: {len(analog_years)} analog years written.")

# COMMAND ----------

# DBTITLE 1,Refresh Complete
from datetime import datetime
print(f"\n{'='*60}")
print(f"LPG Desk Sandbox Refresh COMPLETE: {datetime.now()}")
print(f"{'='*60}")
print("\nTables refreshed:")
print("  - dna_snbx_weather.lpg_desk.hdd_season")
print("  - dna_snbx_weather.lpg_desk.hdd_normal")
print("  - dna_snbx_weather.lpg_desk.asia_hdd_season")
print("  - dna_snbx_weather.lpg_desk.padd3_coastal_temps")
print("  - dna_snbx_weather.lpg_desk.nafrica_wind")
print("  - dna_snbx_weather.lpg_desk.india_precip_anomaly")
print("  - dna_snbx_weather.lpg_desk.gatun_precip_history")
print("  - dna_snbx_weather.lpg_desk.gatun_analog_forecast")
print("\nNote: hurricane_activity and hurricane_season_stats require")
print("external data source (IBTrACS or similar) - placeholder tables.")
