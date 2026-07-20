# Databricks notebook source
# DBTITLE 1,LPG Desk Sandbox Refresh v2
"""
LPG Desk Sandbox Refresh v2
===========================
Refreshes all dna_snbx_weather.lpg_desk.* tables.
Schedule: every 6 hours via Lakeflow Jobs.

Correct column names: curve_name, delivery_start (not curve, date)
"""
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window

print(f"Refresh started: {datetime.now()}")
spark.sql("CREATE SCHEMA IF NOT EXISTS dna_snbx_weather.lpg_desk")

# COMMAND ----------

# DBTITLE 1,1. US HDD Season + Normal
# MAGIC %sql
# MAGIC -- HDD Season (actuals + forecast)
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.hdd_season AS
# MAGIC WITH padd_assign AS (
# MAGIC   SELECT DATE(delivery_start) AS dt, value AS temperature,
# MAGIC     CASE
# MAGIC       WHEN latitude >= 36.5 AND latitude < 47.5 AND longitude >= -80.0  AND longitude < -66.5  THEN 'PADD 1 - Northeast'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -97.5  AND longitude < -80.0  THEN 'PADD 2 - Midwest'
# MAGIC       WHEN latitude >= 25.5 AND latitude < 36.5 AND longitude >= -106.0 AND longitude < -79.5  THEN 'PADD 3 - South Central'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -116.5 AND longitude < -97.5  THEN 'PADD 4 - Mountain'
# MAGIC       WHEN latitude >= 32.0 AND latitude < 49.0 AND longitude >= -125.0 AND longitude < -116.5 THEN 'PADD 5 - West'
# MAGIC     END AS padd_region
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND delivery_start >= '2024-10-01' AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -65
# MAGIC ),
# MAGIC actuals AS (
# MAGIC   SELECT 'Actual (ERA5)' AS model_label, padd_region, dt AS delivery_date, GREATEST(0, 18.0 - AVG(temperature)) AS daily_hdd
# MAGIC   FROM padd_assign WHERE padd_region IS NOT NULL GROUP BY padd_region, dt
# MAGIC ),
# MAGIC fcst_assign AS (
# MAGIC   SELECT DATE(delivery_start) AS dt, value AS temperature,
# MAGIC     CASE
# MAGIC       WHEN latitude >= 36.5 AND latitude < 47.5 AND longitude >= -80.0  AND longitude < -66.5  THEN 'PADD 1 - Northeast'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -97.5  AND longitude < -80.0  THEN 'PADD 2 - Midwest'
# MAGIC       WHEN latitude >= 25.5 AND latitude < 36.5 AND longitude >= -106.0 AND longitude < -79.5  THEN 'PADD 3 - South Central'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -116.5 AND longitude < -97.5  THEN 'PADD 4 - Mountain'
# MAGIC       WHEN latitude >= 32.0 AND latitude < 49.0 AND longitude >= -125.0 AND longitude < -116.5 THEN 'PADD 5 - West'
# MAGIC     END AS padd_region
# MAGIC   FROM dna_prod_silver.meteomatics.temperature_forecast
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d' AND model = 'ecmwf-ens'
# MAGIC     AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.temperature_forecast WHERE model = 'ecmwf-ens')
# MAGIC     AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -65
# MAGIC ),
# MAGIC forecasts AS (
# MAGIC   SELECT 'EC-ens (14d)' AS model_label, padd_region, dt AS delivery_date, GREATEST(0, 18.0 - AVG(temperature)) AS daily_hdd
# MAGIC   FROM fcst_assign WHERE padd_region IS NOT NULL GROUP BY padd_region, dt
# MAGIC ),
# MAGIC combined AS (SELECT * FROM actuals UNION ALL SELECT * FROM forecasts)
# MAGIC SELECT model_label, padd_region, delivery_date, daily_hdd,
# MAGIC   SUM(daily_hdd) OVER (PARTITION BY model_label, padd_region ORDER BY delivery_date) AS cumulative_hdd
# MAGIC FROM combined ORDER BY padd_region, model_label, delivery_date;
# MAGIC
# MAGIC -- HDD Normal
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.hdd_normal AS
# MAGIC WITH padd_daily AS (
# MAGIC   SELECT DAYOFYEAR(delivery_start) AS day_of_year, DATE(delivery_start) AS dt,
# MAGIC     CASE
# MAGIC       WHEN latitude >= 36.5 AND latitude < 47.5 AND longitude >= -80.0  AND longitude < -66.5  THEN 'PADD 1 - Northeast'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -97.5  AND longitude < -80.0  THEN 'PADD 2 - Midwest'
# MAGIC       WHEN latitude >= 25.5 AND latitude < 36.5 AND longitude >= -106.0 AND longitude < -79.5  THEN 'PADD 3 - South Central'
# MAGIC       WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -116.5 AND longitude < -97.5  THEN 'PADD 4 - Mountain'
# MAGIC       WHEN latitude >= 32.0 AND latitude < 49.0 AND longitude >= -125.0 AND longitude < -116.5 THEN 'PADD 5 - West'
# MAGIC     END AS padd_region, value AS temperature
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND YEAR(delivery_start) BETWEEN 2000 AND 2024
# MAGIC     AND (DAYOFYEAR(delivery_start) >= 274 OR DAYOFYEAR(delivery_start) <= 90)
# MAGIC     AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -65
# MAGIC ),
# MAGIC daily_agg AS (
# MAGIC   SELECT padd_region, day_of_year, dt, GREATEST(0, 18.0 - AVG(temperature)) AS daily_hdd
# MAGIC   FROM padd_daily WHERE padd_region IS NOT NULL GROUP BY padd_region, day_of_year, dt
# MAGIC )
# MAGIC SELECT padd_region, day_of_year, AVG(daily_hdd) AS daily_hdd_normal
# MAGIC FROM daily_agg GROUP BY padd_region, day_of_year ORDER BY padd_region, day_of_year;

# COMMAND ----------

# DBTITLE 1,2. US HDD Run Comparison (latest 3 runs)
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window

result_df = spark.sql("""
SELECT 'EC-ens' AS model_label, created_at AS run_ts, padd_region,
  DATE(delivery_start) AS delivery_date, GREATEST(0, 18.0 - AVG(value)) AS daily_hdd
FROM (
  SELECT *, CASE
    WHEN latitude >= 36.5 AND latitude < 47.5 AND longitude >= -80.0  AND longitude < -66.5  THEN 'PADD 1 - Northeast'
    WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -97.5  AND longitude < -80.0  THEN 'PADD 2 - Midwest'
    WHEN latitude >= 25.5 AND latitude < 36.5 AND longitude >= -106.0 AND longitude < -79.5  THEN 'PADD 3 - South Central'
    WHEN latitude >= 36.5 AND latitude < 49.0 AND longitude >= -116.5 AND longitude < -97.5  THEN 'PADD 4 - Mountain'
    WHEN latitude >= 32.0 AND latitude < 49.0 AND longitude >= -125.0 AND longitude < -116.5 THEN 'PADD 5 - West'
  END AS padd_region
  FROM dna_prod_silver.meteomatics.temperature_forecast
  WHERE model = 'ecmwf-ens' AND curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d'
    AND created_at IN (
      SELECT created_at FROM (
        SELECT DISTINCT created_at FROM dna_prod_silver.meteomatics.temperature_forecast
        WHERE model = 'ecmwf-ens' AND curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d'
        ORDER BY created_at DESC LIMIT 3))
    AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -65
) sub WHERE padd_region IS NOT NULL
GROUP BY created_at, padd_region, DATE(delivery_start)
""")

w = Window.orderBy(F.desc("run_ts"))
result_df = (
    result_df
    .withColumn("run_rank", F.dense_rank().over(w).cast("int"))
    .withColumn("run_label", F.when(F.col("run_rank")==1, F.lit("Latest"))
                              .when(F.col("run_rank")==2, F.lit("-12h")).otherwise(F.lit("-24h")))
    .withColumn("snapshot_ts", F.lit(datetime.now()))
    .select("model_label","run_ts","run_rank","run_label","padd_region","delivery_date","daily_hdd","snapshot_ts")
)
result_df.write.mode("overwrite").insertInto("dna_snbx_weather.lpg_desk.hdd_run_comparison")
print(f"hdd_run_comparison: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.hdd_run_comparison').collect()[0][0]} rows")

# COMMAND ----------

# DBTITLE 1,3. Asia HDD Season
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.asia_hdd_season AS
# MAGIC WITH city_temps AS (
# MAGIC   SELECT 'Japan' AS region, DATE(delivery_start) AS dt, AVG(value) AS temperature
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND delivery_start >= '2024-10-01'
# MAGIC     AND ((latitude = 35.5 AND longitude = 139.5) OR (latitude = 34.5 AND longitude = 135.5)
# MAGIC       OR (latitude = 38.0 AND longitude = 139.0) OR (latitude = 43.0 AND longitude = 141.5)
# MAGIC       OR (latitude = 33.5 AND longitude = 130.5) OR (latitude = 35.0 AND longitude = 137.0))
# MAGIC   GROUP BY DATE(delivery_start)
# MAGIC   UNION ALL
# MAGIC   SELECT 'South Korea', DATE(delivery_start), AVG(value)
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND delivery_start >= '2024-10-01'
# MAGIC     AND ((latitude = 37.5 AND longitude = 127.0) OR (latitude = 35.0 AND longitude = 129.0)
# MAGIC       OR (latitude = 37.5 AND longitude = 126.5))
# MAGIC   GROUP BY DATE(delivery_start)
# MAGIC   UNION ALL
# MAGIC   SELECT 'China North', DATE(delivery_start), AVG(value)
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND delivery_start >= '2024-10-01'
# MAGIC     AND ((latitude = 40.0 AND longitude = 116.5) OR (latitude = 45.5 AND longitude = 126.5)
# MAGIC       OR (latitude = 42.0 AND longitude = 123.5) OR (latitude = 38.0 AND longitude = 114.5)
# MAGIC       OR (latitude = 36.5 AND longitude = 117.0))
# MAGIC   GROUP BY DATE(delivery_start)
# MAGIC   UNION ALL
# MAGIC   SELECT 'China East', DATE(delivery_start), AVG(value)
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND delivery_start >= '2024-10-01'
# MAGIC     AND ((latitude = 31.0 AND longitude = 121.5) OR (latitude = 32.0 AND longitude = 119.0)
# MAGIC       OR (latitude = 30.5 AND longitude = 114.5) OR (latitude = 35.0 AND longitude = 113.5))
# MAGIC   GROUP BY DATE(delivery_start)
# MAGIC   UNION ALL
# MAGIC   SELECT 'China South', DATE(delivery_start), AVG(value)
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND delivery_start >= '2024-10-01'
# MAGIC     AND ((latitude = 23.0 AND longitude = 113.5) OR (latitude = 25.0 AND longitude = 102.5)
# MAGIC       OR (latitude = 30.5 AND longitude = 104.0) OR (latitude = 29.5 AND longitude = 107.0))
# MAGIC   GROUP BY DATE(delivery_start)
# MAGIC )
# MAGIC SELECT region, 'Actual (ERA5)' AS model_label, dt AS delivery_date,
# MAGIC   GREATEST(0, 18.0 - temperature) AS daily_hdd,
# MAGIC   SUM(GREATEST(0, 18.0 - temperature)) OVER (PARTITION BY region ORDER BY dt) AS cumulative_hdd
# MAGIC FROM city_temps ORDER BY region, dt;

# COMMAND ----------

# DBTITLE 1,4. PADD3 Coastal + N.Africa Wind + India Precip + Gatun
# MAGIC %sql
# MAGIC -- PADD3 Coastal Temps
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.padd3_coastal_temps AS
# MAGIC WITH actuals AS (
# MAGIC   SELECT CASE
# MAGIC     WHEN latitude = 30.0 AND longitude = -95.5 THEN 'Houston'
# MAGIC     WHEN latitude = 29.0 AND longitude = -95.5 THEN 'Freeport'
# MAGIC     WHEN latitude = 30.0 AND longitude = -94.0 THEN 'Beaumont'
# MAGIC     WHEN latitude = 30.0 AND longitude = -93.0 THEN 'Lake Charles'
# MAGIC     WHEN latitude = 30.0 AND longitude = -90.0 THEN 'New Orleans'
# MAGIC     WHEN latitude = 27.5 AND longitude = -97.5 THEN 'Corpus Christi'
# MAGIC     WHEN latitude = 29.5 AND longitude = -91.0 THEN 'Morgan City'
# MAGIC   END AS city, DATE(delivery_start) AS delivery_date, 'actual' AS source, value AS temperature_c
# MAGIC   FROM dna_prod_silver.meteomatics.temperature
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND delivery_start >= CURRENT_DATE() - INTERVAL 60 DAYS
# MAGIC     AND ((latitude=30.0 AND longitude=-95.5) OR (latitude=29.0 AND longitude=-95.5)
# MAGIC       OR (latitude=30.0 AND longitude=-94.0) OR (latitude=30.0 AND longitude=-93.0)
# MAGIC       OR (latitude=30.0 AND longitude=-90.0) OR (latitude=27.5 AND longitude=-97.5)
# MAGIC       OR (latitude=29.5 AND longitude=-91.0))
# MAGIC ),
# MAGIC forecasts AS (
# MAGIC   SELECT CASE
# MAGIC     WHEN latitude = 30.0 AND longitude = -95.5 THEN 'Houston'
# MAGIC     WHEN latitude = 29.0 AND longitude = -95.5 THEN 'Freeport'
# MAGIC     WHEN latitude = 30.0 AND longitude = -94.0 THEN 'Beaumont'
# MAGIC     WHEN latitude = 30.0 AND longitude = -93.0 THEN 'Lake Charles'
# MAGIC     WHEN latitude = 30.0 AND longitude = -90.0 THEN 'New Orleans'
# MAGIC     WHEN latitude = 27.5 AND longitude = -97.5 THEN 'Corpus Christi'
# MAGIC     WHEN latitude = 29.5 AND longitude = -91.0 THEN 'Morgan City'
# MAGIC   END AS city, DATE(delivery_start) AS delivery_date, 'forecast' AS source, value AS temperature_c
# MAGIC   FROM dna_prod_silver.meteomatics.temperature_forecast
# MAGIC   WHERE curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d' AND model = 'ecmwf-ens'
# MAGIC     AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.temperature_forecast WHERE model='ecmwf-ens')
# MAGIC     AND ((latitude=30.0 AND longitude=-95.5) OR (latitude=29.0 AND longitude=-95.5)
# MAGIC       OR (latitude=30.0 AND longitude=-94.0) OR (latitude=30.0 AND longitude=-93.0)
# MAGIC       OR (latitude=30.0 AND longitude=-90.0) OR (latitude=27.5 AND longitude=-97.5)
# MAGIC       OR (latitude=29.5 AND longitude=-91.0))
# MAGIC )
# MAGIC SELECT * FROM actuals WHERE city IS NOT NULL
# MAGIC UNION ALL SELECT * FROM forecasts WHERE city IS NOT NULL
# MAGIC ORDER BY city, delivery_date;
# MAGIC
# MAGIC -- N.Africa Wind (100m forecast)
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.nafrica_wind AS
# MAGIC WITH wind_raw AS (
# MAGIC   SELECT CASE
# MAGIC       WHEN latitude BETWEEN 35.0 AND 36.0 AND longitude BETWEEN -6.5 AND -5.5 THEN 'Tangier'
# MAGIC       WHEN latitude BETWEEN 35.0 AND 36.0 AND longitude BETWEEN -1.0 AND 0.0  THEN 'Oran'
# MAGIC       WHEN latitude BETWEEN 36.0 AND 37.0 AND longitude BETWEEN 2.5 AND 3.5   THEN 'Algiers'
# MAGIC       WHEN latitude BETWEEN 36.0 AND 37.0 AND longitude BETWEEN 4.5 AND 5.5   THEN 'Bejaia'
# MAGIC       WHEN latitude BETWEEN 36.5 AND 37.5 AND longitude BETWEEN 6.5 AND 7.5   THEN 'Skikda'
# MAGIC       WHEN latitude BETWEEN 36.5 AND 37.5 AND longitude BETWEEN 9.5 AND 10.5  THEN 'Tunis'
# MAGIC       WHEN latitude BETWEEN 32.5 AND 33.5 AND longitude BETWEEN 12.5 AND 13.5 THEN 'Tripoli'
# MAGIC       WHEN latitude BETWEEN 31.5 AND 32.5 AND longitude BETWEEN 19.5 AND 20.5 THEN 'Benghazi'
# MAGIC       WHEN latitude BETWEEN 30.5 AND 31.5 AND longitude BETWEEN 29.5 AND 30.5 THEN 'Alexandria'
# MAGIC       WHEN latitude BETWEEN 31.0 AND 32.0 AND longitude BETWEEN 32.0 AND 33.0 THEN 'Port Said'
# MAGIC     END AS city, DATE(delivery_start) AS delivery_date, 'forecast' AS source, value AS wind_speed_ms
# MAGIC   FROM dna_prod_silver.meteomatics.wind_speed_forecast
# MAGIC   WHERE curve_name = 'wind_speed_100m_ms_ecmwf_ens_p1d' AND model = 'ecmwf-ens'
# MAGIC     AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.wind_speed_forecast WHERE curve_name='wind_speed_100m_ms_ecmwf_ens_p1d' AND model='ecmwf-ens')
# MAGIC     AND latitude BETWEEN 30 AND 38 AND longitude BETWEEN -7 AND 33
# MAGIC )
# MAGIC SELECT * FROM wind_raw WHERE city IS NOT NULL
# MAGIC ORDER BY city, delivery_date;
# MAGIC
# MAGIC -- India Precip Anomaly
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.india_precip_anomaly AS
# MAGIC WITH actuals AS (
# MAGIC   SELECT CASE
# MAGIC     WHEN latitude >= 23.0 AND latitude < 35.0 AND longitude >= 68.0 AND longitude < 78.0 THEN 'Northwest'
# MAGIC     WHEN latitude >= 23.0 AND latitude < 35.0 AND longitude >= 78.0 AND longitude < 90.0 THEN 'Northeast'
# MAGIC     WHEN latitude >= 18.0 AND latitude < 23.0 AND longitude >= 72.0 AND longitude < 85.0 THEN 'Central'
# MAGIC     WHEN latitude >= 8.0  AND latitude < 18.0 AND longitude >= 73.0 AND longitude < 85.0 THEN 'South'
# MAGIC     WHEN latitude >= 10.0 AND latitude < 23.0 AND longitude >= 68.0 AND longitude < 74.0 THEN 'West Coast'
# MAGIC   END AS subregion, DATE(delivery_start) AS delivery_date, AVG(value) AS precip_actual_mm
# MAGIC   FROM dna_prod_silver.meteomatics.precipitation
# MAGIC   WHERE curve_name = 'precip_24h_mm_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND delivery_start >= CURRENT_DATE() - INTERVAL 90 DAYS
# MAGIC     AND latitude BETWEEN 8 AND 35 AND longitude BETWEEN 68 AND 90
# MAGIC   GROUP BY subregion, DATE(delivery_start) HAVING subregion IS NOT NULL
# MAGIC ),
# MAGIC with_avg AS (
# MAGIC   SELECT *, AVG(precip_actual_mm) OVER (PARTITION BY subregion ORDER BY delivery_date ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS precip_clim_mm
# MAGIC   FROM actuals
# MAGIC )
# MAGIC SELECT subregion, delivery_date, precip_actual_mm,
# MAGIC   COALESCE(precip_clim_mm, precip_actual_mm) AS precip_clim_mm,
# MAGIC   CAST(NULL AS DOUBLE) AS precip_forecast_mm,
# MAGIC   CASE WHEN COALESCE(precip_clim_mm, 0) > 0.1 THEN ROUND(100.0*(precip_actual_mm - precip_clim_mm)/precip_clim_mm, 1) ELSE 0.0 END AS deviation_pct
# MAGIC FROM with_avg ORDER BY subregion, delivery_date;
# MAGIC
# MAGIC -- Gatun Precip History
# MAGIC CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.gatun_precip_history AS
# MAGIC WITH daily AS (
# MAGIC   SELECT YEAR(delivery_start) AS year, DAYOFYEAR(delivery_start) AS day_of_year, AVG(value) AS daily_precip_mm
# MAGIC   FROM dna_prod_silver.meteomatics.precipitation
# MAGIC   WHERE curve_name = 'precip_24h_mm_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
# MAGIC     AND latitude BETWEEN 8.5 AND 10.0 AND longitude BETWEEN -80.5 AND -79.0
# MAGIC   GROUP BY YEAR(delivery_start), DAYOFYEAR(delivery_start)
# MAGIC )
# MAGIC SELECT year, day_of_year, SUM(daily_precip_mm) OVER (PARTITION BY year ORDER BY day_of_year) AS cumulative_precip_mm
# MAGIC FROM daily ORDER BY year, day_of_year;

# COMMAND ----------

# DBTITLE 1,5. Refresh Complete
from datetime import datetime
print(f"\n{'='*60}")
print(f"LPG Desk Sandbox Refresh COMPLETE: {datetime.now()}")
print(f"{'='*60}")

# Verify row counts
tables = ["hdd_season", "hdd_normal", "hdd_run_comparison", "asia_hdd_season",
          "padd3_coastal_temps", "nafrica_wind", "india_precip_anomaly",
          "gatun_precip_history"]
for t in tables:
    cnt = spark.sql(f"SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.{t}").collect()[0][0]
    print(f"  {t}: {cnt} rows")
