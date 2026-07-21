# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,LPG Desk Sandbox Refresh
"""
LPG Desk Sandbox Refresh
=========================
Populates dna_snbx_weather.lpg_desk.* tables from the production silver layer.
Schedule: every 6 hours via Lakeflow Jobs.

Replaces lpg_desk_sandbox_refresh.py and lpg_desk_refresh_v2.py (both retired —
the first used the wrong column names (curve/date instead of curve_name/
delivery_start) and never ran successfully; the second fixed the columns but
left hdd_run_comparison writing into a table that's never created, never
populated the Asia forecast leg, and (with the v1 script) never built a table
for Gatun Lake or hurricanes. This version:
  - uses the correct source columns everywhere: curve_name, delivery_start
  - builds every table the Streamlit app's _data.py actually queries, and no
    unused ones (india_precip_map, *_climatology tables, gatun_*, hurricane_*
    were queried by nothing or nothing populated them — dropped; Gatun Lake
    and hurricane data are now fetched live in the app straight from the
    Panama Canal Authority / NHC / IBTrACS, matching the Coal Desk dashboard's
    approach, so they need no sandbox table at all)
  - actually population-weights Asia HDD (previously a flat unweighted city
    average despite ASIA_POPULATION existing in _config.py)
  - adds the Asia forecast leg and Asia climatology normal (previously only
    actuals existed, so the "14-day forecast" bar chart always read zero)
  - computes India's climatology as a true day-of-year normal (2000-2024),
    not a trailing 30-day rolling average, and populates the forecast leg
    that was previously hardcoded NULL

Sandbox tables (must stay in sync with the lookup tables below, which mirror
weather-lpg-desk-app/_config.py — update both if PADD/city/region boundaries
change):
  - hdd_season, hdd_normal, hdd_run_comparison   (US HDD by PADD)
  - asia_hdd_season, asia_hdd_normal             (Asia HDD)
  - india_precip_anomaly                         (India precip vs climatology)
  - padd3_coastal_temps                          (PADD 3 export terminals)
  - nafrica_wind                                 (N.Africa shipping corridor)
"""
from datetime import datetime

print(f"Refresh started: {datetime.now()}")
spark.sql("CREATE SCHEMA IF NOT EXISTS dna_snbx_weather.lpg_desk")

# COMMAND ----------

# DBTITLE 1,Lookup tables (mirror _config.py)
# Keep these in sync with weather-lpg-desk-app/_config.py — they define the
# same regions/cities so the app's labels line up with what this job computes.

PADD_REGIONS = {
    "PADD 1 - Northeast":     {"lat_min": 36.5, "lat_max": 47.5, "lon_min": -80.0, "lon_max": -66.5},
    "PADD 2 - Midwest":       {"lat_min": 36.5, "lat_max": 49.0, "lon_min": -97.5, "lon_max": -80.0},
    "PADD 3 - South Central": {"lat_min": 25.5, "lat_max": 36.5, "lon_min": -106.0, "lon_max": -79.5},
    "PADD 4 - Mountain":      {"lat_min": 36.5, "lat_max": 49.0, "lon_min": -116.5, "lon_max": -97.5},
    "PADD 5 - West":          {"lat_min": 32.0, "lat_max": 49.0, "lon_min": -125.0, "lon_max": -116.5},
}

ASIA_CITY_LOCATIONS = {
    'Tokyo': (35.5, 139.5), 'Osaka': (34.5, 135.5), 'Niigata': (38.0, 139.0),
    'Sendai': (38.5, 141.0), 'Sapporo': (43.0, 141.5), 'Fukuoka': (33.5, 130.5), 'Nagoya': (35.0, 137.0),
    'Seoul': (37.5, 127.0), 'Busan': (35.0, 129.0), 'Incheon': (37.5, 126.5), 'Daegu': (36.0, 128.5),
    'Beijing': (40.0, 116.5), 'Tianjin': (39.5, 117.5), 'Harbin': (45.5, 126.5), 'Shenyang': (42.0, 123.5),
    'Changchun': (44.0, 125.5), 'Hohhot': (41.0, 111.5), 'Urumqi': (44.0, 87.5),
    'Shijiazhuang': (38.0, 114.5), 'Jinan': (36.5, 117.0), 'Xian': (34.5, 109.0),
    'Shanghai': (31.0, 121.5), 'Nanjing': (32.0, 119.0), 'Hangzhou': (30.5, 120.0),
    'Wuhan': (30.5, 114.5), 'Hefei': (32.0, 117.0), 'Zhengzhou': (35.0, 113.5),
    'Guangzhou': (23.0, 113.5), 'Shenzhen': (22.5, 114.0), 'Kunming': (25.0, 102.5),
    'Chengdu': (30.5, 104.0), 'Chongqing': (29.5, 107.0), 'Nanning': (23.0, 108.5),
}
ASIA_POPULATION = {
    'Tokyo': 14000000, 'Osaka': 8800000, 'Niigata': 800000, 'Sendai': 1100000,
    'Sapporo': 2000000, 'Fukuoka': 1600000, 'Nagoya': 2300000,
    'Seoul': 9700000, 'Busan': 3400000, 'Incheon': 3000000, 'Daegu': 2400000,
    'Beijing': 21830000, 'Tianjin': 13640000, 'Harbin': 10000000, 'Shenyang': 9000000,
    'Changchun': 7700000, 'Hohhot': 3500000, 'Urumqi': 4000000, 'Shijiazhuang': 11000000,
    'Jinan': 9200000, 'Xian': 13000000,
    'Shanghai': 24800000, 'Nanjing': 9400000, 'Hangzhou': 12200000, 'Wuhan': 12300000,
    'Hefei': 9400000, 'Zhengzhou': 12700000,
    'Guangzhou': 18700000, 'Shenzhen': 17600000, 'Kunming': 8500000,
    'Chengdu': 21000000, 'Chongqing': 32000000, 'Nanning': 8800000,
}
ASIA_REGION_MAP = {
    'Japan': ['Tokyo', 'Osaka', 'Niigata', 'Sendai', 'Sapporo', 'Fukuoka', 'Nagoya'],
    'South Korea': ['Seoul', 'Busan', 'Incheon', 'Daegu'],
    'China North': ['Beijing', 'Tianjin', 'Harbin', 'Shenyang', 'Changchun',
                    'Hohhot', 'Urumqi', 'Shijiazhuang', 'Jinan', 'Xian'],
    'China East': ['Shanghai', 'Nanjing', 'Hangzhou', 'Wuhan', 'Hefei', 'Zhengzhou'],
    'China South': ['Guangzhou', 'Shenzhen', 'Kunming', 'Chengdu', 'Chongqing', 'Nanning'],
}
ASIA_CITY_TO_REGION = {c: r for r, cities in ASIA_REGION_MAP.items() for c in cities}

INDIA_SUBREGIONS = {
    "Northwest": {"lat_min": 23.0, "lat_max": 35.0, "lon_min": 68.0, "lon_max": 78.0},
    "Northeast": {"lat_min": 23.0, "lat_max": 35.0, "lon_min": 78.0, "lon_max": 90.0},
    "Central":   {"lat_min": 18.0, "lat_max": 23.0, "lon_min": 72.0, "lon_max": 85.0},
    "South":     {"lat_min": 8.0,  "lat_max": 18.0, "lon_min": 73.0, "lon_max": 85.0},
    "West Coast": {"lat_min": 10.0, "lat_max": 23.0, "lon_min": 68.0, "lon_max": 74.0},
}

PADD3_COASTAL_CITIES = {
    'Houston': (30.0, -95.5), 'Freeport': (29.0, -95.5), 'Beaumont': (30.0, -94.0),
    'Lake Charles': (30.0, -93.0), 'New Orleans': (30.0, -90.0), 'Corpus Christi': (27.5, -97.5),
    'Port Arthur': (30.0, -94.0), 'Morgan City': (29.5, -91.0),
}

NAFRICA_WIND_CORRIDOR = {
    'Tangier': (35.5, -6.0), 'Oran': (35.5, -0.5), 'Algiers': (36.5, 3.0), 'Bejaia': (36.5, 5.0),
    'Skikda': (37.0, 7.0), 'Tunis': (37.0, 10.0), 'Tripoli': (33.0, 13.0), 'Benghazi': (32.0, 20.0),
    'Alexandria': (31.0, 30.0), 'Port Said': (31.5, 32.5),
}


def padd_case_sql(lat="latitude", lon="longitude"):
    lines = ["CASE"]
    for name, b in PADD_REGIONS.items():
        lines.append(
            f"  WHEN {lat} >= {b['lat_min']} AND {lat} < {b['lat_max']} "
            f"AND {lon} >= {b['lon_min']} AND {lon} < {b['lon_max']} THEN '{name}'"
        )
    lines.append("END")
    return "\n".join(lines)


def coord_filter_sql(coords, lat="latitude", lon="longitude"):
    parts = [f"({lat} = {la} AND {lon} = {lo})" for la, lo in coords]
    return " OR ".join(parts)


def city_case_sql(city_locations, lat="latitude", lon="longitude"):
    lines = ["CASE"]
    for city, (la, lo) in city_locations.items():
        lines.append(f"  WHEN {lat} = {la} AND {lon} = {lo} THEN '{city}'")
    lines.append("END")
    return "\n".join(lines)


def india_case_sql(lat="latitude", lon="longitude"):
    lines = ["CASE"]
    for name, b in INDIA_SUBREGIONS.items():
        lines.append(
            f"  WHEN {lat} >= {b['lat_min']} AND {lat} < {b['lat_max']} "
            f"AND {lon} >= {b['lon_min']} AND {lon} < {b['lon_max']} THEN '{name}'"
        )
    lines.append("END")
    return "\n".join(lines)


PADD_CASE = padd_case_sql()
INDIA_CASE = india_case_sql()

# COMMAND ----------

# DBTITLE 1,1. US HDD by PADD — season + normal
spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.hdd_season AS
WITH padd_assign AS (
  SELECT DATE(delivery_start) AS dt, value AS temperature, {PADD_CASE} AS padd_region
  FROM dna_prod_silver.meteomatics.temperature
  WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
    AND delivery_start >= DATE_FORMAT(DATEADD(YEAR, -1, CURRENT_DATE()), 'yyyy-10-01')
    AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -65
),
actuals AS (
  SELECT 'Actual (ERA5)' AS model_label, padd_region, dt AS delivery_date,
    GREATEST(0, 18.0 - AVG(temperature)) AS daily_hdd
  FROM padd_assign WHERE padd_region IS NOT NULL GROUP BY padd_region, dt
),
fcst_assign AS (
  SELECT DATE(delivery_start) AS dt, value AS temperature, {PADD_CASE} AS padd_region
  FROM dna_prod_silver.meteomatics.temperature_forecast
  WHERE curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d' AND model = 'ecmwf-ens'
    AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.temperature_forecast WHERE model = 'ecmwf-ens')
    AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -65
),
forecasts AS (
  SELECT 'EC-ens (14d)' AS model_label, padd_region, dt AS delivery_date,
    GREATEST(0, 18.0 - AVG(temperature)) AS daily_hdd
  FROM fcst_assign WHERE padd_region IS NOT NULL GROUP BY padd_region, dt
),
combined AS (SELECT * FROM actuals UNION ALL SELECT * FROM forecasts)
SELECT model_label, padd_region, delivery_date, daily_hdd,
  SUM(daily_hdd) OVER (PARTITION BY model_label, padd_region ORDER BY delivery_date) AS cumulative_hdd
FROM combined ORDER BY padd_region, model_label, delivery_date
""")
print(f"hdd_season: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.hdd_season').first()[0]} rows")

spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.hdd_normal AS
WITH padd_daily AS (
  SELECT DAYOFYEAR(delivery_start) AS day_of_year, DATE(delivery_start) AS dt,
    {PADD_CASE} AS padd_region, value AS temperature
  FROM dna_prod_silver.meteomatics.temperature
  WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
    AND YEAR(delivery_start) BETWEEN 2000 AND 2024
    AND (DAYOFYEAR(delivery_start) >= 274 OR DAYOFYEAR(delivery_start) <= 90)
    AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -65
),
daily_agg AS (
  SELECT padd_region, day_of_year, dt, GREATEST(0, 18.0 - AVG(temperature)) AS daily_hdd
  FROM padd_daily WHERE padd_region IS NOT NULL GROUP BY padd_region, day_of_year, dt
)
SELECT padd_region, day_of_year, AVG(daily_hdd) AS daily_hdd_normal
FROM daily_agg GROUP BY padd_region, day_of_year ORDER BY padd_region, day_of_year
""")
print(f"hdd_normal: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.hdd_normal').first()[0]} rows")

# COMMAND ----------

# DBTITLE 1,2. US HDD run comparison (latest 3 model runs)
# Fully replaced each run (CREATE OR REPLACE, not incremental insertInto —
# the previous version wrote into a table it never created).
spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.hdd_run_comparison AS
WITH ranked_runs AS (
  SELECT DISTINCT created_at,
    DENSE_RANK() OVER (ORDER BY created_at DESC) AS run_rank
  FROM dna_prod_silver.meteomatics.temperature_forecast
  WHERE model = 'ecmwf-ens' AND curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d'
),
top3 AS (SELECT created_at, run_rank FROM ranked_runs WHERE run_rank <= 3),
daily AS (
  SELECT t.created_at AS run_ts, r.run_rank, {PADD_CASE} AS padd_region,
    DATE(t.delivery_start) AS delivery_date, GREATEST(0, 18.0 - AVG(t.value)) AS daily_hdd
  FROM dna_prod_silver.meteomatics.temperature_forecast t
  JOIN top3 r ON t.created_at = r.created_at
  WHERE t.model = 'ecmwf-ens' AND t.curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d'
    AND t.latitude BETWEEN 24 AND 50 AND t.longitude BETWEEN -125 AND -65
  GROUP BY t.created_at, r.run_rank, {PADD_CASE}, DATE(t.delivery_start)
)
SELECT 'EC-ens' AS model_label, run_ts, run_rank,
  CASE run_rank WHEN 1 THEN 'Latest' WHEN 2 THEN '-12h' ELSE '-24h' END AS run_label,
  padd_region, delivery_date, daily_hdd, CURRENT_TIMESTAMP() AS snapshot_ts
FROM daily WHERE padd_region IS NOT NULL
ORDER BY run_rank, padd_region, delivery_date
""")
print(f"hdd_run_comparison: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.hdd_run_comparison').first()[0]} rows")

# COMMAND ----------

# DBTITLE 1,3. Asia HDD — season + normal (population-weighted)
asia_weights_values = ",\n  ".join(
    f"('{city}', '{ASIA_CITY_TO_REGION[city]}', {ASIA_POPULATION[city]}.0)"
    for city in ASIA_CITY_LOCATIONS
)
ASIA_CITY_CASE = city_case_sql(ASIA_CITY_LOCATIONS)
ASIA_COORD_FILTER = coord_filter_sql(ASIA_CITY_LOCATIONS.values())

spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.asia_hdd_season AS
WITH weights AS (
  SELECT * FROM VALUES
  {asia_weights_values}
  AS t(city, region, population)
),
city_actuals AS (
  SELECT {ASIA_CITY_CASE} AS city, DATE(delivery_start) AS dt, AVG(value) AS temperature
  FROM dna_prod_silver.meteomatics.temperature
  WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
    AND delivery_start >= DATE_FORMAT(DATEADD(YEAR, -1, CURRENT_DATE()), 'yyyy-10-01')
    AND ({ASIA_COORD_FILTER})
  GROUP BY {ASIA_CITY_CASE}, DATE(delivery_start)
),
city_forecast AS (
  SELECT {ASIA_CITY_CASE} AS city, DATE(delivery_start) AS dt, AVG(value) AS temperature
  FROM dna_prod_silver.meteomatics.temperature_forecast
  WHERE curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d' AND model = 'ecmwf-ens'
    AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.temperature_forecast WHERE model = 'ecmwf-ens')
    AND ({ASIA_COORD_FILTER})
  GROUP BY {ASIA_CITY_CASE}, DATE(delivery_start)
),
weighted AS (
  SELECT 'Actual (ERA5)' AS model_label, w.region, a.dt AS delivery_date,
    GREATEST(0, 18.0 - a.temperature) * w.population AS weighted_hdd, w.population
  FROM city_actuals a JOIN weights w ON a.city = w.city
  UNION ALL
  SELECT 'EC-ens (14d)' AS model_label, w.region, f.dt AS delivery_date,
    GREATEST(0, 18.0 - f.temperature) * w.population AS weighted_hdd, w.population
  FROM city_forecast f JOIN weights w ON f.city = w.city
),
region_daily AS (
  SELECT model_label, region, delivery_date,
    SUM(weighted_hdd) / SUM(population) AS daily_hdd
  FROM weighted GROUP BY model_label, region, delivery_date
)
SELECT region, model_label, delivery_date, daily_hdd,
  SUM(daily_hdd) OVER (PARTITION BY region, model_label ORDER BY delivery_date) AS cumulative_hdd
FROM region_daily ORDER BY region, model_label, delivery_date
""")
print(f"asia_hdd_season: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.asia_hdd_season').first()[0]} rows")

spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.asia_hdd_normal AS
WITH weights AS (
  SELECT * FROM VALUES
  {asia_weights_values}
  AS t(city, region, population)
),
city_daily AS (
  SELECT {ASIA_CITY_CASE} AS city, DAYOFYEAR(delivery_start) AS day_of_year,
    DATE(delivery_start) AS dt, AVG(value) AS temperature
  FROM dna_prod_silver.meteomatics.temperature
  WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
    AND YEAR(delivery_start) BETWEEN 2000 AND 2024
    AND ({ASIA_COORD_FILTER})
  GROUP BY {ASIA_CITY_CASE}, DAYOFYEAR(delivery_start), DATE(delivery_start)
),
weighted AS (
  SELECT w.region, c.day_of_year, c.dt,
    GREATEST(0, 18.0 - c.temperature) * w.population AS weighted_hdd, w.population
  FROM city_daily c JOIN weights w ON c.city = w.city
),
daily_region AS (
  SELECT region, day_of_year, dt, SUM(weighted_hdd) / SUM(population) AS daily_hdd_normal
  FROM weighted GROUP BY region, day_of_year, dt
)
SELECT region, day_of_year, AVG(daily_hdd_normal) AS daily_hdd_normal
FROM daily_region GROUP BY region, day_of_year ORDER BY region, day_of_year
""")
print(f"asia_hdd_normal: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.asia_hdd_normal').first()[0]} rows")

# COMMAND ----------

# DBTITLE 1,4. India precipitation anomaly (day-of-year climatology + forecast)
spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.india_precip_anomaly AS
WITH actuals AS (
  SELECT {INDIA_CASE} AS subregion, DATE(delivery_start) AS delivery_date, AVG(value) AS precip_actual_mm
  FROM dna_prod_silver.meteomatics.precipitation
  WHERE curve_name = 'precip_24h_mm_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
    AND delivery_start >= CURRENT_DATE() - INTERVAL 120 DAYS
    AND latitude BETWEEN 8 AND 35 AND longitude BETWEEN 68 AND 90
  GROUP BY {INDIA_CASE}, DATE(delivery_start)
  HAVING subregion IS NOT NULL
),
forecasts AS (
  SELECT {INDIA_CASE} AS subregion, DATE(delivery_start) AS delivery_date, AVG(value) AS precip_forecast_mm
  FROM dna_prod_silver.meteomatics.precipitation_forecast
  WHERE curve_name = 'precip_24h_mm_ecmwf_ens_p1d' AND model = 'ecmwf-ens'
    AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.precipitation_forecast WHERE model = 'ecmwf-ens')
    AND latitude BETWEEN 8 AND 35 AND longitude BETWEEN 68 AND 90
  GROUP BY {INDIA_CASE}, DATE(delivery_start)
  HAVING subregion IS NOT NULL
),
clim_daily AS (
  SELECT {INDIA_CASE} AS subregion, DAYOFYEAR(delivery_start) AS doy, DATE(delivery_start) AS dt, AVG(value) AS precip_mm
  FROM dna_prod_silver.meteomatics.precipitation
  WHERE curve_name = 'precip_24h_mm_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
    AND YEAR(delivery_start) BETWEEN 2000 AND 2024
    AND latitude BETWEEN 8 AND 35 AND longitude BETWEEN 68 AND 90
  GROUP BY {INDIA_CASE}, DAYOFYEAR(delivery_start), DATE(delivery_start)
  HAVING subregion IS NOT NULL
),
clim AS (
  SELECT subregion, doy, AVG(precip_mm) AS precip_clim_mm
  FROM clim_daily GROUP BY subregion, doy
)
SELECT a.subregion, a.delivery_date, a.precip_actual_mm, c.precip_clim_mm, NULL AS precip_forecast_mm,
  CASE WHEN c.precip_clim_mm > 0.1
    THEN ROUND(100.0 * (a.precip_actual_mm - c.precip_clim_mm) / c.precip_clim_mm, 1)
    ELSE 0.0 END AS deviation_pct
FROM actuals a JOIN clim c ON a.subregion = c.subregion AND DAYOFYEAR(a.delivery_date) = c.doy
UNION ALL
SELECT f.subregion, f.delivery_date, NULL AS precip_actual_mm, c.precip_clim_mm, f.precip_forecast_mm,
  CASE WHEN c.precip_clim_mm > 0.1
    THEN ROUND(100.0 * (f.precip_forecast_mm - c.precip_clim_mm) / c.precip_clim_mm, 1)
    ELSE 0.0 END AS deviation_pct
FROM forecasts f JOIN clim c ON f.subregion = c.subregion AND DAYOFYEAR(f.delivery_date) = c.doy
ORDER BY subregion, delivery_date
""")
print(f"india_precip_anomaly: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.india_precip_anomaly').first()[0]} rows")

# COMMAND ----------

# DBTITLE 1,5. PADD 3 coastal temperatures (export terminals)
PADD3_CITY_CASE = city_case_sql(PADD3_COASTAL_CITIES)
PADD3_COORD_FILTER = coord_filter_sql(PADD3_COASTAL_CITIES.values())

spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.padd3_coastal_temps AS
WITH actuals AS (
  SELECT {PADD3_CITY_CASE} AS city, DATE(delivery_start) AS delivery_date, 'actual' AS source, value AS temperature_c
  FROM dna_prod_silver.meteomatics.temperature
  WHERE curve_name = 't_mean_2m_24h_c_ecmwf_era5_p1d' AND model = 'ecmwf-era5'
    AND delivery_start >= CURRENT_DATE() - INTERVAL 60 DAYS
    AND ({PADD3_COORD_FILTER})
),
forecasts AS (
  -- ENS stores 50 ensemble members per grid point per day; AVG gives the ensemble mean.
  SELECT {PADD3_CITY_CASE} AS city, DATE(delivery_start) AS delivery_date, 'forecast' AS source,
         AVG(value) AS temperature_c
  FROM dna_prod_silver.meteomatics.temperature_forecast
  WHERE curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d' AND model = 'ecmwf-ens'
    AND created_at = (SELECT MAX(created_at) FROM dna_prod_silver.meteomatics.temperature_forecast
                      WHERE model = 'ecmwf-ens' AND curve_name = 't_mean_2m_24h_c_ecmwf_ens_p1d')
    AND ({PADD3_COORD_FILTER})
  GROUP BY {PADD3_CITY_CASE}, DATE(delivery_start)
)
SELECT * FROM actuals WHERE city IS NOT NULL
UNION ALL SELECT * FROM forecasts WHERE city IS NOT NULL
ORDER BY city, delivery_date
""")
print(f"padd3_coastal_temps: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.padd3_coastal_temps').first()[0]} rows")

# COMMAND ----------

# DBTITLE 1,6. North Africa wind speeds (shipping corridor)
NAFRICA_CITY_CASE = city_case_sql(NAFRICA_WIND_CORRIDOR)
NAFRICA_COORD_FILTER = coord_filter_sql(NAFRICA_WIND_CORRIDOR.values())

# ENS stores 50 members per grid point per day — compute percentiles and risk %.
# Schema: wind_p10/p50/p90 (ensemble spread), pct_above_warning/gale (% members above 15/20 m/s).
# Source: 'actual' = past delivery dates from latest ENS run; 'forecast' = future dates.
spark.sql(f"""
CREATE OR REPLACE TABLE dna_snbx_weather.lpg_desk.nafrica_wind AS
WITH latest_run AS (
  SELECT MAX(created_at) AS max_ts
  FROM dna_prod_silver.meteomatics.wind_speed_forecast
  WHERE model = 'ecmwf-ens' AND curve_name = 'wind_speed_100m_ms_ecmwf_ens_p1d'
),
all_days AS (
  SELECT
    {NAFRICA_CITY_CASE}                                                                 AS city,
    DATE(delivery_start)                                                                AS delivery_date,
    CASE WHEN DATE(delivery_start) <= CURRENT_DATE() THEN 'actual' ELSE 'forecast' END AS source,
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY value)                                 AS wind_p10,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value)                                 AS wind_p50,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY value)                                 AS wind_p90,
    AVG(CASE WHEN value >= 15.0 THEN 100.0 ELSE 0.0 END)                              AS pct_above_warning,
    AVG(CASE WHEN value >= 20.0 THEN 100.0 ELSE 0.0 END)                              AS pct_above_gale
  FROM dna_prod_silver.meteomatics.wind_speed_forecast
  CROSS JOIN latest_run
  WHERE curve_name = 'wind_speed_100m_ms_ecmwf_ens_p1d'
    AND model = 'ecmwf-ens'
    AND created_at = latest_run.max_ts
    AND ({NAFRICA_COORD_FILTER})
  GROUP BY
    {NAFRICA_CITY_CASE}, DATE(delivery_start),
    CASE WHEN DATE(delivery_start) <= CURRENT_DATE() THEN 'actual' ELSE 'forecast' END
)
SELECT * FROM all_days WHERE city IS NOT NULL
ORDER BY city, delivery_date
""")
print(f"nafrica_wind: {spark.sql('SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.nafrica_wind').first()[0]} rows")

# COMMAND ----------

# DBTITLE 1,7. Refresh complete

print(f"LPG Desk Sandbox Refresh COMPLETE: {datetime.now()}")
print(f"{'='*60}")

tables = ["hdd_season", "hdd_normal", "hdd_run_comparison",
          "asia_hdd_season", "asia_hdd_normal",
          "india_precip_anomaly", "padd3_coastal_temps", "nafrica_wind"]
for t in tables:
    cnt = spark.sql(f"SELECT COUNT(*) FROM dna_snbx_weather.lpg_desk.{t}").first()[0]
    print(f"  {t}: {cnt} rows")

print("\nGatun Lake and Hurricanes are fetched live in the app (ACP / NHC / "
      "IBTrACS) and need no sandbox table.")
