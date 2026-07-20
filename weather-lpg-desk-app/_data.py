"""Data access layer — LPG Desk Weather Dashboard.

All queries hit dna_snbx_weather.lpg_desk.* (pre-aggregated sandbox tables
populated by the refresh notebook). App SP has SELECT on the schema.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from _config import (
    SBX_SCHEMA, HDD_BASE_TEMP, SEASON_START_MONTH, SEASON_START_DAY,
    HIST_START_YEAR, HIST_END_YEAR,
    ASIA_CITY_LOCATIONS, ASIA_POPULATION, ASIA_REGION_MAP,
    PADD_REGIONS, PADD3_COASTAL_CITIES, NAFRICA_WIND_CORRIDOR,
    GATUN_N_ANALOGS,
)

# ─── Connection config ────────────────────────────────────────────────────────────
_raw_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_HOST = _raw_host if _raw_host.startswith("https://") else f"https://{_raw_host}"
WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "").split("/")[-1]


def _headers() -> dict:
    """Auth headers using app SP M2M credentials."""
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    h = {"Content-Type": "application/json"}
    tok = w.config.authenticate()
    if isinstance(tok, dict):
        h.update(tok)
    elif isinstance(tok, str):
        h["Authorization"] = f"Bearer {tok}"
    return h


def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL via Statement API and return a DataFrame."""
    resp = requests.post(
        f"{DATABRICKS_HOST}/api/2.0/sql/statements/",
        headers=_headers(),
        json={"warehouse_id": WAREHOUSE_ID, "statement": sql, "wait_timeout": "50s"},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    if data.get("status", {}).get("state") == "FAILED":
        raise RuntimeError(data["status"]["error"]["message"])
    cols = [c["name"] for c in data.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = data.get("result", {}).get("data_array", [])
    return pd.DataFrame(rows, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: US HDD BY PADD
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_hdd_season() -> pd.DataFrame:
    """Season HDD: actuals + forecast cumulative, by PADD."""
    sql = f"""
    SELECT model_label, padd_region, delivery_date, daily_hdd, cumulative_hdd
    FROM {SBX_SCHEMA}.hdd_season
    ORDER BY padd_region, model_label, delivery_date
    """
    df = run_query(sql)
    df["daily_hdd"] = pd.to_numeric(df["daily_hdd"], errors="coerce")
    df["cumulative_hdd"] = pd.to_numeric(df["cumulative_hdd"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_hdd_normal_daily() -> pd.DataFrame:
    """ERA5 climatological daily HDD normal by PADD (day_of_year)."""
    sql = f"""
    SELECT padd_region, day_of_year, daily_hdd_normal
    FROM {SBX_SCHEMA}.hdd_normal
    ORDER BY padd_region, day_of_year
    """
    df = run_query(sql)
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["daily_hdd_normal"] = pd.to_numeric(df["daily_hdd_normal"], errors="coerce")
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_hdd_run_comparison() -> pd.DataFrame:
    """Daily HDD forecast — latest 3 runs, by PADD."""
    sql = f"""
    SELECT model_label, run_ts, run_rank, run_label, padd_region, delivery_date, daily_hdd
    FROM {SBX_SCHEMA}.hdd_run_comparison
    WHERE snapshot_ts = (SELECT MAX(snapshot_ts) FROM {SBX_SCHEMA}.hdd_run_comparison)
    ORDER BY model_label, run_rank, padd_region, delivery_date
    """
    df = run_query(sql)
    df["daily_hdd"] = pd.to_numeric(df["daily_hdd"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    df["run_ts"] = pd.to_datetime(df["run_ts"], errors="coerce", utc=True)
    df["run_rank"] = pd.to_numeric(df["run_rank"], errors="coerce").astype("Int64")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASIA HDD
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_asia_hdd_season() -> pd.DataFrame:
    """Season HDD for Asia regions (Japan, Korea, China N/E/S)."""
    sql = f"""
    SELECT region, model_label, delivery_date, daily_hdd, cumulative_hdd
    FROM {SBX_SCHEMA}.asia_hdd_season
    ORDER BY region, model_label, delivery_date
    """
    df = run_query(sql)
    df["daily_hdd"] = pd.to_numeric(df["daily_hdd"], errors="coerce")
    df["cumulative_hdd"] = pd.to_numeric(df["cumulative_hdd"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_asia_hdd_normal() -> pd.DataFrame:
    """ERA5 climatological daily HDD normal for Asia regions."""
    sql = f"""
    SELECT region, day_of_year, daily_hdd_normal
    FROM {SBX_SCHEMA}.asia_hdd_normal
    ORDER BY region, day_of_year
    """
    df = run_query(sql)
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["daily_hdd_normal"] = pd.to_numeric(df["daily_hdd_normal"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: INDIA PRECIPITATION ANOMALY
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_india_precip_anomaly() -> pd.DataFrame:
    """India precipitation: actual vs climatology, deviation %."""
    sql = f"""
    SELECT subregion, delivery_date, precip_actual_mm, precip_clim_mm,
           precip_forecast_mm, deviation_pct
    FROM {SBX_SCHEMA}.india_precip_anomaly
    ORDER BY subregion, delivery_date
    """
    df = run_query(sql)
    for c in ["precip_actual_mm", "precip_clim_mm", "precip_forecast_mm", "deviation_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_india_precip_map() -> pd.DataFrame:
    """Gridded India precipitation anomaly for map display."""
    sql = f"""
    SELECT latitude, longitude, deviation_pct, period_label
    FROM {SBX_SCHEMA}.india_precip_map
    ORDER BY latitude, longitude
    """
    df = run_query(sql)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["deviation_pct"] = pd.to_numeric(df["deviation_pct"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GATUN LAKE SEASONAL FORECAST (analog years)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_gatun_precip_history() -> pd.DataFrame:
    """Historical cumulative precipitation over Gatun Lake catchment, by year."""
    sql = f"""
    SELECT year, day_of_year, cumulative_precip_mm
    FROM {SBX_SCHEMA}.gatun_precip_history
    ORDER BY year, day_of_year
    """
    df = run_query(sql)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["cumulative_precip_mm"] = pd.to_numeric(df["cumulative_precip_mm"], errors="coerce")
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_gatun_analog_forecast() -> pd.DataFrame:
    """Analog year forecasts for Gatun Lake precipitation (top N similar years)."""
    sql = f"""
    SELECT analog_year, similarity_score, day_of_year, cumulative_precip_mm
    FROM {SBX_SCHEMA}.gatun_analog_forecast
    ORDER BY similarity_score DESC, day_of_year
    """
    df = run_query(sql)
    df["analog_year"] = pd.to_numeric(df["analog_year"], errors="coerce").astype("Int64")
    df["similarity_score"] = pd.to_numeric(df["similarity_score"], errors="coerce")
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["cumulative_precip_mm"] = pd.to_numeric(df["cumulative_precip_mm"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PADD 3 COASTAL TEMPERATURES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_padd3_coastal_temps() -> pd.DataFrame:
    """Temperature actuals + forecast for PADD 3 coastal terminals."""
    sql = f"""
    SELECT city, delivery_date, source, temperature_c
    FROM {SBX_SCHEMA}.padd3_coastal_temps
    ORDER BY city, delivery_date
    """
    df = run_query(sql)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_padd3_coastal_climatology() -> pd.DataFrame:
    """Climatological temperature for PADD 3 coastal terminals."""
    sql = f"""
    SELECT city, day_of_year, temp_normal_c
    FROM {SBX_SCHEMA}.padd3_coastal_climatology
    ORDER BY city, day_of_year
    """
    df = run_query(sql)
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["temp_normal_c"] = pd.to_numeric(df["temp_normal_c"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: HURRICANES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_hurricane_activity() -> pd.DataFrame:
    """Current tropical cyclone activity (tracks + wind intensity)."""
    sql = f"""
    SELECT basin, storm_name, timestamp, latitude, longitude,
           max_wind_kt, category, status
    FROM {SBX_SCHEMA}.hurricane_activity
    ORDER BY basin, storm_name, timestamp
    """
    df = run_query(sql)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["max_wind_kt"] = pd.to_numeric(df["max_wind_kt"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_hurricane_season_stats() -> pd.DataFrame:
    """Season-to-date ACE and storm count by basin."""
    sql = f"""
    SELECT basin, season_year, storms_total, hurricanes_total,
           ace_index, ace_normal, ace_anomaly_pct
    FROM {SBX_SCHEMA}.hurricane_season_stats
    ORDER BY basin, season_year
    """
    df = run_query(sql)
    for c in ["storms_total", "hurricanes_total", "ace_index", "ace_normal", "ace_anomaly_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype("Int64")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: NORTH AFRICA WIND SPEEDS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_nafrica_wind() -> pd.DataFrame:
    """Wind speeds (actual + forecast) at North Africa corridor points."""
    sql = f"""
    SELECT city, delivery_date, source, wind_speed_ms
    FROM {SBX_SCHEMA}.nafrica_wind
    ORDER BY city, delivery_date
    """
    df = run_query(sql)
    df["wind_speed_ms"] = pd.to_numeric(df["wind_speed_ms"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_nafrica_wind_climatology() -> pd.DataFrame:
    """Climatological wind speed for North Africa corridor."""
    sql = f"""
    SELECT city, day_of_year, wind_speed_normal_ms
    FROM {SBX_SCHEMA}.nafrica_wind_climatology
    ORDER BY city, day_of_year
    """
    df = run_query(sql)
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["wind_speed_normal_ms"] = pd.to_numeric(df["wind_speed_normal_ms"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_cumulative_hdd(df: pd.DataFrame, date_col: str = "delivery_date",
                          hdd_col: str = "daily_hdd") -> pd.DataFrame:
    """Add cumulative_hdd column within each season (Oct 1 – Mar 31)."""
    df = df.sort_values(date_col).copy()
    df["cumulative_hdd"] = df[hdd_col].cumsum()
    return df
