"""Data access layer — LPG Desk Weather Dashboard.

Two data patterns are used, matching the Coal Desk dashboard's precedent:
  - HDD / precipitation / temperature / wind tabs hit pre-aggregated sandbox
    tables (dna_snbx_weather.lpg_desk.*), populated by lpg_desk_refresh.py.
  - Gatun Lake and Hurricanes are fetched live and directly (Panama Canal
    Authority CSVs; NHC + IBTrACS), no sandbox table needed — both sources
    publish exactly what's needed with no auth and update on their own
    schedule, so landing a copy in Databricks would just add staleness.
"""
from __future__ import annotations

import io
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from _config import (
    SBX_SCHEMA,
    GATUN_HIST_URL, GATUN_PROJECTION_URL, GATUN_ANALOG_YEARS,
    GATUN_EXTRA_LAG_DAYS, GATUN_COMPOSITE_YEARS,
    GATUN_ENSO_HORIZON_DAYS, GATUN_ENSO_INDEX_BAND, GATUN_ENSO_DECAY_MONTHS,
    GATUN_ENSO_MIN_SAMPLES,
    GATUN_EL_NINO_ONI_THRESHOLD, GATUN_ANALOG_HISTORY_START,
    GATUN_FALLBACK_ANALOG_YEARS,
    CPC_ONI_URL, ONI_SEASON_CENTRE_MONTH,
    PANAMA_PRECIP_LOOKBACK_DAYS, PANAMA_PRECIP_FORECAST_DAYS,
    HURRICANE_BASINS,
    HDD_RANK_POOL_YEARS, hdd_season_day_index,
)

# ─── Connection config ────────────────────────────────────────────────────────────
_raw_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_HOST = _raw_host if _raw_host.startswith("https://") else f"https://{_raw_host}"
WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH", "").split("/")[-1]


def _headers() -> dict:
    """Auth headers using the app SP (AZURE_CLIENT_ID M2M credentials)."""
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
    """Execute SQL via the Statement API and return a DataFrame."""
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
    """Daily HDD forecast — latest 3 model runs, by PADD."""
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_hdd_season_history() -> pd.DataFrame:
    """Daily + cumulative HDD for prior completed winters, by PADD."""
    sql = f"""
    SELECT padd_region, season_start_year, season_label, day_of_year, delivery_date, daily_hdd, cumulative_hdd
    FROM {SBX_SCHEMA}.hdd_season_history
    ORDER BY padd_region, season_start_year, delivery_date
    """
    df = run_query(sql)
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").astype("Int64")
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["daily_hdd"] = pd.to_numeric(df["daily_hdd"], errors="coerce")
    df["cumulative_hdd"] = pd.to_numeric(df["cumulative_hdd"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SEASON KPIs — cumulative-to-date, vs-normal delta, coldest/warmest rank
# Shared by US HDD (padd_region) and Asia HDD (region); works identically
# in-season (partial-season progress) and off-season (last winter's final
# numbers), since both just compare "cumulative HDD at the same point in the
# season" — there's no separate off-season code path.
# ══════════════════════════════════════════════════════════════════════════════

def compute_season_kpis(season_df: pd.DataFrame, normal_df: pd.DataFrame, history_df: pd.DataFrame,
                         group_col: str, group_value: str,
                         rank_pool_years: int = HDD_RANK_POOL_YEARS) -> dict | None:
    """Cumulative-to-date, delta vs normal, and coldest/warmest rank for one PADD/region.

    Ranks against history_df's prior completed winters at the same
    days-elapsed-since-Oct-1 point (not raw calendar date), so a season that
    started tracking late or a rank pulled while off-season (full completed
    winter) both compare apples to apples. Returns None if there's no actual
    data yet for group_value (e.g. brand-new PADD with no ERA5 rows).
    """
    sdf = season_df[season_df[group_col] == group_value] if group_col in season_df.columns else season_df
    actual = sdf[sdf["model_label"].str.contains("Actual", na=False)].dropna(subset=["cumulative_hdd", "delivery_date"])
    if actual.empty:
        return None
    latest = actual.sort_values("delivery_date").iloc[-1]
    cumulative_to_date = float(latest["cumulative_hdd"])
    as_of = latest["delivery_date"]
    day_idx = hdd_season_day_index(pd.Timestamp(as_of).dayofyear)

    delta_vs_normal = None
    ndf = normal_df[normal_df[group_col] == group_value] if group_col in normal_df.columns else normal_df
    if not ndf.empty:
        ndf = ndf.copy()
        ndf["day_idx"] = ndf["day_of_year"].apply(lambda d: hdd_season_day_index(int(d)))
        ndf = ndf.sort_values("day_idx")
        ndf["cum_normal"] = ndf["daily_hdd_normal"].cumsum()
        upto = ndf[ndf["day_idx"] <= day_idx]
        if not upto.empty:
            delta_vs_normal = cumulative_to_date - float(upto["cum_normal"].iloc[-1])

    rank, pool_size = None, 0
    hdf = history_df[history_df[group_col] == group_value] if group_col in history_df.columns else history_df
    if not hdf.empty:
        hdf = hdf.copy()
        hdf["day_idx"] = hdf["day_of_year"].apply(lambda d: hdd_season_day_index(int(d)))
        years = sorted(hdf["season_start_year"].dropna().unique())[-rank_pool_years:]
        colder_or_equal = 0
        for yr in years:
            ydf = hdf[(hdf["season_start_year"] == yr) & (hdf["day_idx"] <= day_idx)]
            if ydf.empty:
                continue
            yr_cum = float(ydf.sort_values("day_idx")["cumulative_hdd"].iloc[-1])
            pool_size += 1
            if yr_cum >= cumulative_to_date:
                colder_or_equal += 1
        if pool_size:
            rank = colder_or_equal + 1

    return {
        "cumulative_to_date": cumulative_to_date,
        "delta_vs_normal": delta_vs_normal,
        "rank": rank,
        "rank_pool_size": pool_size,
        "as_of": as_of,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASIA HDD
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_asia_hdd_season() -> pd.DataFrame:
    """Population-weighted season HDD for Asia regions (actual + forecast)."""
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
    """Population-weighted ERA5 climatological daily HDD normal, by region."""
    sql = f"""
    SELECT region, day_of_year, daily_hdd_normal
    FROM {SBX_SCHEMA}.asia_hdd_normal
    ORDER BY region, day_of_year
    """
    df = run_query(sql)
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["daily_hdd_normal"] = pd.to_numeric(df["daily_hdd_normal"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_asia_hdd_season_history() -> pd.DataFrame:
    """Daily + cumulative population-weighted HDD for prior completed winters, by region."""
    sql = f"""
    SELECT region, season_start_year, season_label, day_of_year, delivery_date, daily_hdd, cumulative_hdd
    FROM {SBX_SCHEMA}.asia_hdd_season_history
    ORDER BY region, season_start_year, delivery_date
    """
    df = run_query(sql)
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce").astype("Int64")
    df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
    df["daily_hdd"] = pd.to_numeric(df["daily_hdd"], errors="coerce")
    df["cumulative_hdd"] = pd.to_numeric(df["cumulative_hdd"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_asia_hdd_run_comparison() -> pd.DataFrame:
    """Daily HDD forecast — latest 3 model runs, by Asia region (population-weighted)."""
    sql = f"""
    SELECT model_label, run_ts, run_rank, run_label, region, delivery_date, daily_hdd
    FROM {SBX_SCHEMA}.asia_hdd_run_comparison
    WHERE snapshot_ts = (SELECT MAX(snapshot_ts) FROM {SBX_SCHEMA}.asia_hdd_run_comparison)
    ORDER BY model_label, run_rank, region, delivery_date
    """
    df = run_query(sql)
    df["daily_hdd"] = pd.to_numeric(df["daily_hdd"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    df["run_ts"] = pd.to_datetime(df["run_ts"], errors="coerce", utc=True)
    df["run_rank"] = pd.to_numeric(df["run_rank"], errors="coerce").astype("Int64")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: INDIA PRECIPITATION ANOMALY
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_india_precip_anomaly() -> pd.DataFrame:
    """India precipitation: actual/forecast vs day-of-year climatology, deviation %."""
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GATUN LAKE — live ACP feed + El Nino analog years
# ══════════════════════════════════════════════════════════════════════════════

def _drop_level_spikes(df: pd.DataFrame, max_deviation_ft: float = 1.5) -> pd.DataFrame:
    """Drop isolated 1-2 day transcription spikes from ACP's raw history file.

    Gatun Lake's surface area (~430 km2) makes >1.5 ft/day physically
    implausible; a handful of such spike-and-revert days exist in the ACP
    archive (e.g. Dec 1997) and would otherwise dominate an analog-year
    overlay. Flags points that deviate from a 5-day centered rolling median
    by more than the threshold — genuine multi-day trends aren't touched
    since the median isn't swayed by one or two outlier days.
    """
    if df.empty:
        return df
    med = df['level_ft'].rolling(5, center=True, min_periods=3).median()
    keep = (df['level_ft'] - med).abs() <= max_deviation_ft
    return df[keep].reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_gatun_lake_levels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Gatun Lake observed levels + official ACP forecast.

    Source: Panama Canal Authority — evtms-rpts.pancanal.com. Returns
    (hist_df, proj_df), both with columns date (datetime), level_ft (float).
    hist_df holds the full multi-year record; proj_df holds the official
    projection published alongside it.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LPG-Desk-Dashboard/1.0)"}

    def _fetch(url):
        r = requests.get(url, headers=headers, timeout=30, verify=False)
        r.raise_for_status()
        return r.text

    hist_raw = _fetch(GATUN_HIST_URL)
    hist_df = pd.read_csv(io.StringIO(hist_raw))
    hist_df.columns = [c.strip() for c in hist_df.columns]
    date_col = next((c for c in hist_df.columns if any(k in c.upper() for k in ['DATE', 'FECHA'])), hist_df.columns[0])
    level_col = next((c for c in hist_df.columns if any(k in c.upper() for k in ['LEVEL', 'FEET', 'NIVEL'])), hist_df.columns[1])
    hist_df[date_col] = pd.to_datetime(hist_df[date_col], errors='coerce')
    hist_df[level_col] = pd.to_numeric(hist_df[level_col], errors='coerce')
    hist_df = (hist_df[[date_col, level_col]].dropna()
               .rename(columns={date_col: 'date', level_col: 'level_ft'})
               .query('level_ft > 60')
               .sort_values('date').reset_index(drop=True))
    hist_df = _drop_level_spikes(hist_df)

    proj_raw = _fetch(GATUN_PROJECTION_URL)
    proj_df = None
    for skip in (2, 1, 0):
        try:
            candidate = pd.read_csv(io.StringIO(proj_raw), skiprows=skip)
            candidate.columns = [c.strip() for c in candidate.columns]
            if len(candidate) >= 3 and len(candidate.columns) >= 2:
                proj_df = candidate
                break
        except Exception:
            continue

    if proj_df is not None:
        date_col = next((c for c in proj_df.columns if 'date' in c.lower()), proj_df.columns[0])
        level_col = next((c for c in proj_df.columns if any(k in c.lower() for k in ['level', 'water', 'gatun'])), proj_df.columns[1])
        proj_df[date_col] = pd.to_datetime(proj_df[date_col], errors='coerce')
        proj_df[level_col] = pd.to_numeric(proj_df[level_col], errors='coerce')
        proj_df = (proj_df[[date_col, level_col]].dropna()
                   .rename(columns={date_col: 'date', level_col: 'level_ft'})
                   .sort_values('date').reset_index(drop=True))
    else:
        proj_df = pd.DataFrame(columns=['date', 'level_ft'])

    return hist_df, proj_df


def _gatun_window(proj_df: pd.DataFrame, extra_lag_days: int = GATUN_EXTRA_LAG_DAYS
                  ) -> tuple[pd.Timestamp, int, pd.Timestamp]:
    """Common (window_start, window_days, forecast_end) for every analog overlay.

    The window opens one year back from today and closes `extra_lag_days` past
    the official forecast's last date, so a shifted analog year covers both the
    observed stretch already on the chart and the space the extension needs.
    """
    window_start = pd.Timestamp.today().normalize() - pd.DateOffset(days=365)
    forecast_end = proj_df['date'].max() if not proj_df.empty else pd.Timestamp.today().normalize()
    window_end = forecast_end + pd.Timedelta(days=extra_lag_days)
    return window_start, (window_end - window_start).days, forecast_end


def _gatun_analog_series(hist_df: pd.DataFrame, analog_year: int,
                         window_start: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """One analog year's observed levels, time-shifted onto today's calendar.

    `analog_year` Y is the year the event *developed* in: the slice runs from
    window_start's month/day in Y-1 forward across `window_days`, then is shifted
    so that start lands exactly on window_start. Every analog therefore shares
    one calendar x-axis with the current observed/forecast window.

    Returns columns (display_date, level_ft); empty if the archive doesn't
    reach back that far.
    """
    start_year = analog_year - 1
    try:
        analog_start = window_start.replace(year=start_year)
    except ValueError:
        # window_start is Feb 29 and start_year isn't a leap year
        analog_start = window_start.replace(year=start_year, day=28)
    analog_end = analog_start + pd.Timedelta(days=window_days)

    mask = (hist_df['date'] >= analog_start) & (hist_df['date'] <= analog_end)
    sub = hist_df.loc[mask].copy()
    if sub.empty:
        return sub
    sub['display_date'] = sub['date'] + (window_start - analog_start)
    return sub[['display_date', 'level_ft']].sort_values('display_date')


def _anchor_to_forecast_end(sub: pd.DataFrame, forecast_end: pd.Timestamp,
                            forecast_end_level: float) -> pd.Series | None:
    """Re-express an analog's post-forecast stretch as a delta off the ACP curve.

    Takes the analog's level *change* from the point matching the official
    forecast's last date onward and applies it to the official forecast's last
    level, so the extension continues the ACP line instead of jumping to the
    analog year's absolute level (which reflects a different starting storage).
    """
    if sub.empty:
        return None
    anchor_level = sub.loc[(sub['display_date'] - forecast_end).abs().idxmin(), 'level_ft']
    post = sub[sub['display_date'] >= forecast_end]
    if post.empty:
        return None
    return pd.Series((forecast_end_level + (post['level_ft'] - anchor_level)).values,
                     index=post['display_date'].values)


def _forecast_end_level(hist_df: pd.DataFrame, proj_df: pd.DataFrame) -> float:
    """Level the extension forecasts start from — the ACP forecast's last point,
    or the latest observation when no official projection is published."""
    if not proj_df.empty:
        return float(proj_df.loc[proj_df['date'].idxmax(), 'level_ft'])
    return float(hist_df.sort_values('date')['level_ft'].iloc[-1])


def build_gatun_analog_overlays(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                                 analog_years: list[int] = GATUN_ANALOG_YEARS) -> dict[int, pd.DataFrame]:
    """Time-shift each analog year's observed trajectory onto today's calendar."""
    if hist_df.empty:
        return {}
    window_start, window_days, _ = _gatun_window(proj_df)
    overlays = {}
    for year in analog_years:
        sub = _gatun_analog_series(hist_df, year, window_start, window_days)
        if not sub.empty:
            overlays[year] = sub.rename(columns={'display_date': 'date'})
    return overlays


def build_gatun_analog_extension(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                                  analog_years: list[int] = GATUN_ANALOG_YEARS,
                                  extra_lag_days: int = GATUN_EXTRA_LAG_DAYS
                                  ) -> tuple[dict[int, pd.Series], pd.Series | None]:
    """Extend the official forecast past its horizon using each analog's own path.

    Returns ({year: level series indexed by date}, mean across those years).
    The mean is the headline line; the per-year curves show the spread it hides.
    """
    if hist_df.empty:
        return {}, None

    window_start, window_days, forecast_end = _gatun_window(proj_df, extra_lag_days)
    end_level = _forecast_end_level(hist_df, proj_df)

    extensions = {}
    for year in analog_years:
        sub = _gatun_analog_series(hist_df, year, window_start, window_days)
        series = _anchor_to_forecast_end(sub, forecast_end, end_level)
        if series is not None:
            extensions[year] = series

    if not extensions:
        return {}, None
    mean_series = pd.concat(extensions.values(), axis=1).mean(axis=1, skipna=True)
    return extensions, mean_series


def build_gatun_composite_forecast(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                                    composite_years: list[int] = GATUN_COMPOSITE_YEARS,
                                    horizon_days: int = GATUN_ENSO_HORIZON_DAYS
                                    ) -> tuple[dict[int, pd.Series], pd.Series | None, pd.Series | None]:
    """Mean post-forecast trajectory of the strongest El Nino events on record.

    The alternative to the regression model below: rather than fitting a
    continuous slope against Nino3.4 — which historically explains very little
    of Gatun's month-to-month change, and would have to be extrapolated past any
    observed index value — this just averages what the lake actually did in the
    strongest events. Returns ({year: series}, mean, std across years).
    """
    if hist_df.empty:
        return {}, None, None
    extensions, mean_series = build_gatun_analog_extension(
        hist_df, proj_df, composite_years, extra_lag_days=horizon_days)
    if not extensions:
        return {}, None, None
    std_series = pd.concat(extensions.values(), axis=1).std(axis=1, skipna=True)
    return extensions, mean_series, std_series


# ─── ENSO index model ────────────────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False)
def load_oni_monthly() -> pd.DataFrame:
    """NOAA CPC Oceanic Nino Index — monthly 3-month-running-mean Nino3.4 anomaly.

    Source: cpc.ncep.noaa.gov/data/indices/oni.ascii.txt (1950-present, updated
    monthly). Columns SEAS/YR/TOTAL/ANOM, one row per overlapping 3-month season;
    CPC labels each season by the year of its centre month, so 'DJF 1998' is
    Dec 1997-Feb 1998 centred on Jan 1998. Returns (date, oni) at month starts.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    resp = requests.get(CPC_ONI_URL, timeout=30, verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; LPG-Desk-Dashboard/1.0)"})
    resp.raise_for_status()

    rows = []
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] not in ONI_SEASON_CENTRE_MONTH:
            continue  # header and any malformed line
        try:
            year, oni = int(parts[1]), float(parts[3])
        except ValueError:
            continue
        rows.append((pd.Timestamp(year=year, month=ONI_SEASON_CENTRE_MONTH[parts[0]], day=1), oni))

    if not rows:
        raise RuntimeError("CPC ONI file returned no parseable rows")
    return pd.DataFrame(rows, columns=['date', 'oni']).sort_values('date').reset_index(drop=True)


def oni_to_daily(oni_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate the monthly ONI onto a daily index for the sensitivity fit.

    The regression below pairs each day's lake-level change with that day's index
    value; ONI is a 3-month running mean, so linear interpolation between month
    centres adds no false precision — it just avoids 12 step discontinuities a
    year that would otherwise land arbitrarily in the monthly buckets.
    """
    if oni_df.empty:
        return pd.DataFrame(columns=['date', 'index'])
    s = (oni_df.set_index('date')['oni']
         .resample('D').interpolate('time'))
    return pd.DataFrame({'date': s.index, 'index': s.values})


def list_el_nino_onset_years(oni_df: pd.DataFrame,
                             threshold: float = GATUN_EL_NINO_ONI_THRESHOLD,
                             first_year: int = GATUN_ANALOG_HISTORY_START) -> dict[int, float]:
    """El Nino onset years covered by the ACP archive, with their Sep-Jan mean ONI.

    An onset year Y is scored over Sep Y - Jan Y+1 — the window that matters for
    Panama, since that's when a developing El Nino suppresses the rains feeding
    the lake ahead of the following dry season. Returns {year: mean ONI},
    strongest first, which is also the order the analog picker offers them in.
    """
    if oni_df.empty:
        return {}
    m = oni_df.copy()
    # Label Sep-Dec with their own year and Jan with the previous one, so a
    # single groupby collects each event's whole Sep-Jan window.
    m['onset_year'] = m['date'].dt.year.where(m['date'].dt.month >= 9,
                                              m['date'].dt.year - 1)
    m = m[m['date'].dt.month.isin([9, 10, 11, 12, 1])]

    scores = {}
    for year, grp in m.groupby('onset_year'):
        if year < first_year or len(grp) < 5:   # need the full Sep-Jan window
            continue
        mean_oni = float(grp['oni'].mean())
        if mean_oni >= threshold:
            scores[int(year)] = round(mean_oni, 2)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def fit_gatun_enso_sensitivity(hist_df: pd.DataFrame, nino_daily: pd.DataFrame,
                                min_samples: int = GATUN_ENSO_MIN_SAMPLES) -> dict[int, tuple[float, float]]:
    """Per-calendar-month fit of daily lake-level change against the Nino3.4 index.

    For each month, regresses level_change = a + b * index across every
    historical day where both series exist. `b` is that month's ENSO sensitivity
    (feet of daily change per unit of index) and `a` its ENSO-neutral baseline.
    Months with too few overlapping days keep their mean change and a zero slope,
    so a thin month contributes climatology rather than a noisy trend.
    """
    lake = hist_df.sort_values('date').copy()
    lake['delta'] = lake['level_ft'].diff()
    merged = (lake[['date', 'delta']]
              .merge(nino_daily, on='date', how='inner')
              .dropna(subset=['delta', 'index']))
    merged['month'] = merged['date'].dt.month

    coeffs = {}
    for month in range(1, 13):
        sub = merged[merged['month'] == month]
        if len(sub) < min_samples:
            coeffs[month] = (float(sub['delta'].mean()) if len(sub) else 0.0, 0.0)
            continue
        slope, intercept = np.polyfit(sub['index'], sub['delta'], 1)
        coeffs[month] = (float(intercept), float(slope))
    return coeffs


def _nino_scenario_value(date: pd.Timestamp, elevated_end: pd.Timestamp,
                         decay_end: pd.Timestamp, target_index: float) -> float:
    """Assumed index on `date`: flat at target through January, then linear to neutral."""
    if date <= elevated_end:
        return target_index
    if date <= decay_end:
        return target_index * (1 - (date - elevated_end).days / (decay_end - elevated_end).days)
    return 0.0


def _roll_enso_scenario(coeffs: dict[int, tuple[float, float]], forecast_start: pd.Timestamp,
                        start_level: float, horizon_days: int, target_index: float,
                        decay_months: int = GATUN_ENSO_DECAY_MONTHS) -> pd.Series:
    """Roll the daily sensitivity model forward from `start_level`, holding the
    index at `target_index` through the upcoming January and decaying it linearly
    to neutral over `decay_months` after that."""
    jan31 = pd.Timestamp(year=forecast_start.year, month=1, day=31)
    elevated_end = jan31 if jan31 > forecast_start else pd.Timestamp(year=forecast_start.year + 1, month=1, day=31)
    decay_end = elevated_end + pd.DateOffset(months=decay_months)

    dates = pd.date_range(start=forecast_start, periods=horizon_days, freq='D')
    levels, level = [], start_level
    for d in dates:
        intercept, slope = coeffs[d.month]
        level += intercept + slope * _nino_scenario_value(d, elevated_end, decay_end, target_index)
        levels.append(level)
    return pd.Series(levels, index=dates)


def build_gatun_enso_index_forecast(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                                     nino_daily: pd.DataFrame, target_index: float,
                                     band: float = GATUN_ENSO_INDEX_BAND,
                                     horizon_days: int = GATUN_ENSO_HORIZON_DAYS
                                     ) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    """Central / low / high ENSO-model forecasts under an assumed Nino3.4 level.

    Low and high re-run the same model at target -/+ `band`, so the ribbon shows
    sensitivity to the index assumption — not a statistical confidence interval.
    """
    if hist_df.empty or nino_daily.empty:
        return None, None, None

    coeffs = fit_gatun_enso_sensitivity(hist_df, nino_daily)
    _, _, forecast_end = _gatun_window(proj_df)
    start_level = _forecast_end_level(hist_df, proj_df)
    forecast_start = forecast_end + pd.Timedelta(days=1)

    return (
        _roll_enso_scenario(coeffs, forecast_start, start_level, horizon_days, target_index),
        _roll_enso_scenario(coeffs, forecast_start, start_level, horizon_days, target_index - band),
        _roll_enso_scenario(coeffs, forecast_start, start_level, horizon_days, target_index + band),
    )


# ─── Panama Canal watershed precipitation ────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def load_panama_precip_anomaly() -> pd.DataFrame:
    """Canal watershed area-average precipitation vs day-of-year climatology.

    Populated by lpg_desk_refresh.py from ERA5 actuals + ECMWF-ENS forecast
    against a 2000-2024 'mix' normal. Columns: delivery_date, source
    ('actual'/'forecast'), precip_mm, precip_clim_mm, precip_clim_std_mm,
    deviation_pct.
    """
    sql = f"""
    SELECT delivery_date, source, precip_mm, precip_clim_mm, precip_clim_std_mm, deviation_pct
    FROM {SBX_SCHEMA}.panama_precip_anomaly
    ORDER BY delivery_date
    """
    df = run_query(sql)
    for c in ["precip_mm", "precip_clim_mm", "precip_clim_std_mm", "deviation_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


def summarize_panama_precip(df: pd.DataFrame,
                            lookback_days: int = PANAMA_PRECIP_LOOKBACK_DAYS,
                            forecast_days: int = PANAMA_PRECIP_FORECAST_DAYS) -> dict:
    """Totals vs normal over the recent-actuals and forecast windows.

    Each window reports accumulated mm, the normal for those same calendar days,
    and the departure in mm and %. Comparing accumulations (rather than averaging
    daily deviation %) keeps a single dry day near zero normal from dominating —
    what matters to the lake is total water in, not the ratio on any one day.
    """
    out = {"lookback": None, "forecast": None, "cumulative_departure_mm": None}
    if df.empty:
        return out

    def _window(sub: pd.DataFrame, days: int, take_last: bool) -> dict | None:
        sub = sub.dropna(subset=["precip_mm"]).sort_values("delivery_date")
        if sub.empty:
            return None
        sub = sub.tail(days) if take_last else sub.head(days)
        actual = float(sub["precip_mm"].sum())
        normal = float(sub["precip_clim_mm"].sum()) if sub["precip_clim_mm"].notna().any() else None
        dev_mm = actual - normal if normal is not None else None
        dev_pct = (100.0 * dev_mm / normal) if normal else None
        return {
            "days": len(sub),
            "start": sub["delivery_date"].iloc[0],
            "end": sub["delivery_date"].iloc[-1],
            "actual_mm": actual,
            "normal_mm": normal,
            "departure_mm": dev_mm,
            "departure_pct": dev_pct,
        }

    out["lookback"] = _window(df[df["source"] == "actual"], lookback_days, take_last=True)
    out["forecast"] = _window(df[df["source"] == "forecast"], forecast_days, take_last=False)

    # Running mm ahead of / behind normal across every actual on file — the
    # slow-moving number, since the lake integrates rainfall over months.
    act = df[(df["source"] == "actual")].dropna(subset=["precip_mm", "precip_clim_mm"])
    if not act.empty:
        out["cumulative_departure_mm"] = float((act["precip_mm"] - act["precip_clim_mm"]).sum())
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PADD 3 COASTAL TEMPERATURES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_padd3_coastal_temps() -> pd.DataFrame:
    """Temperature actuals + forecast for PADD 3 coastal export terminals."""
    sql = f"""
    SELECT city, delivery_date, source, temperature_c
    FROM {SBX_SCHEMA}.padd3_coastal_temps
    ORDER BY city, delivery_date
    """
    df = run_query(sql)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: HURRICANES — live NHC + IBTrACS, scoped to US Gulf/Atlantic + NW Australia
# ══════════════════════════════════════════════════════════════════════════════

def _kt_to_category(wind_kt) -> str:
    """Saffir-Simpson / tropical cyclone category from 1-min sustained wind (kt)."""
    try:
        w = float(wind_kt)
    except (TypeError, ValueError):
        return 'TD'
    if w < 34:    return 'TD'
    elif w < 64:  return 'TS'
    elif w < 83:  return 'Cat 1'
    elif w < 96:  return 'Cat 2'
    elif w < 113: return 'Cat 3'
    elif w < 137: return 'Cat 4'
    else:         return 'Cat 5'


def _in_bbox(lat: float, lon: float, bbox: dict) -> bool:
    return bbox["lat_min"] <= lat <= bbox["lat_max"] and bbox["lon_min"] <= lon <= bbox["lon_max"]


_MONTH_NUM = {m: i + 1 for i, m in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])}


def _parse_nhc_forecast_positions(text: str) -> list[dict]:
    """Parse the forecast track out of an NHC Forecast/Advisory (TCM) product.

    NHC's real format is prose, not the old fixed-width table this used to
    assume: the initial position reads "...CENTER LOCATED NEAR 28.7N  86.4W
    AT 21/0900Z" and each forecast point is "FORECAST VALID 21/1800Z 29.0N
    86.8W" / "OUTLOOK VALID 25/0600Z ...". Valid times only carry a
    day-of-month + HHMM, so hours-from-now is computed by resolving each into
    a real datetime against the advisory's issuance month/year, rolling the
    month forward whenever the day number goes backwards (i.e. the forecast
    crosses a month boundary).
    """
    month_m = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})\s+(\d{4})\b', text)
    init_m = re.search(
        r'CENTER LOCATED NEAR\s+(\d+\.?\d*)\s*([NS])\s+(\d+\.?\d*)\s*([EW])\s+AT\s+(\d{1,2})/(\d{4})Z',
        text,
    )
    if not month_m or not init_m:
        return []

    year, month = int(month_m.group(3)), _MONTH_NUM[month_m.group(1)]
    init_day = int(init_m.group(5))
    init_hhmm = init_m.group(6)
    try:
        init_dt = datetime(year, month, init_day, int(init_hhmm[:2]), int(init_hhmm[2:]))
    except ValueError:
        return []

    out = []
    prev_day = init_day
    for m in re.finditer(
        r'(?:FORECAST|OUTLOOK)\s+VALID\s+(\d{1,2})/(\d{4})Z\s+(\d+\.?\d*)\s*([NS])\s+(\d+\.?\d*)\s*([EW])',
        text,
    ):
        day, hhmm, lat_v, lat_h, lon_v, lon_h = m.groups()
        day = int(day)
        if day < prev_day:   # forecast valid time rolled into the next month
            month += 1
            if month > 12:
                month, year = 1, year + 1
        prev_day = day
        try:
            valid_dt = datetime(year, month, day, int(hhmm[:2]), int(hhmm[2:]))
        except ValueError:
            continue
        lat = float(lat_v) * (1 if lat_h.upper() == 'N' else -1)
        lon = float(lon_v) * (1 if lon_h.upper() == 'E' else -1)
        hours = round((valid_dt - init_dt).total_seconds() / 3600)
        out.append({'hours': hours, 'lat': lat, 'lon': lon})
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def load_hurricane_data() -> tuple[list[dict], dict[str, str]]:
    """Fetch active tropical cyclones affecting US Gulf/Atlantic or NW Australia.

    NHC CurrentStorms.json covers the Atlantic; IBTrACS' near-real-time
    ACTIVE feed is the only source with Southern Hemisphere coverage, used
    here for NW Australia. Both are public, no auth. Returns (storms, sources).

    verify=False on every request here: some corporate egress paths (proxy
    TLS inspection) present a self-signed cert for these .gov/.noaa domains,
    same tradeoff already made for the Gatun Lake ACP fetch above.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; LPG-Desk-Dashboard/1.0)"}
    storms: list[dict] = []
    sources: dict[str, str] = {}

    us_box = HURRICANE_BASINS["US Gulf & Atlantic"]
    au_box = HURRICANE_BASINS["NW Australia (Cyclone)"]

    # ── NHC: Atlantic ──────────────────────────────────────────────────────────
    try:
        r = requests.get("https://www.nhc.noaa.gov/CurrentStorms.json", headers=headers, timeout=20, verify=False)
        if r.status_code == 200:
            n_added = 0
            for s in r.json().get('activeStorms', []):
                sid = s.get('id', '')
                if not sid.startswith('al'):   # Atlantic basin storms only
                    continue
                lat, lon = s.get('latitudeNumeric', 0.0), s.get('longitudeNumeric', 0.0)
                if not _in_bbox(lat, lon, us_box):
                    continue
                wind_kt = int(s.get('intensity', 0) or 0)
                storm = {
                    'id': sid, 'name': s.get('name', 'Unknown'), 'basin': 'US Gulf & Atlantic',
                    'lat': lat, 'lon': lon, 'wind_kt': wind_kt,
                    'pressure_mb': s.get('pressure', 'N/A'),
                    'category': _kt_to_category(wind_kt),
                    'classification': s.get('classification', 'TD'),
                    'last_update': s.get('lastUpdate', ''),
                    'forecast_track': [],
                    'observed_track': [],   # filled in below from IBTrACS (global best-track), if matched
                }
                fcst_url = (s.get('forecastAdvisory') or {}).get('url', '')
                if fcst_url:
                    try:
                        fr = requests.get(fcst_url, headers=headers, timeout=15, verify=False)
                        if fr.status_code == 200:
                            storm['forecast_track'] = _parse_nhc_forecast_positions(fr.text)
                    except Exception:
                        pass
                storms.append(storm)
                n_added += 1
            sources['NHC (Atlantic)'] = f'ok ({n_added} storm(s) in region)'
        else:
            sources['NHC (Atlantic)'] = f'HTTP {r.status_code}'
    except Exception as e:
        sources['NHC (Atlantic)'] = str(e)

    # ── IBTrACS ACTIVE: observed tracks (global) + NW Australia storms ─────────
    # IBTrACS is the only source used for NW Australia storms at all (no NHC
    # coverage there), but it's also a global best-track feed — every row for
    # a still-active storm's full history, not just its latest position — so
    # the same fetch doubles as the *observed* (past) track for the Atlantic
    # storms above too, which until now only ever carried a forecast track.
    try:
        ibtracs_url = (
            "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
            "stewardship-ibtracs/v04r00/access/csv/ibtracs.ACTIVE.list.v04r00.csv"
        )
        ir = requests.get(ibtracs_url, headers=headers, timeout=30, verify=False)
        if ir.status_code == 200:
            df = pd.read_csv(io.StringIO(ir.text), skiprows=[1], low_memory=False)
            df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
            df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
            df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
            df = df.dropna(subset=['ISO_TIME', 'LAT', 'LON'])
            if 'TRACK_TYPE' in df.columns:
                prov = df[df['TRACK_TYPE'].astype(str).str.strip().str.lower() == 'provisional']
                if not prov.empty:
                    df = prov

            def _track_points(sub_df: pd.DataFrame) -> list[dict]:
                sub_df = sub_df.sort_values('ISO_TIME')
                return [{'lat': float(r['LAT']), 'lon': float(r['LON'])} for _, r in sub_df.iterrows()]

            tracks_by_sid: dict[str, list[dict]] = {}
            tracks_by_name: dict[str, list[dict]] = {}
            for sid, g in df.groupby('SID'):
                pts = _track_points(g)
                tracks_by_sid[str(sid)] = pts
                nm = str(g['NAME'].iloc[-1]).strip().upper()
                if nm and nm not in ('UNNAMED', 'NOT_NAMED', 'NAN'):
                    tracks_by_name[nm] = pts

            # Attach the observed track to the Atlantic storms found via NHC above.
            for s in storms:
                if s['basin'] == 'US Gulf & Atlantic':
                    pts = tracks_by_name.get(s['name'].strip().upper(), [])
                    if len(pts) > 1:
                        s['observed_track'] = pts[:-1]   # last point == current position, drawn separately

            latest = df.sort_values('ISO_TIME').groupby('SID').last().reset_index()
            cutoff = pd.Timestamp(datetime.utcnow() - timedelta(hours=48))
            latest = latest[latest['ISO_TIME'] >= cutoff]

            n_added = 0
            for _, row in latest.iterrows():
                try:
                    lat, lon = float(row['LAT']), float(row['LON'])
                except (ValueError, TypeError):
                    continue
                if abs(lat) > 90 or abs(lon) > 180 or not _in_bbox(lat, lon, au_box):
                    continue
                name = str(row.get('NAME', '')).strip()
                if not name or name.upper() in ('UNNAMED', 'NOT_NAMED', 'NAN'):
                    name = f"TC-{row.get('SID', '?')}"
                wind_kt = 0
                for wcol in ('WMO_WIND', 'USA_WIND'):
                    raw = str(row.get(wcol, '')).strip()
                    if raw and raw.lower() != 'nan':
                        wind_kt = int(float(raw))
                        break
                pressure = 'N/A'
                for pcol in ('WMO_PRES', 'USA_PRES'):
                    raw = str(row.get(pcol, '')).strip()
                    if raw and raw.lower() != 'nan':
                        pressure = raw
                        break
                sid_str = str(row.get('SID', ''))
                track_pts = tracks_by_sid.get(sid_str, [])
                storms.append({
                    'id': sid_str or name, 'name': name.capitalize(),
                    'basin': 'NW Australia (Cyclone)', 'lat': lat, 'lon': lon, 'wind_kt': wind_kt,
                    'pressure_mb': pressure, 'category': _kt_to_category(wind_kt),
                    'classification': 'TY' if wind_kt >= 64 else ('TS' if wind_kt >= 34 else 'TD'),
                    'last_update': str(row['ISO_TIME'])[:16], 'forecast_track': [],
                    'observed_track': track_pts[:-1] if len(track_pts) > 1 else [],
                })
                n_added += 1
            sources['IBTrACS (NW Australia)'] = f'ok ({n_added} storm(s) in region)'
        else:
            sources['IBTrACS (NW Australia)'] = f'HTTP {ir.status_code}'
    except Exception as e:
        sources['IBTrACS (NW Australia)'] = str(e)

    return storms, sources


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: NORTH AFRICA WIND SPEEDS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_nafrica_wind() -> pd.DataFrame:
    """Wind speeds (actual + forecast) at North Africa corridor ports."""
    sql = f"""
    SELECT city, delivery_date, source,
           wind_p10, wind_p50, wind_p90, pct_above_warning, pct_above_gale
    FROM {SBX_SCHEMA}.nafrica_wind
    ORDER BY city, delivery_date
    """
    df = run_query(sql)
    for col in ["wind_p10", "wind_p50", "wind_p90", "pct_above_warning", "pct_above_gale"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df
