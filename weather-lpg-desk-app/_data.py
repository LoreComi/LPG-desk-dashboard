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

import pandas as pd
import requests
import streamlit as st

from _config import (
    SBX_SCHEMA,
    GATUN_HIST_URL, GATUN_PROJECTION_URL, GATUN_ANALOG_YEARS,
    HURRICANE_BASINS,
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


def build_gatun_analog_overlays(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                                 analog_years: list[int] = GATUN_ANALOG_YEARS) -> dict[int, pd.DataFrame]:
    """Time-shift each analog year's observed trajectory onto today's calendar.

    For an analog year Y, pulls Y-1's data starting from today's month/day one
    year back through the same span covered by hist+forecast, then shifts it
    forward so that start lands exactly on today — landing all analog curves
    on the same x-axis as the current observed/forecast window.
    """
    if hist_df.empty:
        return {}

    window_start = pd.Timestamp.today().normalize() - pd.DateOffset(days=365)
    forecast_end = proj_df['date'].max() if not proj_df.empty else pd.Timestamp.today()
    window_end = forecast_end + pd.Timedelta(days=350)
    window_days = (window_end - window_start).days

    overlays = {}
    for year in analog_years:
        start_year = year - 1
        try:
            analog_start = window_start.replace(year=start_year)
        except ValueError:
            analog_start = window_start.replace(year=start_year, day=28)
        analog_end = analog_start + pd.Timedelta(days=window_days)

        mask = (hist_df['date'] >= analog_start) & (hist_df['date'] <= analog_end)
        sub = hist_df.loc[mask].copy()
        if sub.empty:
            continue
        shift = window_start - analog_start
        sub['display_date'] = sub['date'] + shift
        overlays[year] = sub[['display_date', 'level_ft']].rename(columns={'display_date': 'date'})

    return overlays


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


def _parse_nhc_forecast_positions(text: str) -> list[dict]:
    """Parse 5-day forecast positions out of an NHC forecast advisory."""
    pattern = re.compile(
        r'(INIT|\d+\s*H[RS]?)\s+\d+/\d{4}Z\s+(\d+\.?\d*)\s*([NS])\s+(\d+\.?\d*)\s*([EW])\s+(\d+)\s+KT',
        re.IGNORECASE,
    )
    out = []
    for m in pattern.finditer(text):
        step, lat_v, lat_h, lon_v, lon_h, wind = m.groups()
        step_clean = re.sub(r'[^0-9]', '', step)
        hours = 0 if step.upper().strip() == 'INIT' else (int(step_clean) if step_clean else 0)
        lat = float(lat_v) * (1 if lat_h.upper() == 'N' else -1)
        lon = float(lon_v) * (1 if lon_h.upper() == 'E' else -1)
        out.append({'hours': hours, 'lat': lat, 'lon': lon, 'wind_kt': int(wind)})
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
                }
                fcst_url = (s.get('forecastAdvisory') or {}).get('product', '')
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

    # ── IBTrACS ACTIVE: NW Australia (Southern Hemisphere cyclones) ────────────
    try:
        ibtracs_url = (
            "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
            "stewardship-ibtracs/v04r00/access/csv/ibtracs.ACTIVE.list.v04r00.csv"
        )
        ir = requests.get(ibtracs_url, headers=headers, timeout=30, verify=False)
        if ir.status_code == 200:
            df = pd.read_csv(io.StringIO(ir.text), skiprows=[1], low_memory=False)
            df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
            df = df.dropna(subset=['ISO_TIME', 'LAT', 'LON'])
            if 'TRACK_TYPE' in df.columns:
                prov = df[df['TRACK_TYPE'].astype(str).str.strip().str.lower() == 'provisional']
                if not prov.empty:
                    df = prov
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
                storms.append({
                    'id': str(row.get('SID', name)), 'name': name.capitalize(),
                    'basin': 'NW Australia (Cyclone)', 'lat': lat, 'lon': lon, 'wind_kt': wind_kt,
                    'pressure_mb': pressure, 'category': _kt_to_category(wind_kt),
                    'classification': 'TY' if wind_kt >= 64 else ('TS' if wind_kt >= 34 else 'TD'),
                    'last_update': str(row['ISO_TIME'])[:16], 'forecast_track': [],
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
