"""Configuration — LPG Desk Weather Dashboard.

Covers: US HDD by PADD, Asia HDD (Japan/Korea/China), India precipitation,
Gatun Lake analog forecast, PADD 3 coastal temperatures, hurricanes
(US Gulf/Atlantic + NW Australia), North Africa high wind speeds.

Series colors are assigned from the fixed 8-hue categorical palette in
_style.py, in a fixed order per tab — never cycled, never re-assigned when
the selection changes (color follows the entity, not its rank in the list).
"""
from __future__ import annotations

from _style import CATEGORICAL, STORM_RAMP


def _assign_categorical_colors(names, priority: list[str] = ()) -> dict[str, str]:
    """Assign the 8-hue categorical palette, giving `priority` names first pick.

    With more than 8 entities two names must share a hue no matter the
    ordering (there are only 8 categorical slots) — putting the
    commonly-shown/default names first guarantees the collision, if any,
    lands on names that are rarely shown together instead of on the default
    view.
    """
    ordered = [n for n in priority if n in names] + [n for n in names if n not in priority]
    return {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(ordered)}

# ══════════════════════════════════════════════════════════════════════════════
# GENERAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
HDD_BASE_TEMP = 18.0       # °C — Heating Degree Day base
SEASON_START_MONTH = 10    # HDD season starts October
SEASON_START_DAY = 1
HIST_START_YEAR = 2000
HIST_END_YEAR = 2025

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE TABLES (sandbox — populated by lpg_desk_refresh.py)
# ══════════════════════════════════════════════════════════════════════════════
SBX_SCHEMA = "dna_snbx_weather.lpg_desk"

# Production silver tables + correct column names (curve_name, delivery_start),
# used by the refresh job — documented here for reference, not imported by it
# (Databricks notebooks are standalone and inline these constants themselves).
TABLE_TEMP_ACTUALS = "dna_prod_silver.meteomatics.temperature"
TABLE_TEMP_FORECAST = "dna_prod_silver.meteomatics.temperature_forecast"
TABLE_PRECIP_ACTUALS = "dna_prod_silver.meteomatics.precipitation"
TABLE_PRECIP_FORECAST = "dna_prod_silver.meteomatics.precipitation_forecast"
TABLE_WIND_ACTUALS = "dna_prod_silver.meteomatics.wind"
TABLE_WIND_FORECAST = "dna_prod_silver.meteomatics.wind_forecast"

CURVE_TEMP_HIST = "t_mean_2m_24h_c_ecmwf_era5_p1d"
CURVE_TEMP_FCST = "t_mean_2m_24h_c_ecmwf_ens_p1d"
CURVE_PRECIP_HIST = "precip_24h_mm_ecmwf_era5_p1d"
CURVE_PRECIP_FCST = "precip_24h_mm_ecmwf_ens_p1d"
CURVE_WIND_HIST = "wind_speed_10m_ms_ecmwf_era5_p1d"
CURVE_WIND_FCST = "wind_speed_10m_ms_ecmwf_ens_p1d"
MODEL_HIST = "ecmwf-era5"
MODEL_FCST = "ecmwf-ens"

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: US HDD BY PADD
# ══════════════════════════════════════════════════════════════════════════════
# Bounding boxes — half-open [min, max) to avoid double-counting
PADD_REGIONS: dict[str, dict] = {
    "PADD 1 - Northeast":     {"lat_min": 36.5, "lat_max": 47.5, "lon_min": -80.0, "lon_max": -66.5},
    "PADD 2 - Midwest":       {"lat_min": 36.5, "lat_max": 49.0, "lon_min": -97.5, "lon_max": -80.0},
    "PADD 3 - South Central": {"lat_min": 25.5, "lat_max": 36.5, "lon_min": -106.0, "lon_max": -79.5},
    "PADD 4 - Mountain":      {"lat_min": 36.5, "lat_max": 49.0, "lon_min": -116.5, "lon_max": -97.5},
    "PADD 5 - West":          {"lat_min": 32.0, "lat_max": 49.0, "lon_min": -125.0, "lon_max": -116.5},
}

PADD_COLORS: dict[str, str] = dict(zip(PADD_REGIONS.keys(), CATEGORICAL))

US_LAT_MIN, US_LAT_MAX = 24.0, 50.0
US_LON_MIN, US_LON_MAX = -125.0, -65.0

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASIA HDD — Japan, South Korea, China (North, East, South)
# ══════════════════════════════════════════════════════════════════════════════
ASIA_CITY_LOCATIONS: dict[str, dict] = {
    # Japan
    'Tokyo':     {'latitude': 35.5, 'longitude': 139.5},
    'Osaka':     {'latitude': 34.5, 'longitude': 135.5},
    'Niigata':   {'latitude': 38.0, 'longitude': 139.0},
    'Sendai':    {'latitude': 38.5, 'longitude': 141.0},
    'Sapporo':   {'latitude': 43.0, 'longitude': 141.5},
    'Fukuoka':   {'latitude': 33.5, 'longitude': 130.5},
    'Nagoya':    {'latitude': 35.0, 'longitude': 137.0},
    # South Korea
    'Seoul':     {'latitude': 37.5, 'longitude': 127.0},
    'Busan':     {'latitude': 35.0, 'longitude': 129.0},
    'Incheon':   {'latitude': 37.5, 'longitude': 126.5},
    'Daegu':     {'latitude': 36.0, 'longitude': 128.5},
    # China North (heating demand heavy)
    'Beijing':       {'latitude': 40.0, 'longitude': 116.5},
    'Tianjin':       {'latitude': 39.5, 'longitude': 117.5},
    'Harbin':        {'latitude': 45.5, 'longitude': 126.5},
    'Shenyang':      {'latitude': 42.0, 'longitude': 123.5},
    'Changchun':     {'latitude': 44.0, 'longitude': 125.5},
    'Hohhot':        {'latitude': 41.0, 'longitude': 111.5},
    'Urumqi':        {'latitude': 44.0, 'longitude': 87.5},
    'Shijiazhuang':  {'latitude': 38.0, 'longitude': 114.5},
    'Jinan':         {'latitude': 36.5, 'longitude': 117.0},
    'Xian':          {'latitude': 34.5, 'longitude': 109.0},
    # China East
    'Shanghai':  {'latitude': 31.0, 'longitude': 121.5},
    'Nanjing':   {'latitude': 32.0, 'longitude': 119.0},
    'Hangzhou':  {'latitude': 30.5, 'longitude': 120.0},
    'Wuhan':     {'latitude': 30.5, 'longitude': 114.5},
    'Hefei':     {'latitude': 32.0, 'longitude': 117.0},
    'Zhengzhou': {'latitude': 35.0, 'longitude': 113.5},
    # China South
    'Guangzhou': {'latitude': 23.0, 'longitude': 113.5},
    'Shenzhen':  {'latitude': 22.5, 'longitude': 114.0},
    'Kunming':   {'latitude': 25.0, 'longitude': 102.5},
    'Chengdu':   {'latitude': 30.5, 'longitude': 104.0},
    'Chongqing': {'latitude': 29.5, 'longitude': 107.0},
    'Nanning':   {'latitude': 23.0, 'longitude': 108.5},
}

ASIA_POPULATION: dict[str, int] = {
    'Tokyo': 14000000, 'Osaka': 8800000, 'Niigata': 800000,
    'Sendai': 1100000, 'Sapporo': 2000000, 'Fukuoka': 1600000, 'Nagoya': 2300000,
    'Seoul': 9700000, 'Busan': 3400000, 'Incheon': 3000000, 'Daegu': 2400000,
    'Beijing': 21830000, 'Tianjin': 13640000, 'Harbin': 10000000,
    'Shenyang': 9000000, 'Changchun': 7700000, 'Hohhot': 3500000,
    'Urumqi': 4000000, 'Shijiazhuang': 11000000, 'Jinan': 9200000, 'Xian': 13000000,
    'Shanghai': 24800000, 'Nanjing': 9400000, 'Hangzhou': 12200000,
    'Wuhan': 12300000, 'Hefei': 9400000, 'Zhengzhou': 12700000,
    'Guangzhou': 18700000, 'Shenzhen': 17600000, 'Kunming': 8500000,
    'Chengdu': 21000000, 'Chongqing': 32000000, 'Nanning': 8800000,
}

ASIA_REGION_MAP: dict[str, list[str]] = {
    'Japan': ['Tokyo', 'Osaka', 'Niigata', 'Sendai', 'Sapporo', 'Fukuoka', 'Nagoya'],
    'South Korea': ['Seoul', 'Busan', 'Incheon', 'Daegu'],
    'China North': ['Beijing', 'Tianjin', 'Harbin', 'Shenyang', 'Changchun',
                    'Hohhot', 'Urumqi', 'Shijiazhuang', 'Jinan', 'Xian'],
    'China East': ['Shanghai', 'Nanjing', 'Hangzhou', 'Wuhan', 'Hefei', 'Zhengzhou'],
    'China South': ['Guangzhou', 'Shenzhen', 'Kunming', 'Chengdu', 'Chongqing', 'Nanning'],
}

ASIA_COLORS: dict[str, str] = dict(zip(ASIA_REGION_MAP.keys(), CATEGORICAL))
DEFAULT_ASIA_REGIONS = list(ASIA_REGION_MAP.keys())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: INDIA PRECIPITATION ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
INDIA_PRECIP_BBOX = {"lat_min": 8.0, "lat_max": 35.0, "lon_min": 68.0, "lon_max": 90.0}

INDIA_SUBREGIONS: dict[str, dict] = {
    "Northwest": {"lat_min": 23.0, "lat_max": 35.0, "lon_min": 68.0, "lon_max": 78.0},
    "Northeast": {"lat_min": 23.0, "lat_max": 35.0, "lon_min": 78.0, "lon_max": 90.0},
    "Central":   {"lat_min": 18.0, "lat_max": 23.0, "lon_min": 72.0, "lon_max": 85.0},
    "South":     {"lat_min": 8.0,  "lat_max": 18.0, "lon_min": 73.0, "lon_max": 85.0},
    "West Coast":{"lat_min": 10.0, "lat_max": 23.0, "lon_min": 68.0, "lon_max": 74.0},
}
INDIA_COLORS: dict[str, str] = dict(zip(INDIA_SUBREGIONS.keys(), CATEGORICAL))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GATUN LAKE — live ACP feed + El Nino analog years
# ══════════════════════════════════════════════════════════════════════════════
# Source: Panama Canal Authority (evtms-rpts.pancanal.com) — fetched live by
# the app, no sandbox table needed. Analog years are the strongest historical
# El Nino events on record (labeled by the year their Panama dry season fell
# in), used to overlay how the lake behaved in comparably strong events.
GATUN_HIST_URL = "https://evtms-rpts.pancanal.com/eng/h2o/Download_Gatun_Lake_Water_Level_History.csv"
GATUN_PROJECTION_URL = "https://evtms-rpts.pancanal.com/eng/h2o/Gatun_Water_Level_Projection.csv"
GATUN_ANALOG_YEARS = [1997, 2015, 2023]
GATUN_ANALOG_COLORS = dict(zip(GATUN_ANALOG_YEARS, CATEGORICAL[1:]))
GATUN_RATE_SURCHARGE_FT = 85.0   # increased shipping rates threshold
GATUN_RESTRICTION_FT = 82.0      # significant draft restrictions threshold

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PADD 3 COASTAL TEMPERATURES
# ══════════════════════════════════════════════════════════════════════════════
# Key LPG export terminals along PADD 3 coast. Default selection favors the
# largest export terminals to avoid an overplotted 8-line chart; the rest are
# one multiselect click away.
PADD3_COASTAL_CITIES: dict[str, dict] = {
    'Houston':       {'latitude': 30.0, 'longitude': -95.5},
    'Freeport':      {'latitude': 29.0, 'longitude': -95.5},
    'Beaumont':      {'latitude': 30.0, 'longitude': -94.0},
    'Lake Charles':  {'latitude': 30.0, 'longitude': -93.0},
    'New Orleans':   {'latitude': 30.0, 'longitude': -90.0},
    'Corpus Christi':{'latitude': 27.5, 'longitude': -97.5},
    'Port Arthur':   {'latitude': 30.0, 'longitude': -94.0},
    'Morgan City':   {'latitude': 29.5, 'longitude': -91.0},
}
DEFAULT_PADD3_CITIES = ['Houston', 'Corpus Christi', 'Lake Charles', 'New Orleans']
PADD3_COASTAL_COLORS: dict[str, str] = _assign_categorical_colors(PADD3_COASTAL_CITIES, DEFAULT_PADD3_CITIES)

# Freeze thresholds that impact operations
FREEZE_THRESHOLD_C = 0.0   # hard freeze
FROST_THRESHOLD_C = 2.0    # frost advisory

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: HURRICANES — US Gulf/Atlantic & NW Australia
# ══════════════════════════════════════════════════════════════════════════════
# Live feed (NHC + IBTrACS active), filtered to these two boxes — no sandbox
# table needed. IBTrACS basin codes: NA=Atlantic, SP/SI cover NW Australia.
HURRICANE_BASINS: dict[str, dict] = {
    "US Gulf & Atlantic": {
        "lat_min": 8.0, "lat_max": 35.0,
        "lon_min": -100.0, "lon_max": -55.0,
        "season_months": [6, 7, 8, 9, 10, 11],
    },
    "NW Australia (Cyclone)": {
        "lat_min": -25.0, "lat_max": -8.0,
        "lon_min": 105.0, "lon_max": 145.0,
        "season_months": [11, 12, 1, 2, 3, 4],
    },
}
STORM_CATEGORY_COLORS = STORM_RAMP

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: NORTH AFRICA HIGH WIND SPEEDS
# ══════════════════════════════════════════════════════════════════════════════
# Tangier -> Algeria -> Port Said shipping corridor
NAFRICA_WIND_CORRIDOR: dict[str, dict] = {
    'Tangier':    {'latitude': 35.5, 'longitude': -6.0},
    'Oran':       {'latitude': 35.5, 'longitude': -0.5},
    'Algiers':    {'latitude': 36.5, 'longitude': 3.0},
    'Bejaia':     {'latitude': 36.5, 'longitude': 5.0},
    'Skikda':     {'latitude': 37.0, 'longitude': 7.0},
    'Tunis':      {'latitude': 37.0, 'longitude': 10.0},
    'Tripoli':    {'latitude': 33.0, 'longitude': 13.0},
    'Benghazi':   {'latitude': 32.0, 'longitude': 20.0},
    'Alexandria': {'latitude': 31.0, 'longitude': 30.0},
    'Port Said':  {'latitude': 31.5, 'longitude': 32.5},
}
# 10 ports > 8 categorical hues — two ports must share a hue no matter the
# ordering. Assigning the 4 default ports first keeps that collision off the
# default view entirely; it only shows up if a trader manually adds enough
# extra ports to exceed the 8-hue budget.
DEFAULT_NAFRICA_PORTS = ['Tangier', 'Algiers', 'Port Said', 'Alexandria']
NAFRICA_WIND_COLORS: dict[str, str] = _assign_categorical_colors(NAFRICA_WIND_CORRIDOR, DEFAULT_NAFRICA_PORTS)

NAFRICA_BBOX = {"lat_min": 30.0, "lat_max": 38.0, "lon_min": -7.0, "lon_max": 33.0}

# Wind speed thresholds (m/s) for shipping disruption
WIND_CAUTION_MS = 10.0     # ~20 knots — small vessel caution
WIND_WARNING_MS = 15.0     # ~30 knots — loading operations suspended
WIND_GALE_MS = 20.0        # ~40 knots — port closure risk
