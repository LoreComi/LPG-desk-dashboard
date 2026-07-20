"""LPG Desk Weather Dashboard — Trader Edition

Tabs:
  1. US HDD by PADD        — Cumulative HDD + run comparison
  2. Asia HDD              — Japan, South Korea, China (N/E/S)
  3. India Precipitation   — Anomaly vs climatology
  4. Gatun Lake            — Seasonal precipitation forecast (analog years)
  5. PADD 3 Coastal Temps  — Export terminal freeze risk
  6. Hurricanes            — US Gulf/Atlantic + Australia cyclones
  7. North Africa Wind     — Tangier-Algeria-Port Said corridor
"""
from __future__ import annotations

import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    PADD_REGIONS, PADD_COLORS, PADD_POPULATION_WEIGHTS,
    ASIA_REGION_MAP, ASIA_COLORS, DEFAULT_ASIA_REGIONS,
    INDIA_SUBREGIONS,
    GATUN_N_ANALOGS,
    PADD3_COASTAL_CITIES, PADD3_COASTAL_COLORS,
    FREEZE_THRESHOLD_C, FROST_THRESHOLD_C,
    HURRICANE_BASINS,
    NAFRICA_WIND_CORRIDOR, WIND_CAUTION_MS, WIND_WARNING_MS, WIND_GALE_MS,
    HDD_BASE_TEMP,
)
from _data import (
    load_hdd_season, load_hdd_normal_daily, load_hdd_run_comparison,
    load_asia_hdd_season, load_asia_hdd_normal,
    load_india_precip_anomaly, load_india_precip_map,
    load_gatun_precip_history, load_gatun_analog_forecast,
    load_padd3_coastal_temps, load_padd3_coastal_climatology,
    load_hurricane_activity, load_hurricane_season_stats,
    load_nafrica_wind, load_nafrica_wind_climatology,
)
from _charts import (
    make_hdd_season_chart, make_hdd_comparison_chart,
    make_asia_hdd_season_chart, make_asia_hdd_bar_chart,
    make_india_precip_timeseries, make_india_precip_deviation_chart,
    make_gatun_analog_chart,
    make_padd3_temp_chart,
    make_hurricane_track_map, make_ace_comparison_chart,
    make_wind_corridor_chart, make_wind_exceedance_chart,
)
from _style import CUSTOM_CSS


st.set_page_config(
    page_title="LPG Desk — Weather Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── KPI Helper ──────────────────────────────────────────────────────────────────
def kpi_card(label: str, value, unit: str, delta=None, card_class: str = "") -> str:
    try:
        val_str = f"{float(value):+.1f}" if value is not None and not np.isnan(float(value)) else "N/A"
    except (TypeError, ValueError):
        val_str = "N/A"
    delta_html = ""
    if delta is not None:
        try:
            d = float(delta)
            if not np.isnan(d):
                d_cls = "kpi-delta-up" if d > 0 else ("kpi-delta-down" if d < 0 else "kpi-delta-flat")
                arrow = "▲" if d > 0 else ("▼" if d < 0 else "●")
                delta_html = f'<div class="kpi-delta {d_cls}">{arrow} {abs(d):.1f} vs normal</div>'
        except (TypeError, ValueError):
            pass
    card_class = card_class or ("kpi-card-warm" if val_str.startswith("+") else "kpi-card-cool" if val_str.startswith("-") else "kpi-card-neutral")
    return f'<div class="kpi-card {card_class}"><div class="kpi-label">{label}</div><div class="kpi-value">{val_str} {unit}</div>{delta_html}</div>'


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: US HDD BY PADD
# ══════════════════════════════════════════════════════════════════════════════
def render_us_hdd():
    st.markdown("#### US HEATING DEGREE DAYS BY PADD")
    st.caption(f"Base {HDD_BASE_TEMP}°C. Season Oct 1 – Mar 31. Source: ERA5 + EC-ens forecast.")

    try:
        season_df = load_hdd_season()
        normal_df = load_hdd_normal_daily()
        run_df = load_hdd_run_comparison()
    except Exception as e:
        st.error(f"Failed to load US HDD data: {e}")
        return

    selected_padds = st.multiselect(
        "PADDs", list(PADD_REGIONS.keys()), list(PADD_REGIONS.keys()),
        key="us_hdd_padds", label_visibility="collapsed",
    )
    if not selected_padds:
        st.info("Select at least one PADD.")
        return

    # Season charts
    cols = st.columns(min(len(selected_padds), 3))
    for i, padd in enumerate(selected_padds):
        with cols[i % len(cols)]:
            fig = make_hdd_season_chart(season_df, padd, normal_df)
            st.plotly_chart(fig, use_container_width=True)

    # Run comparison
    if not run_df.empty:
        st.markdown("---")
        st.markdown("##### Run Comparison (latest 3 model runs)")
        cols2 = st.columns(min(len(selected_padds), 3))
        for i, padd in enumerate(selected_padds):
            with cols2[i % len(cols2)]:
                fig = make_hdd_comparison_chart(run_df, padd)
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASIA HDD
# ══════════════════════════════════════════════════════════════════════════════
def render_asia_hdd():
    st.markdown("#### ASIA HEATING DEGREE DAYS")
    st.caption("Population-weighted HDD. Japan, South Korea, China North/East/South.")

    try:
        season_df = load_asia_hdd_season()
    except Exception as e:
        st.error(f"Failed to load Asia HDD data: {e}")
        return

    selected = st.multiselect(
        "Regions", list(ASIA_REGION_MAP.keys()), DEFAULT_ASIA_REGIONS,
        key="asia_regions", label_visibility="collapsed",
    )
    if not selected:
        st.info("Select at least one region.")
        return

    # Combined cumulative chart
    fig = make_asia_hdd_season_chart(season_df, selected)
    st.plotly_chart(fig, use_container_width=True)

    # 14-day bar comparison
    fig2 = make_asia_hdd_bar_chart(season_df, selected)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: INDIA PRECIPITATION ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
def render_india_precip():
    st.markdown("#### INDIA PRECIPITATION ANOMALY")
    st.caption("Monsoon and off-season precipitation vs ERA5 climatology.")

    try:
        anom_df = load_india_precip_anomaly()
    except Exception as e:
        st.error(f"Failed to load India precipitation data: {e}")
        return

    if anom_df.empty:
        st.info("No India precipitation data available.")
        return

    # KPI row
    kpi_cols = st.columns(len(INDIA_SUBREGIONS))
    latest = anom_df.groupby("subregion").last().reset_index()
    for i, (_, row) in enumerate(latest.iterrows()):
        with kpi_cols[i % len(kpi_cols)]:
            st.markdown(kpi_card(row["subregion"], row["deviation_pct"], "%"), unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = make_india_precip_timeseries(anom_df)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = make_india_precip_deviation_chart(anom_df)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GATUN LAKE
# ══════════════════════════════════════════════════════════════════════════════
def render_gatun_lake():
    st.markdown("#### GATUN LAKE — SEASONAL PRECIPITATION FORECAST")
    st.caption("Panama Canal water levels depend on Gatun Lake catchment rainfall. "
               "Analog years selected by cumulative precipitation similarity.")

    try:
        hist_df = load_gatun_precip_history()
        analog_df = load_gatun_analog_forecast()
    except Exception as e:
        st.error(f"Failed to load Gatun Lake data: {e}")
        return

    current_year = datetime.now().year
    fig = make_gatun_analog_chart(hist_df, analog_df, current_year)
    st.plotly_chart(fig, use_container_width=True)

    # Analog year summary table
    if not analog_df.empty:
        st.markdown("##### Top Analog Years")
        summary = analog_df.groupby("analog_year").agg(
            similarity=pd.NamedAgg(column="similarity_score", aggfunc="first"),
            total_precip_mm=pd.NamedAgg(column="cumulative_precip_mm", aggfunc="max"),
        ).reset_index().sort_values("similarity", ascending=False).head(GATUN_N_ANALOGS)
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PADD 3 COASTAL
# ══════════════════════════════════════════════════════════════════════════════
def render_padd3_coastal():
    st.markdown("#### PADD 3 COASTAL TEMPERATURES")
    st.caption("Key LPG export terminals. Freeze events impact loading operations and vessel scheduling.")

    try:
        temp_df = load_padd3_coastal_temps()
    except Exception as e:
        st.error(f"Failed to load PADD 3 coastal data: {e}")
        return

    if temp_df.empty:
        st.info("No PADD 3 coastal temperature data available.")
        return

    # Freeze alert KPIs
    fcst = temp_df[temp_df["source"] == "forecast"]
    freeze_days = fcst[fcst["temperature_c"] <= FREEZE_THRESHOLD_C]
    frost_days = fcst[(fcst["temperature_c"] > FREEZE_THRESHOLD_C) & (fcst["temperature_c"] <= FROST_THRESHOLD_C)]

    if not freeze_days.empty:
        cities_at_risk = freeze_days["city"].unique().tolist()
        st.markdown(
            f'<div class="freeze-alert">FREEZE ALERT: {len(cities_at_risk)} terminal(s) '
            f'forecast below 0°C — {", ".join(cities_at_risk)}</div>',
            unsafe_allow_html=True,
        )

    selected_cities = st.multiselect(
        "Terminals", list(PADD3_COASTAL_CITIES.keys()), list(PADD3_COASTAL_CITIES.keys()),
        key="padd3_cities", label_visibility="collapsed",
    )

    fig = make_padd3_temp_chart(temp_df, selected_cities)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: HURRICANES
# ══════════════════════════════════════════════════════════════════════════════
def render_hurricanes():
    st.markdown("#### TROPICAL CYCLONES — US & AUSTRALIA")
    st.caption("Active storms + season statistics. LPG supply disruption risk.")

    try:
        activity_df = load_hurricane_activity()
        stats_df = load_hurricane_season_stats()
    except Exception as e:
        st.error(f"Failed to load hurricane data: {e}")
        return

    # Track map
    if not activity_df.empty:
        fig = make_hurricane_track_map(activity_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active tropical cyclones currently tracked.")

    # ACE comparison
    if not stats_df.empty:
        fig2 = make_ace_comparison_chart(stats_df)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: NORTH AFRICA WIND
# ══════════════════════════════════════════════════════════════════════════════
def render_nafrica_wind():
    st.markdown("#### NORTH AFRICA — HIGH WIND SPEEDS")
    st.caption("Tangier → Algeria → Port Said shipping corridor. "
               "Sustained high winds disrupt LPG loading and tanker scheduling.")

    try:
        wind_df = load_nafrica_wind()
    except Exception as e:
        st.error(f"Failed to load North Africa wind data: {e}")
        return

    if wind_df.empty:
        st.info("No wind data available.")
        return

    # Wind warning alerts
    fcst = wind_df[wind_df["source"] == "forecast"]
    gale_events = fcst[fcst["wind_speed_ms"] >= WIND_GALE_MS]
    if not gale_events.empty:
        cities_gale = gale_events["city"].unique().tolist()
        st.markdown(
            f'<div class="wind-warning">GALE WARNING: {", ".join(cities_gale)} '
            f'forecast >=20 m/s in next 14 days</div>',
            unsafe_allow_html=True,
        )

    selected_cities = st.multiselect(
        "Ports", list(NAFRICA_WIND_CORRIDOR.keys()), list(NAFRICA_WIND_CORRIDOR.keys()),
        key="nafrica_cities", label_visibility="collapsed",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = make_wind_corridor_chart(wind_df, selected_cities)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = make_wind_exceedance_chart(wind_df)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.title("⛽ LPG Desk — Weather Intelligence")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🇺🇸 US HDD",
    "🇯🇵 Asia HDD",
    "🇮🇳 India Precip",
    "🇵🇦 Gatun Lake",
    "🏭 PADD 3 Coastal",
    "🌀 Hurricanes",
    "💨 N.Africa Wind",
])

with tab1:
    render_us_hdd()
with tab2:
    render_asia_hdd()
with tab3:
    render_india_precip()
with tab4:
    render_gatun_lake()
with tab5:
    render_padd3_coastal()
with tab6:
    render_hurricanes()
with tab7:
    render_nafrica_wind()
