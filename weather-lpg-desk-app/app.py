"""LPG Desk Weather Dashboard — Trader Edition

Tabs:
  1. US HDD by PADD        — Cumulative HDD + run comparison
  2. Asia HDD              — Japan, South Korea, China (N/E/S)
  3. India Precipitation   — Anomaly vs climatology
  4. Gatun Lake            — Observed level + official forecast + El Nino analogs
  5. PADD 3 Coastal Temps  — Export terminal freeze risk
  6. Hurricanes            — US Gulf/Atlantic + NW Australia cyclones
  7. North Africa Wind     — Tangier-Algeria-Port Said corridor
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _config import (
    PADD_REGIONS,
    ASIA_REGION_MAP, DEFAULT_ASIA_REGIONS,
    INDIA_SUBREGIONS,
    PADD3_COASTAL_CITIES, DEFAULT_PADD3_CITIES,
    FREEZE_THRESHOLD_C,
    NAFRICA_WIND_CORRIDOR, DEFAULT_NAFRICA_PORTS, WIND_GALE_MS,
    HDD_BASE_TEMP,
    GATUN_ANALOG_YEARS, GATUN_RESTRICTION_FT, GATUN_RATE_SURCHARGE_FT,
    HURRICANE_BASINS,
)
from _data import (
    load_hdd_season, load_hdd_normal_daily, load_hdd_run_comparison,
    load_asia_hdd_season, load_asia_hdd_normal,
    load_india_precip_anomaly,
    load_gatun_lake_levels, build_gatun_analog_overlays,
    load_padd3_coastal_temps,
    load_hurricane_data,
    load_nafrica_wind,
)
from _charts import (
    make_hdd_season_chart, make_hdd_comparison_chart,
    make_asia_hdd_season_chart, make_asia_hdd_bar_chart,
    make_india_precip_timeseries, make_india_precip_deviation_chart,
    make_gatun_lake_chart,
    make_padd3_temp_chart,
    make_hurricane_map,
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


# ─── KPI / status helpers ────────────────────────────────────────────────────────
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


def raw_kpi_card(label: str, value_str: str, card_class: str = "kpi-card-neutral") -> str:
    """KPI card for values that aren't a signed float (status text, counts)."""
    return f'<div class="kpi-card {card_class}"><div class="kpi-label">{label}</div><div class="kpi-value">{value_str}</div></div>'


def status_banner(icon: str, text: str, level: str = "warning") -> str:
    """Alert banner — icon + label, never color alone (status tokens are reserved)."""
    return f'<div class="status-banner status-banner-{level}">{icon} {text}</div>'


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: US HDD BY PADD
# ══════════════════════════════════════════════════════════════════════════════
def render_us_hdd():
    st.markdown("#### US HEATING DEGREE DAYS BY PADD")
    st.caption(f"Base {HDD_BASE_TEMP:.0f}°C. Season Oct 1 – Mar 31. Source: ERA5 + EC-ens forecast.")

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

    cols = st.columns(min(len(selected_padds), 3))
    for i, padd in enumerate(selected_padds):
        with cols[i % len(cols)]:
            fig = make_hdd_season_chart(season_df, padd, normal_df)
            st.plotly_chart(fig, use_container_width=True)

    if not run_df.empty:
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
        normal_df = load_asia_hdd_normal()
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

    fig = make_asia_hdd_season_chart(season_df, selected, normal_df)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = make_asia_hdd_bar_chart(season_df, selected)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: INDIA PRECIPITATION ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
def render_india_precip():
    st.markdown("#### INDIA PRECIPITATION ANOMALY")
    st.caption("Precipitation vs 2000–2024 ERA5 day-of-year climatology, actual + 14-day forecast.")

    try:
        anom_df = load_india_precip_anomaly()
    except Exception as e:
        st.error(f"Failed to load India precipitation data: {e}")
        return

    if anom_df.empty:
        st.info("No India precipitation data available.")
        return

    kpi_cols = st.columns(len(INDIA_SUBREGIONS))
    latest = anom_df.sort_values("delivery_date").groupby("subregion").last().reset_index()
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
    st.markdown("#### GATUN LAKE — PANAMA CANAL")
    st.caption("Observed water level + official ACP forecast, overlaid with the strongest historical "
               f"El Nino years ({', '.join(str(y) for y in GATUN_ANALOG_YEARS)}), time-shifted onto today's "
               "calendar. Source: Panama Canal Authority (live).")

    try:
        hist_df, proj_df = load_gatun_lake_levels()
    except Exception as e:
        st.error(f"Gatun Lake data unavailable: {e}")
        return

    if hist_df.empty:
        st.warning("No Gatun Lake level data — the ACP endpoint may be unreachable.")
        return

    current_level = float(hist_df.sort_values("date")["level_ft"].iloc[-1])
    if current_level < GATUN_RESTRICTION_FT:
        status, card_class = "Critical — Major Restrictions", "kpi-card-critical"
    elif current_level < GATUN_RATE_SURCHARGE_FT:
        status, card_class = "Caution — Rate Surcharges", "kpi-card-warning"
    else:
        status, card_class = "Normal Operations", "kpi-card-good"

    kc1, kc2 = st.columns([1, 3])
    with kc1:
        st.markdown(raw_kpi_card("Gatun Lake Level", f"{current_level:.1f} ft", card_class), unsafe_allow_html=True)
    with kc2:
        st.markdown(raw_kpi_card("Status", status, card_class), unsafe_allow_html=True)

    analog_overlays = build_gatun_analog_overlays(hist_df, proj_df)
    fig = make_gatun_lake_chart(hist_df, proj_df, analog_overlays)
    st.plotly_chart(fig, use_container_width=True)


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

    fcst = temp_df[temp_df["source"] == "forecast"]
    freeze_days = fcst[fcst["temperature_c"] <= FREEZE_THRESHOLD_C]
    if not freeze_days.empty:
        cities_at_risk = freeze_days["city"].unique().tolist()
        st.markdown(
            status_banner("❄", f"FREEZE ALERT: {len(cities_at_risk)} terminal(s) forecast below "
                                f"{FREEZE_THRESHOLD_C:.0f}°C — {', '.join(cities_at_risk)}", "critical"),
            unsafe_allow_html=True,
        )

    selected_cities = st.multiselect(
        "Terminals", list(PADD3_COASTAL_CITIES.keys()), DEFAULT_PADD3_CITIES,
        key="padd3_cities", label_visibility="collapsed",
    )
    if not selected_cities:
        st.info("Select at least one terminal.")
        return

    fig = make_padd3_temp_chart(temp_df, selected_cities)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: HURRICANES
# ══════════════════════════════════════════════════════════════════════════════
def render_hurricanes():
    st.markdown("#### TROPICAL CYCLONES — US GULF/ATLANTIC & NW AUSTRALIA")
    st.caption("Live active-storm tracking (NHC + IBTrACS). LPG export/loading disruption risk.")

    with st.spinner("Fetching live storm data…"):
        try:
            storms, sources = load_hurricane_data()
        except Exception as e:
            st.error(f"Hurricane data unavailable: {e}")
            return

    src_caption = "  ·  ".join(f"**{k}**: {v}" for k, v in sources.items())
    st.caption(f"Sources — {src_caption}")

    map_cols = st.columns(2)
    for col, (region_name, bbox) in zip(map_cols, HURRICANE_BASINS.items()):
        with col:
            fig = make_hurricane_map(storms, region_name, bbox)
            st.plotly_chart(fig, use_container_width=True)

    if not storms:
        st.markdown(status_banner("✓", "No active tropical cyclones in the US Gulf/Atlantic or "
                                        "NW Australia region at this time.", "good"), unsafe_allow_html=True)
        return

    max_wind = max(s["wind_kt"] for s in storms)
    if max_wind >= 64:
        st.markdown(status_banner("🌀", f"{len(storms)} active storm(s) tracked — strongest at "
                                        f"{max_wind} kt.", "critical"), unsafe_allow_html=True)
    else:
        st.markdown(status_banner("🌀", f"{len(storms)} active storm(s) tracked — strongest at "
                                        f"{max_wind} kt.", "warning"), unsafe_allow_html=True)

    st.markdown("##### ACTIVE STORMS")
    rows = [{
        "Name": s["name"], "Region": s["basin"], "Category": s["category"],
        "Wind (kt)": s["wind_kt"], "Pressure (mb)": s.get("pressure_mb", "N/A"),
        "Lat": f"{s['lat']:.1f}°{'N' if s['lat'] >= 0 else 'S'}",
        "Lon": f"{abs(s['lon']):.1f}°{'E' if s['lon'] >= 0 else 'W'}",
        "Last update": s.get("last_update", "N/A"),
    } for s in storms]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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

    fcst = wind_df[wind_df["source"] == "forecast"]
    gale_events = fcst[fcst["wind_speed_ms"] >= WIND_GALE_MS]
    if not gale_events.empty:
        cities_gale = gale_events["city"].unique().tolist()
        st.markdown(
            status_banner("💨", f"GALE WARNING: {', '.join(cities_gale)} forecast "
                                f"≥{WIND_GALE_MS:.0f} m/s in the next 14 days", "critical"),
            unsafe_allow_html=True,
        )

    selected_cities = st.multiselect(
        "Ports", list(NAFRICA_WIND_CORRIDOR.keys()), DEFAULT_NAFRICA_PORTS,
        key="nafrica_cities", label_visibility="collapsed",
    )
    if not selected_cities:
        st.info("Select at least one port.")
        return

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
