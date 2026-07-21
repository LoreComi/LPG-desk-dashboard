"""LPG Desk Weather Dashboard — Trader Edition

Sections (picked from the sidebar):
  1. US HDD by PADD        — Cumulative HDD + run comparison
  2. Asia HDD              — Japan, South Korea, China (N/E/S)
  3. India Precipitation   — Anomaly vs climatology
  4. Gatun Lake            — Observed level + official forecast + El Nino analogs
  5. PADD 3 Coastal Temps  — Export terminal freeze risk
  6. Hurricanes            — US Gulf/Atlantic + NW Australia cyclones
  7. North Africa Wind     — Tangier-Algeria-Port Said corridor

A separate "Generate AI Brief" button (not a section) sits below the nav —
one click both switches to that view and runs the 6-agent analysis.
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
    is_in_hdd_season, hdd_season_label, hdd_season_start_year,
)
from _data import (
    load_hdd_season, load_hdd_normal_daily, load_hdd_run_comparison, load_hdd_season_history,
    load_asia_hdd_season, load_asia_hdd_normal, load_asia_hdd_season_history,
    compute_season_kpis,
    load_india_precip_anomaly,
    load_gatun_lake_levels, build_gatun_analog_overlays,
    load_padd3_coastal_temps,
    load_hurricane_data,
    load_nafrica_wind,
)
from _ai_lpg_brief import generate_lpg_brief, brief_to_html_bullets
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
    layout="wide",
    initial_sidebar_state="expanded",
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


_STATUS_TAGS = {"critical": "Alert", "warning": "Notice", "good": "OK"}


def status_banner(text: str, level: str = "warning") -> str:
    """Alert banner — bold text tag + label, never color alone (status tokens are reserved)."""
    tag = _STATUS_TAGS.get(level, "Notice")
    return f'<div class="status-banner status-banner-{level}"><span class="status-tag">{tag}</span>{text}</div>'


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def hdd_kpi_card(label: str, kpis: dict | None) -> str:
    """Season-to-date HDD KPI card — cumulative, vs-normal delta, coldest/warmest rank.

    An HDD delta's sign is the opposite of a raw temperature anomaly: MORE
    HDD than normal means COLDER, so a positive delta gets the cool/blue
    accent here (unlike kpi_card(), built for values where positive == warmer).
    """
    if kpis is None:
        return raw_kpi_card(label, "N/A", "kpi-card-neutral")

    value_str = f"{kpis['cumulative_to_date']:.0f} HDD"
    card_class = "kpi-card-neutral"
    delta_html = ""
    delta = kpis.get("delta_vs_normal")
    if delta is not None and not np.isnan(delta):
        card_class = "kpi-card-cool" if delta > 0 else ("kpi-card-warm" if delta < 0 else "kpi-card-neutral")
        d_cls = "kpi-delta-up" if delta > 0 else ("kpi-delta-down" if delta < 0 else "kpi-delta-flat")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "●")
        word = "colder" if delta > 0 else ("milder" if delta < 0 else "on normal")
        delta_html = f'<div class="kpi-delta {d_cls}">{arrow} {abs(delta):.0f} HDD {word}</div>'

    rank_html = ""
    rank, pool = kpis.get("rank"), kpis.get("rank_pool_size")
    if rank and pool:
        rank_html = f'<div class="kpi-rank">{_ordinal(rank)} coldest of last {pool} yrs</div>'

    return (f'<div class="kpi-card {card_class}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value_str}</div>{delta_html}{rank_html}</div>')


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
        history_df = load_hdd_season_history()
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

    padd_kpis = {padd: compute_season_kpis(season_df, normal_df, history_df, "padd_region", padd)
                 for padd in selected_padds}

    if not is_in_hdd_season():
        as_of = next((k["as_of"] for k in padd_kpis.values() if k), None)
        label = hdd_season_label(hdd_season_start_year(as_of)) if as_of is not None else "last winter"
        st.markdown(
            status_banner(f"Not in HDD seasonality — showing final values for {label} below.", "warning"),
            unsafe_allow_html=True,
        )

    kpi_cols = st.columns(len(selected_padds))
    for i, padd in enumerate(selected_padds):
        with kpi_cols[i % len(kpi_cols)]:
            st.markdown(hdd_kpi_card(padd, padd_kpis[padd]), unsafe_allow_html=True)

    st.divider()

    cols = st.columns(min(len(selected_padds), 3))
    for i, padd in enumerate(selected_padds):
        with cols[i % len(cols)]:
            fig = make_hdd_season_chart(season_df, padd, normal_df, history_df)
            st.plotly_chart(fig, use_container_width=True)

    if not run_df.empty:
        st.divider()
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
        history_df = load_asia_hdd_season_history()
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

    region_kpis = {region: compute_season_kpis(season_df, normal_df, history_df, "region", region)
                   for region in selected}

    if not is_in_hdd_season():
        as_of = next((k["as_of"] for k in region_kpis.values() if k), None)
        label = hdd_season_label(hdd_season_start_year(as_of)) if as_of is not None else "last winter"
        st.markdown(
            status_banner(f"Not in HDD seasonality — showing final values for {label} below.", "warning"),
            unsafe_allow_html=True,
        )

    kpi_cols = st.columns(len(selected))
    for i, region in enumerate(selected):
        with kpi_cols[i % len(kpi_cols)]:
            st.markdown(hdd_kpi_card(region, region_kpis[region]), unsafe_allow_html=True)

    st.divider()

    fig = make_asia_hdd_season_chart(season_df, selected, normal_df, history_df)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = make_asia_hdd_bar_chart(season_df, selected)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: INDIA PRECIPITATION ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
def render_india_precip():
    st.markdown("#### INDIA PRECIPITATION ANOMALY")
    st.caption("Precipitation vs ERA5 day-of-year climatology, actual + 14-day forecast.")

    try:
        anom_df = load_india_precip_anomaly()
    except Exception as e:
        st.error(f"Failed to load India precipitation data: {e}")
        return

    if anom_df.empty:
        st.info("No India precipitation data available.")
        return

    # KPI cards: show latest actual precip (mm) or deviation % if available
    kpi_cols = st.columns(len(INDIA_SUBREGIONS))
    latest = anom_df[anom_df["precip_actual_mm"].notna()].sort_values("delivery_date").groupby("subregion").last().reset_index()
    for i, (_, row) in enumerate(latest.iterrows()):
        with kpi_cols[i % len(kpi_cols)]:
            dev = row.get("deviation_pct")
            if dev is not None and not np.isnan(dev):
                st.markdown(kpi_card(row["subregion"], dev, "%"), unsafe_allow_html=True)
            else:
                # Show raw precip when no climatology baseline available
                val = row.get("precip_actual_mm", 0)
                st.markdown(raw_kpi_card(row["subregion"], f"{val:.1f} mm/d", "kpi-card-neutral"), unsafe_allow_html=True)

    # Note when deviation is unavailable
    if anom_df["deviation_pct"].isna().all():
        st.markdown(
            status_banner("Anomaly % unavailable — climatology baseline still building (< 1 year of ERA5 history).", "warning"),
            unsafe_allow_html=True,
        )

    st.divider()

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

    st.divider()

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
            status_banner(f"{len(cities_at_risk)} terminal(s) forecast below "
                          f"{FREEZE_THRESHOLD_C:.0f}°C: {', '.join(cities_at_risk)}", "critical"),
            unsafe_allow_html=True,
        )

    selected_cities = st.multiselect(
        "Terminals", list(PADD3_COASTAL_CITIES.keys()), DEFAULT_PADD3_CITIES,
        key="padd3_cities", label_visibility="collapsed",
    )
    if not selected_cities:
        st.info("Select at least one terminal.")
        return

    st.divider()

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
        st.markdown(status_banner("No active tropical cyclones in the US Gulf/Atlantic or "
                                   "NW Australia region at this time.", "good"), unsafe_allow_html=True)
        return

    max_wind = max(s["wind_kt"] for s in storms)
    if max_wind >= 64:
        st.markdown(status_banner(f"{len(storms)} active storm(s) tracked — strongest at "
                                   f"{max_wind} kt.", "critical"), unsafe_allow_html=True)
    else:
        st.markdown(status_banner(f"{len(storms)} active storm(s) tracked — strongest at "
                                   f"{max_wind} kt.", "warning"), unsafe_allow_html=True)

    st.divider()
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
    # Flag if ≥10% of ensemble members (5/50) forecast gale-force winds
    gale_events = fcst[fcst["pct_above_gale"] >= 10.0]
    if not gale_events.empty:
        cities_gale = gale_events["city"].unique().tolist()
        max_pct = int(gale_events.groupby("city")["pct_above_gale"].max().max())
        st.markdown(
            status_banner(f"{', '.join(cities_gale)} — up to {max_pct}% of members "
                          f"forecast ≥{WIND_GALE_MS:.0f} m/s in the next 14 days", "critical"),
            unsafe_allow_html=True,
        )

    selected_cities = st.multiselect(
        "Ports", list(NAFRICA_WIND_CORRIDOR.keys()), DEFAULT_NAFRICA_PORTS,
        key="nafrica_cities", label_visibility="collapsed",
    )
    if not selected_cities:
        st.info("Select at least one port.")
        return

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = make_wind_corridor_chart(wind_df, selected_cities)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = make_wind_exceedance_chart(wind_df)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0: AI LPG MARKET BRIEF
# ══════════════════════════════════════════════════════════════════════════════

_BRIEF_CSS = """
<style>
.brief-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin: 12px 0;
}
.brief-bullet {
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.92rem;
    line-height: 1.5;
    color: var(--text-primary);
}
.brief-bullet:last-child { border-bottom: none; }
.brief-bullet b { color: var(--accent); }
.signal-card {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.signal-bullish { background: #e9f7e9; color: #0f5c0f; border: 1px solid rgba(12,163,12,0.30); }
.signal-bearish { background: #fbeaea; color: #7a1f1f; border: 1px solid rgba(194,47,47,0.30); }
.signal-neutral { background: rgba(20,20,15,0.05); color: var(--text-secondary); border: 1px solid var(--border); }
</style>
"""


def _get_az_credentials() -> tuple:
    """Retrieve Azure OpenAI credentials — same pattern as Coal Desk app."""
    import os

    # 1. Try environment variables (app.yaml valueFrom)
    tenant = os.environ.get("AZURE_TENANT_ID", "")
    client = os.environ.get("AZURE_CLIENT_ID", "")
    secret = os.environ.get("AZURE_CLIENT_SECRET", "")
    if all([tenant, client, secret]):
        return tenant, client, secret

    # 2. Databricks SDK dbutils (works in Apps runtime context)
    try:
        from databricks.sdk.runtime import dbutils as sdk_dbutils
        tenant = sdk_dbutils.secrets.get("commodity-news-secrets", "azure-tenant-id")
        client = sdk_dbutils.secrets.get("commodity-news-secrets", "azure-client-id")
        secret = sdk_dbutils.secrets.get("commodity-news-secrets", "azure-client-secret")
        if all([tenant, client, secret]):
            return tenant, client, secret
    except Exception:
        pass

    # 3. WorkspaceClient secrets API
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        tenant = w.dbutils.secrets.get(scope="commodity-news-secrets", key="azure-tenant-id")
        client = w.dbutils.secrets.get(scope="commodity-news-secrets", key="azure-client-id")
        secret = w.dbutils.secrets.get(scope="commodity-news-secrets", key="azure-client-secret")
        if all([tenant, client, secret]):
            return tenant, client, secret
    except Exception:
        pass

    return "", "", ""


def _render_signal_pill(label: str, signal: str) -> str:
    cls = f"signal-{signal.lower()}"
    arrow = "▲" if signal == "BULLISH" else ("▼" if signal == "BEARISH" else "●")
    return f'<span class="signal-card {cls}">{arrow} {label}: {signal}</span>'


def render_lpg_brief(auto_generate: bool = False):
    """AI LPG market brief. `auto_generate=True` runs the analysis immediately
    (used when this page is reached via the sidebar's "Generate AI Brief"
    button, so that one click both navigates here and starts the run —
    no separate in-page Generate click needed for the first run)."""
    st.markdown("#### AI LPG MARKET BRIEF")
    st.caption(
        "6-agent AI analysis: storm supply risk · US propane demand · "
        "Asia LPG demand · India monsoon logistics · Panama Canal transit → LPG market outlook"
    )
    st.markdown(_BRIEF_CSS, unsafe_allow_html=True)

    tenant_id, client_id, client_secret = _get_az_credentials()
    if not all([tenant_id, client_id, client_secret]):
        st.warning(
            "Azure OpenAI credentials not found. "
            "Configure `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` "
            "in environment variables or Databricks secrets (scope: **commodity-news-secrets**)."
        )
        return

    cached = st.session_state.get("lpg_brief_result")
    regenerate = False
    if cached:
        col_txt, col_btn = st.columns([5, 1])
        with col_btn:
            regenerate = st.button("Regenerate", key="lpg_brief_btn")

    generate = auto_generate or regenerate

    if not generate and not cached:
        generate = st.button("Generate AI Brief", type="primary", key="lpg_brief_btn_inline")
        if not generate:
            st.caption("Runs 6 Azure OpenAI calls (~20-30 seconds).")
            with st.expander("What each agent analyses"):
                st.markdown("""
| Agent | Data source | Market signal |
|---|---|---|
| Hurricane | NHC + JTWC live storms | PADD 3 terminal disruption → Mont Belvieu |
| US HDD | ERA5 + EC-ens by PADD | Propane heating demand → stock draws |
| Asia HDD | Population-weighted Japan/Korea/China | NE Asia LPG imports → FEI |
| India Monsoon | ERA5 + EC-ens precipitation | Port congestion → India CFR premium |
| Panama Canal | Gatun Lake live level | VLGC transit constraints → freight/ARB |
| Synthesis | All 5 agents combined | Net LPG market direction |
""")
            return

    if generate:
        progress = st.empty()

        def _prog(msg: str):
            progress.caption(msg)

        try:
            _prog("Loading storm data…")
            storms_result = load_hurricane_data()
            storms, _ = storms_result if isinstance(storms_result, tuple) else (storms_result, {})

            _prog("Computing US HDD KPIs…")
            padd_kpis = {}
            try:
                season_df = load_hdd_season()
                normal_df = load_hdd_normal_daily()
                history_df = load_hdd_season_history()
                for padd in PADD_REGIONS:
                    padd_kpis[padd] = compute_season_kpis(
                        season_df, normal_df, history_df, "padd_region", padd)
            except Exception:
                pass

            _prog("Computing Asia HDD KPIs…")
            asia_kpis = {}
            try:
                asia_season = load_asia_hdd_season()
                asia_normal = load_asia_hdd_normal()
                asia_history = load_asia_hdd_season_history()
                for region in ASIA_REGION_MAP:
                    asia_kpis[region] = compute_season_kpis(
                        asia_season, asia_normal, asia_history, "region", region)
            except Exception:
                pass

            _prog("Loading India precipitation…")
            india_summary = {}
            try:
                anom_df = load_india_precip_anomaly()
                if not anom_df.empty:
                    actuals = anom_df[anom_df["precip_actual_mm"].notna()]
                    for sub in actuals["subregion"].unique():
                        sdf = actuals[actuals["subregion"] == sub]
                        india_summary[sub] = {
                            "avg_actual_mm": float(sdf["precip_actual_mm"].mean()),
                            "avg_clim_mm": float(sdf["precip_clim_mm"].mean()) if sdf["precip_clim_mm"].notna().any() else None,
                            "avg_deviation_pct": float(sdf["deviation_pct"].mean()) if sdf["deviation_pct"].notna().any() else None,
                        }
            except Exception:
                pass

            _prog("Loading Gatun Lake level…")
            gatun_level = None
            gatun_trend = ""
            try:
                hist_df, _ = load_gatun_lake_levels()
                if not hist_df.empty:
                    recent = hist_df.sort_values("date").tail(14)
                    gatun_level = float(recent["level_ft"].iloc[-1])
                    if len(recent) >= 7:
                        week_ago = float(recent["level_ft"].iloc[-7])
                        delta = gatun_level - week_ago
                        gatun_trend = f"{'rising' if delta > 0 else 'falling'} {abs(delta):.1f} ft over last 7 days"
            except Exception:
                pass

            result = generate_lpg_brief(
                storms=storms,
                padd_kpis=padd_kpis,
                asia_kpis=asia_kpis,
                india_precip_summary=india_summary,
                gatun_level_ft=gatun_level,
                gatun_trend=gatun_trend,
                azure_tenant_id=tenant_id,
                azure_client_id=client_id,
                azure_client_secret=client_secret,
                progress_cb=_prog,
            )
            st.session_state["lpg_brief_result"] = result
            cached = result

        except Exception as exc:
            progress.empty()
            st.error(f"Brief generation failed: {exc}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return

        progress.empty()

    if not cached:
        return

    # Agent signal pills
    signals = cached.get("signals", {})
    pills = " &nbsp; ".join([
        _render_signal_pill("Storms", signals.get("hurricane", "NEUTRAL")),
        _render_signal_pill("US HDD", signals.get("us_hdd", "NEUTRAL")),
        _render_signal_pill("Asia HDD", signals.get("asia_hdd", "NEUTRAL")),
        _render_signal_pill("India", signals.get("india_precip", "NEUTRAL")),
        _render_signal_pill("Panama", signals.get("panama_canal", "NEUTRAL")),
    ])
    overall = signals.get("synthesis", "NEUTRAL")
    overall_pill = _render_signal_pill("NET", overall)
    st.markdown(f"{pills} &nbsp;&nbsp; {overall_pill}", unsafe_allow_html=True)

    st.markdown("---")

    # Synthesis brief
    gatun_str = (f" · Gatun {cached['gatun_level_ft']:.1f} ft"
                 if cached.get("gatun_level_ft") else "")
    st.markdown(f"**LPG Market Brief**{gatun_str} · {cached['generated_at']}")
    st.markdown(
        f'<div class="brief-box">{brief_to_html_bullets(cached["synthesis"])}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### SPECIALIST AGENT OUTPUTS")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Hurricane / terminal supply risk"):
            st.markdown(cached["hurricane"])
        with st.expander("US HDD / propane heating demand"):
            st.markdown(cached["us_hdd"])
        with st.expander("India monsoon / LPG logistics"):
            st.markdown(cached["india_precip"])
    with c2:
        with st.expander("Asia HDD / LPG import demand"):
            st.markdown(cached["asia_hdd"])
        gatun_exp = (f"Panama Canal — {cached['gatun_level_ft']:.1f} ft"
                     if cached.get("gatun_level_ft") else "Panama Canal")
        with st.expander(gatun_exp):
            st.markdown(cached["panama_canal"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
from datetime import datetime, timezone

_now_utc = datetime.now(timezone.utc)
_refresh_hour = _now_utc.strftime("%H:%M UTC")
_refresh_date = _now_utc.strftime("%d %b %Y")

# Branded header
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
    <div>
        <span style="font-size:1.7rem; font-weight:800; letter-spacing:-0.03em; color:var(--text-primary);">
            LPG Desk</span>
        <span style="font-size:1.7rem; font-weight:300; letter-spacing:-0.03em; color:var(--text-muted); margin-left:8px;">
            Weather Intelligence</span>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
            background:#e9f7e9; border:1px solid rgba(12,163,12,0.28);
            border-radius:999px; font-size:0.72rem; font-weight:600; color:#0f5c0f;">
            <span style="width:6px;height:6px;background:#0ca30c;border-radius:50%;display:inline-block;"></span>
            Live
        </span>
        <span style="font-size:0.72rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">
            {_refresh_date} &middot; {_refresh_hour}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

PAGES = {
    "US HDD": render_us_hdd,
    "Asia HDD": render_asia_hdd,
    "India Precipitation": render_india_precip,
    "Gatun Lake": render_gatun_lake,
    "PADD 3 Coastal": render_padd3_coastal,
    "Hurricanes": render_hurricanes,
    "N. Africa Wind": render_nafrica_wind,
}

with st.sidebar:
    st.markdown("### Sections")
    selected_page = st.radio(
        "Sections", list(PAGES.keys()), label_visibility="collapsed", key="nav_page",
    )
    st.divider()
    # A separate, distinct action (not a section to browse) — one click both
    # switches to this view and runs the analysis, instead of navigate-then-Generate.
    generate_clicked = st.button(
        "Generate AI Brief", type="primary", use_container_width=True, key="ai_brief_sidebar_btn",
    )

if generate_clicked:
    st.session_state["show_ai_brief"] = True
elif st.session_state.get("_last_nav_page") != selected_page:
    st.session_state["show_ai_brief"] = False
st.session_state["_last_nav_page"] = selected_page

if st.session_state.get("show_ai_brief"):
    render_lpg_brief(auto_generate=generate_clicked)
else:
    PAGES[selected_page]()
