"""LPG Desk Weather Dashboard — Trader Edition

Sections (picked from the sidebar):
  1. US HDD by PADD        — Cumulative HDD + run comparison
  2. Asia HDD              — Japan, South Korea, China (N/E/S)
  3. India Precipitation   — Anomaly vs climatology
  4. Panama Canal          — Gatun level, El Nino / ENSO forecasts, watershed rain
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
    HDD_BASE_TEMP, GAS_DEMAND_WEIGHTS, US_WEIGHTED_LABEL, gas_weight_season,
    GATUN_ANALOG_YEARS, GATUN_RESTRICTION_FT, GATUN_RATE_SURCHARGE_FT,
    GATUN_FORECAST_METHODS, GATUN_COMPOSITE_YEARS, GATUN_FALLBACK_ANALOG_YEARS,
    GATUN_ENSO_HORIZON_DAYS, GATUN_ENSO_INDEX_BAND, GATUN_ENSO_DECAY_MONTHS,
    PANAMA_PRECIP_LOOKBACK_DAYS, PANAMA_PRECIP_FORECAST_DAYS,
    PANAMA_PRECIP_NEAR_NORMAL_PCT,
    HURRICANE_BASINS,
    is_in_hdd_season, hdd_season_label, hdd_season_start_year,
)
from _data import (
    load_hdd_season, load_hdd_normal_daily, load_hdd_run_comparison, load_hdd_season_history,
    load_asia_hdd_season, load_asia_hdd_normal, load_asia_hdd_season_history, load_asia_hdd_run_comparison,
    compute_season_kpis, build_weighted_us_frames, run_window_totals, run_over_run_change,
    load_india_precip_anomaly,
    load_gatun_lake_levels, build_gatun_analog_overlays, build_gatun_analog_extension,
    build_gatun_composite_forecast, build_gatun_enso_index_forecast,
    load_oni_monthly, oni_to_daily, list_el_nino_onset_years,
    load_panama_precip_anomaly, summarize_panama_precip,
    load_padd3_coastal_temps,
    load_hurricane_data,
    load_nafrica_wind,
)
from _ai_lpg_brief import generate_lpg_brief, brief_to_html_bullets
from _charts import (
    make_hdd_season_chart, make_hdd_run_totals_chart, make_hdd_run_change_chart,
    make_asia_hdd_season_chart, make_asia_hdd_bar_chart,
    make_asia_hdd_region_chart, make_asia_hdd_comparison_chart,

    make_india_precip_timeseries, make_india_precip_deviation_chart,
    make_gatun_lake_chart, make_gatun_enso_index_chart, make_gatun_composite_chart,
    make_panama_precip_chart, make_panama_precip_departure_chart,
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


def forecast_kpi_card(label: str, fcst_hdd, avg_hdd, small: bool = False) -> str:
    """14-day forecast KPI card — styled identically to the season KPI cards."""
    if fcst_hdd is None:
        return raw_kpi_card(label, "N/A", "kpi-card-neutral")
    value_str = f"{fcst_hdd:.0f} HDD"
    if avg_hdd is None:
        return raw_kpi_card(label, value_str, "kpi-card-neutral")
    delta = fcst_hdd - avg_hdd
    card_class = "kpi-card-cool" if delta > 0 else ("kpi-card-warm" if delta < 0 else "kpi-card-neutral")
    d_cls = "kpi-delta-up" if delta > 0 else ("kpi-delta-down" if delta < 0 else "kpi-delta-flat")
    arrow = "\u25b2" if delta > 0 else ("\u25bc" if delta < 0 else "\u25cf")
    word = "colder" if delta > 0 else ("milder" if delta < 0 else "on avg")
    delta_html = f'<div class="kpi-delta {d_cls}">{arrow} {abs(delta):.0f} {word} vs {avg_hdd:.0f} (10-yr avg)</div>'
    size_class = " kpi-card-sm" if small else ""
    return (f'<div class="kpi-card {card_class}{size_class}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value_str}</div>{delta_html}</div>')


def _forecast_vs_10yr(season_df, history_df, group_col, group_value, n_years=10):
    """(forecast_14d_hdd, ten_yr_avg_hdd) for the same day-of-year window."""
    sdf = season_df[season_df[group_col] == group_value] if group_col in season_df.columns else season_df
    fcst = sdf[~sdf["model_label"].str.contains("Actual", na=False)]
    if fcst.empty:
        return None, None
    fcst_total = float(fcst["daily_hdd"].sum())
    fcst_doys = set(pd.to_datetime(fcst["delivery_date"]).dt.dayofyear)
    hdf = history_df[history_df[group_col] == group_value] if group_col in history_df.columns else history_df
    if hdf.empty:
        return fcst_total, None
    years = sorted(hdf["season_start_year"].dropna().unique())[-n_years:]
    yr_totals = []
    for yr in years:
        ydf = hdf[(hdf["season_start_year"] == yr) & (hdf["day_of_year"].isin(fcst_doys))]
        if not ydf.empty:
            yr_totals.append(float(ydf["daily_hdd"].sum()))
    avg = sum(yr_totals) / len(yr_totals) if yr_totals else None
    return fcst_total, avg


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def hdd_kpi_card(label: str, kpis: dict | None, small: bool = False) -> str:
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
        # rank counts this season plus the `pool` history years (1 = coldest of
        # all of them, up to pool+1 = mildest of all of them). Whichever end
        # it's closer to reads more naturally — e.g. "1st mildest" instead of
        # the technically-equivalent but confusing "(pool+1)th coldest".
        total = pool + 1
        if rank <= total - rank + 1:
            rank_html = f'<div class="kpi-rank">{_ordinal(rank)} coldest of last {pool} yrs</div>'
        else:
            mild_rank = total - rank + 1
            rank_html = f'<div class="kpi-rank">{_ordinal(mild_rank)} mildest of last {pool} yrs</div>'

    size_class = " kpi-card-sm" if small else ""
    return (f'<div class="kpi-card {card_class}{size_class}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value_str}</div>{delta_html}{rank_html}</div>')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: US HDD BY PADD
# ══════════════════════════════════════════════════════════════════════════════
def _off_season_banner(as_of) -> None:
    """Warn that the numbers below are last winter's final values."""
    if is_in_hdd_season():
        return
    label = hdd_season_label(hdd_season_start_year(as_of)) if as_of is not None else "last winter"
    st.markdown(
        status_banner(f"Not in HDD seasonality — showing final values for {label} below.", "warning"),
        unsafe_allow_html=True,
    )


def render_us_hdd(view: str = "Forecast"):
    """US HDD page. view: 'Forecast' (14-day vs 10-yr avg + run comparison)
    or 'Season so far' (cumulative-to-date vs normal + season charts)."""
    st.markdown(f"#### US HEATING DEGREE DAYS — {view.upper()}")
    st.caption(f"Base {HDD_BASE_TEMP:.0f}°C. Season Oct 1 – Mar 31. Source: ERA5 + EC-ens forecast.")

    try:
        season_df = load_hdd_season()
        normal_df = load_hdd_normal_daily()
        run_df = load_hdd_run_comparison()
        history_df = load_hdd_season_history()
    except Exception as e:
        st.error(f"Failed to load US HDD data: {e}")
        return

    # L48 total: always all five PADDs, whatever the PADD picker below shows.
    us = build_weighted_us_frames(season_df, normal_df, history_df, run_df)
    us_kpis = compute_season_kpis(us["season"], us["normal"], us["history"],
                                  "padd_region", US_WEIGHTED_LABEL)
    _off_season_banner(us_kpis["as_of"] if us_kpis else None)

    # Headline L48 box, then the PADDs smaller underneath.
    _, box_col, _ = st.columns([1, 2, 1])
    with box_col:
        if view == "Forecast":
            fhdd, ahdd = _forecast_vs_10yr(us["season"], us["history"], "padd_region", US_WEIGHTED_LABEL)
            card = forecast_kpi_card(f"{US_WEIGHTED_LABEL} · 14-day", fhdd, ahdd)
        else:
            card = hdd_kpi_card(f"{US_WEIGHTED_LABEL} · season so far", us_kpis)
        st.markdown(card, unsafe_allow_html=True)
    weight_set = gas_weight_season(us_kpis["as_of"].month) if us_kpis else "winter"
    st.caption(
        f"Gas-demand weighted ({weight_set} weights): "
        + ", ".join(f"{p.split(' - ')[0]} {w:.0%}" for p, w in GAS_DEMAND_WEIGHTS[weight_set].items())
    )

    selected_padds = st.multiselect(
        "PADDs", list(PADD_REGIONS.keys()), list(PADD_REGIONS.keys()),
        key="us_hdd_padds", label_visibility="collapsed",
    )
    if not selected_padds:
        st.info("Select at least one PADD.")
    else:
        padd_cols = st.columns(len(selected_padds))
        for i, padd in enumerate(selected_padds):
            with padd_cols[i]:
                if view == "Forecast":
                    fhdd, ahdd = _forecast_vs_10yr(season_df, history_df, "padd_region", padd)
                    card = forecast_kpi_card(padd, fhdd, ahdd, small=True)
                else:
                    kpis = compute_season_kpis(season_df, normal_df, history_df, "padd_region", padd)
                    card = hdd_kpi_card(padd, kpis, small=True)
                st.markdown(card, unsafe_allow_html=True)

    st.divider()

    names = [US_WEIGHTED_LABEL] + selected_padds
    if view == "Forecast":
        if run_df.empty:
            st.info("No model-run data available.")
            return
        st.markdown("##### Run Comparison — 5-day window totals (latest 3 model runs)")
        st.caption(
            "Each box: total HDD per window for each run (darkest = latest), with the run-over-run "
            "change underneath (+ = newer run colder). Only dates covered by all 3 runs are summed, "
            "so runs compare like-for-like."
        )
        cols = st.columns(min(len(names), 3))
        for i, name in enumerate(names):
            totals = run_window_totals(us["run"] if name == US_WEIGHTED_LABEL else run_df, name)
            if totals.empty:
                continue
            with cols[i % len(cols)]:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.plotly_chart(make_hdd_run_totals_chart(totals, name), use_container_width=True)
                    st.plotly_chart(make_hdd_run_change_chart(run_over_run_change(totals), name),
                                    use_container_width=True)
    else:
        st.markdown("##### Cumulative Season HDD")
        cols = st.columns(min(len(names), 3))
        for i, name in enumerate(names):
            with cols[i % len(cols)]:
                if name == US_WEIGHTED_LABEL:
                    fig = make_hdd_season_chart(us["season"], name, us["normal"], us["history"])
                else:
                    fig = make_hdd_season_chart(season_df, name, normal_df, history_df)
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASIA HDD
# ══════════════════════════════════════════════════════════════════════════════
def render_asia_hdd(view: str = "Forecast"):
    """Asia HDD page. view: 'Forecast' (14-day vs 10-yr avg + run comparison)
    or 'Season so far' (cumulative-to-date vs normal + season charts)."""
    st.markdown(f"#### ASIA HEATING DEGREE DAYS — {view.upper()}")
    st.caption("Population-weighted HDD. Japan, South Korea, China North/East/South.")

    try:
        season_df = load_asia_hdd_season()
        normal_df = load_asia_hdd_normal()
        history_df = load_asia_hdd_season_history()
        run_df = load_asia_hdd_run_comparison()
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
    _off_season_banner(next((k["as_of"] for k in region_kpis.values() if k), None))

    kpi_cols = st.columns(len(selected))
    for i, region in enumerate(selected):
        with kpi_cols[i % len(kpi_cols)]:
            if view == "Forecast":
                fhdd, ahdd = _forecast_vs_10yr(season_df, history_df, "region", region)
                st.markdown(forecast_kpi_card(region, fhdd, ahdd), unsafe_allow_html=True)
            else:
                st.markdown(hdd_kpi_card(region, region_kpis[region]), unsafe_allow_html=True)

    st.divider()

    if view == "Forecast":
        if run_df.empty:
            st.info("No model-run data available.")
            return
        st.markdown("##### Run Comparison (latest 3 model runs)")
        cols = st.columns(min(len(selected), 3))
        for i, region in enumerate(selected):
            with cols[i % len(cols)]:
                st.plotly_chart(make_asia_hdd_comparison_chart(run_df, region), use_container_width=True)
    else:
        st.markdown("##### Cumulative Season HDD")
        cols = st.columns(min(len(selected), 3))
        for i, region in enumerate(selected):
            with cols[i % len(cols)]:
                st.plotly_chart(make_asia_hdd_region_chart(season_df, region, normal_df, history_df),
                                use_container_width=True)



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
# TAB 4: PANAMA CANAL
# ══════════════════════════════════════════════════════════════════════════════
def _gatun_status(level_ft: float) -> tuple[str, str]:
    """Draft-restriction status and its KPI card class for a lake level."""
    if level_ft < GATUN_RESTRICTION_FT:
        return "Critical — Major Restrictions", "kpi-card-critical"
    if level_ft < GATUN_RATE_SURCHARGE_FT:
        return "Caution — Rate Surcharges", "kpi-card-warning"
    return "Normal Operations", "kpi-card-good"


def _precip_kpi_card(label: str, window: dict | None) -> str:
    """Accumulated rainfall over a window, with its departure from normal.

    Below-normal rain is the bearish-for-transit case, so it takes the warning
    card; wet gets the neutral 'cool' accent. Departures inside the near-normal
    band are called out as such rather than shown as a small signed number the
    reader would over-read.
    """
    if not window:
        return raw_kpi_card(label, "N/A", "kpi-card-neutral")

    value = f"{window['actual_mm']:.0f} mm"
    pct = window.get("departure_pct")
    if pct is None:
        return raw_kpi_card(label, value, "kpi-card-neutral")

    if abs(pct) < PANAMA_PRECIP_NEAR_NORMAL_PCT:
        card_class, word, arrow, d_cls = "kpi-card-neutral", "near normal", "●", "kpi-delta-flat"
    elif pct < 0:
        card_class, word, arrow, d_cls = "kpi-card-warning", "below normal", "▼", "kpi-delta-down"
    else:
        card_class, word, arrow, d_cls = "kpi-card-cool", "above normal", "▲", "kpi-delta-up"

    delta = f'<div class="kpi-delta {d_cls}">{arrow} {abs(pct):.0f}% {word}</div>'
    return (f'<div class="kpi-card {card_class}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{delta}</div>')


def _render_gatun_forecast(hist_df: pd.DataFrame, proj_df: pd.DataFrame):
    """Level-forecast block: pick a method, tune its assumptions, draw the curve.

    All three methods extend the *official* ACP forecast rather than replacing
    it — each is anchored on the ACP curve's last level — so they answer "what
    happens after the official horizon runs out", which is the question the
    60-day ACP projection leaves open.
    """
    st.markdown("##### Level forecast")

    # The analog picker is driven by the live ONI record so newly-qualifying
    # events show up on their own; the hardcoded list is only the offline path.
    oni_df, oni_error = pd.DataFrame(), None
    try:
        oni_df = load_oni_monthly()
    except Exception as e:
        oni_error = str(e)

    el_nino_years = list_el_nino_onset_years(oni_df) if not oni_df.empty else {}
    latest_oni = float(oni_df["oni"].iloc[-1]) if not oni_df.empty else 0.0
    available_years = sorted(el_nino_years) or GATUN_FALLBACK_ANALOG_YEARS

    headline = None   # series the threshold verdict below is read off
    ctl, opt = st.columns([1, 2])
    with ctl:
        method = st.radio("Forecast method", GATUN_FORECAST_METHODS,
                          key="gatun_method", horizontal=False)

    if method == "Analog years":
        with opt:
            c1, c2 = st.columns([3, 1])
            with c1:
                years = st.multiselect(
                    "Analog El Nino years",
                    options=available_years,
                    default=[y for y in GATUN_ANALOG_YEARS if y in available_years] or available_years[-3:],
                    format_func=lambda y: (f"{y}  (ONI {el_nino_years[y]:+.2f})"
                                           if y in el_nino_years else str(y)),
                    key="gatun_analog_years",
                    help="Each year is an El Nino onset year — the event that developed "
                         "Sep of that year through the following Jan. Its observed lake "
                         "trajectory is shifted onto today's calendar.",
                )
            with c2:
                extend = st.checkbox("Extend", value=True, key="gatun_extend",
                                     help="Continue each analog past the official forecast's "
                                          "end, applying its level change to the ACP curve's "
                                          "last level.")
        if not years:
            st.info("Pick at least one analog year to draw.")
            return

        overlays = build_gatun_analog_overlays(hist_df, proj_df, years)
        extensions, mean_series = (build_gatun_analog_extension(hist_df, proj_df, years)
                                   if extend else ({}, None))
        missing = [y for y in years if y not in overlays]
        if missing:
            st.caption(f"No ACP level history for {', '.join(str(y) for y in missing)} — skipped.")
        fig = make_gatun_lake_chart(hist_df, proj_df, overlays, extensions, mean_series)
        st.plotly_chart(fig, use_container_width=True)
        headline = mean_series

    elif method == "ENSO index model":
        if oni_df.empty:
            st.error(f"Nino3.4 index unavailable from NOAA CPC — {oni_error}. "
                     "The ENSO index model needs it to fit monthly sensitivities.")
            return
        with opt:
            c1, c2, c3 = st.columns(3)
            with c1:
                target = st.slider("Assumed Nino3.4", -2.5, 3.0, round(latest_oni, 1), 0.1,
                                   key="gatun_enso_target",
                                   help=f"Index held at this level through the coming January, "
                                        f"then decaying to neutral over {GATUN_ENSO_DECAY_MONTHS} "
                                        f"months. Defaults to the latest observed ONI "
                                        f"({latest_oni:+.2f}).")
            with c2:
                band = st.slider("± band", 0.0, 1.5, GATUN_ENSO_INDEX_BAND, 0.1,
                                 key="gatun_enso_band",
                                 help="The model is re-run at target ± this, drawn as a ribbon.")
            with c3:
                horizon = st.slider("Horizon (days)", 90, 730, GATUN_ENSO_HORIZON_DAYS, 30,
                                    key="gatun_enso_horizon")

        nino_daily = oni_to_daily(oni_df)
        central, low, high = build_gatun_enso_index_forecast(
            hist_df, proj_df, nino_daily, target, band, horizon)
        if central is None:
            st.warning("Not enough overlapping lake/index history to fit the ENSO model.")
            return
        fig = make_gatun_enso_index_chart(hist_df, proj_df, central, low, high, target, band)
        st.plotly_chart(fig, use_container_width=True)
        headline = central
        st.caption(
            f"Latest observed ONI: **{latest_oni:+.2f}** ({oni_df['date'].iloc[-1]:%b %Y}, NOAA CPC). "
            "Monthly sensitivities are fit on the full 1965-present record, so the ribbon reflects "
            "the index assumption — not statistical confidence. Nino3.4 historically explains only "
            "a small share of Gatun's month-to-month change; treat the composite below as the "
            "sanity check on this curve."
        )

    else:  # ENSO composite
        with opt:
            c1, c2 = st.columns([3, 1])
            with c1:
                comp_years = st.multiselect(
                    "Composite years",
                    options=available_years,
                    default=[y for y in GATUN_COMPOSITE_YEARS if y in available_years] or available_years[-4:],
                    format_func=lambda y: (f"{y}  (ONI {el_nino_years[y]:+.2f})"
                                           if y in el_nino_years else str(y)),
                    key="gatun_composite_years",
                    help="Averaged rather than regressed — what the lake actually did in "
                         "these events, instead of an extrapolated index slope.",
                )
            with c2:
                horizon = st.slider("Horizon (days)", 90, 730, GATUN_ENSO_HORIZON_DAYS, 30,
                                    key="gatun_comp_horizon")
        if not comp_years:
            st.info("Pick at least one year to composite.")
            return

        extensions, mean_series, std_series = build_gatun_composite_forecast(
            hist_df, proj_df, comp_years, horizon)
        if not extensions:
            st.warning("No ACP level history for the selected composite years.")
            return
        fig = make_gatun_composite_chart(hist_df, proj_df, extensions, mean_series, std_series)
        st.plotly_chart(fig, use_container_width=True)
        headline = mean_series

    # Whichever method is showing, say plainly whether it breaches a threshold.
    _render_forecast_verdict(headline)


def _render_forecast_verdict(series):
    """Flag the deepest point of whichever projection is on screen against the
    two draft thresholds, so the chart's trading implication is stated in words."""
    if series is None or len(series) == 0:
        return
    trough = float(series.min())
    trough_date = series.idxmin()
    if trough < GATUN_RESTRICTION_FT:
        level, msg = "critical", (f"Projection troughs at {trough:.1f} ft around "
                                  f"{pd.Timestamp(trough_date):%b %Y} — below the "
                                  f"{GATUN_RESTRICTION_FT:.0f} ft restriction line. VLGCs would "
                                  "face draft limits or a Cape reroute.")
    elif trough < GATUN_RATE_SURCHARGE_FT:
        level, msg = "warning", (f"Projection troughs at {trough:.1f} ft around "
                                 f"{pd.Timestamp(trough_date):%b %Y} — below the "
                                 f"{GATUN_RATE_SURCHARGE_FT:.0f} ft surcharge line, above the "
                                 "restriction line.")
    else:
        level, msg = "good", (f"Projection stays above {GATUN_RATE_SURCHARGE_FT:.0f} ft "
                              f"throughout (trough {trough:.1f} ft) — no transit constraint implied.")
    st.markdown(status_banner(msg, level), unsafe_allow_html=True)


def _render_panama_precip():
    """Watershed rainfall vs normal — the upstream driver of the level charts above."""
    st.markdown("##### Watershed precipitation vs normal")
    st.caption("Area-average rainfall over the Gatun / Alhajuela catchment against its "
               "2000-2024 day-of-year normal. Rain into this catchment is the lake's only "
               "inflow, so a sustained dry departure leads the level down by weeks to months.")

    try:
        precip_df = load_panama_precip_anomaly()
    except Exception as e:
        st.markdown(
            status_banner("Watershed precipitation unavailable. If panama_precip_anomaly has "
                          "not been built yet, run lpg_desk_refresh.py in Databricks to "
                          f"populate it. ({e})", "warning"),
            unsafe_allow_html=True,
        )
        return

    if precip_df.empty:
        st.info("No Panama watershed precipitation data available.")
        return

    summary = summarize_panama_precip(precip_df)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(_precip_kpi_card(f"Last {PANAMA_PRECIP_LOOKBACK_DAYS}d rainfall",
                                     summary["lookback"]), unsafe_allow_html=True)
    with k2:
        st.markdown(_precip_kpi_card(f"Next {PANAMA_PRECIP_FORECAST_DAYS}d forecast",
                                     summary["forecast"]), unsafe_allow_html=True)
    with k3:
        cum = summary["cumulative_departure_mm"]
        if cum is None:
            st.markdown(raw_kpi_card("Cumulative vs normal", "N/A"), unsafe_allow_html=True)
        else:
            cls = "kpi-card-warning" if cum < 0 else "kpi-card-cool"
            st.markdown(raw_kpi_card("Cumulative vs normal", f"{cum:+.0f} mm", cls),
                        unsafe_allow_html=True)

    fcst = summary.get("forecast")
    if fcst and fcst.get("departure_pct") is not None:
        pct, mm = fcst["departure_pct"], fcst["departure_mm"]
        if abs(pct) < PANAMA_PRECIP_NEAR_NORMAL_PCT:
            level, verdict = "good", f"near normal ({pct:+.0f}%)"
        elif pct < 0:
            level, verdict = "warning", f"BELOW normal ({pct:+.0f}%, {mm:+.0f} mm)"
        else:
            level, verdict = "good", f"ABOVE normal ({pct:+.0f}%, {mm:+.0f} mm)"
        st.markdown(
            status_banner(f"Next {fcst['days']} days of catchment rainfall are forecast "
                          f"{verdict} — {fcst['actual_mm']:.0f} mm vs a "
                          f"{fcst['normal_mm']:.0f} mm normal.", level),
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_panama_precip_chart(precip_df), use_container_width=True)
    with c2:
        st.plotly_chart(make_panama_precip_departure_chart(precip_df), use_container_width=True)


def render_gatun_lake():
    st.markdown("#### PANAMA CANAL — GATUN LAKE")
    st.caption("Observed water level and official ACP forecast, extended by El Nino analogs or "
               "an ENSO-index model, plus the watershed rainfall driving them. "
               "Sources: Panama Canal Authority + NOAA CPC (live), ECMWF via Databricks.")

    try:
        hist_df, proj_df = load_gatun_lake_levels()
    except Exception as e:
        st.error(f"Gatun Lake data unavailable: {e}")
        return

    if hist_df.empty:
        st.warning("No Gatun Lake level data — the ACP endpoint may be unreachable.")
        return

    recent = hist_df.sort_values("date")
    current_level = float(recent["level_ft"].iloc[-1])
    status, card_class = _gatun_status(current_level)

    trend = ""
    week_ago = recent[recent["date"] <= recent["date"].iloc[-1] - pd.Timedelta(days=7)]
    if not week_ago.empty:
        delta = current_level - float(week_ago["level_ft"].iloc[-1])
        trend = f"{'▲' if delta > 0 else '▼'} {abs(delta):.2f} ft over 7 days"

    kc1, kc2, kc3 = st.columns([1, 1, 2])
    with kc1:
        st.markdown(raw_kpi_card("Gatun Lake Level", f"{current_level:.1f} ft", card_class),
                    unsafe_allow_html=True)
    with kc2:
        st.markdown(raw_kpi_card("7-day trend", trend or "N/A", "kpi-card-neutral"),
                    unsafe_allow_html=True)
    with kc3:
        st.markdown(raw_kpi_card("Status", status, card_class), unsafe_allow_html=True)

    st.divider()
    _render_gatun_forecast(hist_df, proj_df)
    st.divider()
    _render_panama_precip()


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
    st.caption(
        "Live active-storm tracking (NHC + IBTrACS). LPG export/loading disruption risk. "
        "Solid line = observed track so far, dotted line = forecast."
    )

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
    "Panama Canal": render_gatun_lake,
    "PADD 3 Coastal": render_padd3_coastal,
    "Hurricanes": render_hurricanes,
    "N. Africa Wind": render_nafrica_wind,
}

# Sections that open a sub-menu under their sidebar entry when selected.
SUBPAGES = {
    "US HDD": ["Forecast", "Season so far"],
    "Asia HDD": ["Forecast", "Season so far"],
}

# One radio holds both levels: the active section's sub-items are spliced in
# right under it. Its key carries a generation counter, because radio state is
# stored by option *index* — a changed option list (or a click that needs
# normalising, see below) must start a fresh widget rather than reuse stale state.
nav_parent = st.session_state.get("nav_parent", "US HDD")
nav_sub = st.session_state.get("nav_sub", SUBPAGES.get(nav_parent, [None])[0])
nav_gen = st.session_state.get("nav_gen", 0)
nav_options = []
for _page in PAGES:
    nav_options.append((_page, None))
    if _page == nav_parent:
        nav_options += [(_page, _s) for _s in SUBPAGES.get(_page, [])]
_current = (nav_parent, nav_sub) if nav_parent in SUBPAGES else (nav_parent, None)
if _current not in nav_options:
    _current = nav_options[0]

with st.sidebar:
    st.markdown("### Sections")
    nav_choice = st.radio(
        "Sections", nav_options, index=nav_options.index(_current),
        format_func=lambda o: o[0] if o[1] is None else f" ↳ {o[1]}",
        label_visibility="collapsed", key=f"nav_{nav_gen}",
    )
    st.divider()
    # A separate, distinct action (not a section to browse) — one click both
    # switches to this view and runs the analysis, instead of navigate-then-Generate.
    generate_clicked = st.button(
        "Generate AI Brief", type="primary", use_container_width=True, key="ai_brief_sidebar_btn",
    )

new_parent, new_sub = nav_choice
if new_parent != nav_parent or (new_sub is None and new_parent in SUBPAGES):
    # Moved to another section, or clicked the already-open section's own row:
    # land on its first sub-view (or keep the current one) and redraw the menu.
    st.session_state["nav_parent"] = new_parent
    st.session_state["nav_sub"] = (nav_sub if new_parent == nav_parent
                                   else SUBPAGES.get(new_parent, [None])[0])
    st.session_state["nav_gen"] = nav_gen + 1
    st.rerun()
st.session_state["nav_sub"] = new_sub if new_parent in SUBPAGES else None
selected_page = new_parent

if generate_clicked:
    st.session_state["show_ai_brief"] = True
elif st.session_state.get("_last_nav_page") != selected_page:
    st.session_state["show_ai_brief"] = False
st.session_state["_last_nav_page"] = selected_page

if st.session_state.get("show_ai_brief"):
    render_lpg_brief(auto_generate=generate_clicked)
elif selected_page in SUBPAGES:
    PAGES[selected_page](st.session_state["nav_sub"])
else:
    PAGES[selected_page]()
