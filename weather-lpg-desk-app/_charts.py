"""Chart builders — LPG Desk Weather Dashboard.

All functions return Plotly Figure objects styled with the light trading-desk
theme. Color follows the job it does: categorical for series identity
(fixed per-entity color from _config, never re-assigned when a selection
changes), status tokens for threshold/alert lines, one ordinal ramp for storm
severity, and the diverging blue/red pair for anomaly polarity.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from _config import (
    PADD_COLORS, ASIA_COLORS, INDIA_COLORS, PADD3_COASTAL_COLORS,
    NAFRICA_WIND_COLORS, GATUN_ANALOG_RAMP,
    GATUN_RATE_SURCHARGE_FT, GATUN_RESTRICTION_FT,
    WIND_CAUTION_MS, WIND_WARNING_MS, WIND_GALE_MS,
    FREEZE_THRESHOLD_C, FROST_THRESHOLD_C,
    HDD_HISTORY_COLORS, HDD_OVERLAY_YEARS, HDD_AVG_YEARS, hdd_season_label,
)
from _style import (
    PLOTLY_LAYOUT, GEO_LAYOUT, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE,
    BASELINE, DIV_POS, DIV_NEG, STATUS_WARNING, STATUS_CRITICAL,
    STORM_RAMP, CAT_BLUE, CAT_GREEN, CAT_ORANGE, CAT_RED,
)


def _base_fig(title: str = "", height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14)), height=height)
    return fig


def _season_start_year(dates: pd.Series) -> int:
    """Infer the Oct-1 season start year from a set of in-season dates."""
    if dates.empty:
        return pd.Timestamp.today().year
    first = pd.to_datetime(dates).min()
    return first.year if first.month >= 10 else first.year - 1


def _doy_to_season_date(day_of_year: pd.Series, season_start_year: int) -> pd.Series:
    """Map an Oct1-Mar31 day-of-year series onto real calendar dates."""
    def _one(doy):
        year = season_start_year if doy >= 274 else season_start_year + 1
        return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(doy) - 1)
    return day_of_year.apply(_one)


def _add_hdd_history_overlay(fig: go.Figure, history_df: pd.DataFrame, group_col: str, group_value: str,
                              season_year: int, overlay_years: int = HDD_OVERLAY_YEARS,
                              name_prefix: str = "", showlegend: bool = True) -> None:
    """Overlay prior completed winters' cumulative HDD (muted gray, dotted) onto the
    current season's calendar, so they read as history rather than competing with
    the categorical PADD/region color used for the current season."""
    if history_df is None or history_df.empty:
        return
    hdf = history_df[history_df[group_col] == group_value] if group_col in history_df.columns else history_df
    if hdf.empty:
        return
    years = sorted(hdf["season_start_year"].dropna().unique())[-overlay_years:]
    palette = HDD_HISTORY_COLORS[-len(years):]
    for color, yr in zip(palette, years):
        ydf = hdf[hdf["season_start_year"] == yr].copy()
        if ydf.empty:
            continue
        ydf["date"] = _doy_to_season_date(ydf["day_of_year"], season_year)
        ydf = ydf.sort_values("date")
        label = ydf["season_label"].iloc[0] if "season_label" in ydf.columns else hdd_season_label(yr)
        fig.add_trace(go.Scatter(
            x=ydf["date"], y=ydf["cumulative_hdd"],
            name=f"{name_prefix}{label}", mode="lines",
            line=dict(color=color, dash="dot", width=1.3),
            opacity=0.9, showlegend=showlegend,
        ))


def _add_hdd_5yr_avg_line(fig: go.Figure, history_df: pd.DataFrame, group_col: str, group_value: str,
                           season_year: int, avg_years: int = HDD_AVG_YEARS,
                           showlegend: bool = True) -> None:
    """Overlay the average cumulative HDD of the last N completed winters as a
    single reference line — thicker dashed, distinct from the individual prior-
    winter overlays so the reader can quickly compare the current season to
    'recent normal'."""
    if history_df is None or history_df.empty:
        return
    hdf = history_df[history_df[group_col] == group_value] if group_col in history_df.columns else history_df
    if hdf.empty:
        return
    years = sorted(hdf["season_start_year"].dropna().unique())[-avg_years:]
    if len(years) < 2:
        return
    frames = []
    for yr in years:
        ydf = hdf[hdf["season_start_year"] == yr].copy()
        if ydf.empty:
            continue
        ydf["date"] = _doy_to_season_date(ydf["day_of_year"], season_year)
        # Ensure unique dates per year (take last cumulative value per date)
        series = ydf.groupby("date")["cumulative_hdd"].last()
        frames.append(series.rename(yr))
    if not frames:
        return
    avg = pd.concat(frames, axis=1).mean(axis=1).dropna()
    if avg.empty:
        return
    fig.add_trace(go.Scatter(
        x=avg.index, y=avg.values,
        name=f"{len(years)}-yr avg", mode="lines",
        line=dict(color=INK_SECONDARY, dash="dashdot", width=2.2),
        opacity=0.85, showlegend=showlegend,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: US HDD SEASON CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def make_hdd_season_chart(df: pd.DataFrame, padd: str, normal_df: pd.DataFrame = None,
                           history_df: pd.DataFrame = None) -> go.Figure:
    """Cumulative HDD season chart for a single PADD — actual, forecast, normal, prior winters."""
    fig = _base_fig(f"Cumulative HDD — {padd}")
    padd_data = df[df["padd_region"] == padd] if "padd_region" in df.columns else df
    color = PADD_COLORS.get(padd, INK_MUTED)
    season_year = _season_start_year(padd_data["delivery_date"])

    for model in padd_data["model_label"].unique():
        mdf = padd_data[padd_data["model_label"] == model]
        dash = "solid" if "Actual" in model else "dash"
        fig.add_trace(go.Scatter(
            x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
            name=model, mode="lines",
            line=dict(color=color, dash=dash, width=2),
        ))

    # Bridge: connect last actual to first forecast with a thin dashed segment
    actual_sub = padd_data[padd_data["model_label"].str.contains("Actual", na=False)]
    fcst_sub = padd_data[~padd_data["model_label"].str.contains("Actual", na=False)]
    if not actual_sub.empty and not fcst_sub.empty:
        la = actual_sub.sort_values("delivery_date").iloc[-1]
        ff = fcst_sub.sort_values("delivery_date").iloc[0]
        fig.add_trace(go.Scatter(
            x=[la["delivery_date"], ff["delivery_date"]],
            y=[la["cumulative_hdd"], ff["cumulative_hdd"]],
            mode="lines", line=dict(color=color, dash="dash", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    _add_hdd_history_overlay(fig, history_df, "padd_region", padd, season_year)
    _add_hdd_5yr_avg_line(fig, history_df, "padd_region", padd, season_year)

    if normal_df is not None and not normal_df.empty:
        ndf = normal_df[normal_df["padd_region"] == padd] if "padd_region" in normal_df.columns else normal_df
        if not ndf.empty:
            ndf = ndf.copy()
            ndf["date"] = _doy_to_season_date(ndf["day_of_year"], season_year)
            ndf = ndf.sort_values("date")   # season order (Oct1->Mar31), not raw day_of_year
            ndf["cumulative_hdd_normal"] = ndf["daily_hdd_normal"].cumsum()
            fig.add_trace(go.Scatter(
                x=ndf["date"], y=ndf["cumulative_hdd_normal"],
                name="Normal", mode="lines",
                line=dict(color=INK_MUTED, dash="dot", width=1.5),
            ))

    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


def make_hdd_comparison_chart(df: pd.DataFrame, padd: str) -> go.Figure:
    """Run comparison — daily HDD bars for the latest 3 model runs."""
    fig = _base_fig(f"Daily HDD Forecast — {padd}", height=320)
    padd_data = df[df["padd_region"] == padd] if "padd_region" in df.columns else df
    base = PADD_COLORS.get(padd, INK_MUTED)
    opacities = {1: 1.0, 2: 0.6, 3: 0.35}

    for rank in sorted(padd_data["run_rank"].dropna().unique()):
        rdf = padd_data[padd_data["run_rank"] == rank]
        label = rdf["run_label"].iloc[0] if not rdf.empty else f"Run {rank}"
        fig.add_trace(go.Bar(
            x=rdf["delivery_date"], y=rdf["daily_hdd"],
            name=label, marker_color=base, opacity=opacities.get(int(rank), 0.3),
        ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASIA HDD
# ══════════════════════════════════════════════════════════════════════════════

def make_asia_hdd_season_chart(df: pd.DataFrame, selected_regions: list[str],
                                normal_df: pd.DataFrame = None,
                                history_df: pd.DataFrame = None) -> go.Figure:
    """Cumulative HDD for selected Asia regions — actual, forecast, normal, prior winters."""
    fig = _base_fig("Asia HDD Season — Cumulative (population-weighted)")

    for region in selected_regions:
        rdf = df[df["region"] == region]
        color = ASIA_COLORS.get(region, INK_MUTED)
        season_year = _season_start_year(rdf["delivery_date"])
        for model in rdf["model_label"].unique():
            mdf = rdf[rdf["model_label"] == model]
            dash = "solid" if "Actual" in model else "dash"
            fig.add_trace(go.Scatter(
                x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
                name=f"{region} ({model})", mode="lines",
                line=dict(color=color, dash=dash, width=2),
            ))
        _add_hdd_history_overlay(fig, history_df, "region", region, season_year,
                                  name_prefix=f"{region} ", showlegend=False)
        if normal_df is not None and not normal_df.empty:
            ndf = normal_df[normal_df["region"] == region]
            if not ndf.empty:
                ndf = ndf.copy()
                ndf["date"] = _doy_to_season_date(ndf["day_of_year"], season_year)
                ndf = ndf.sort_values("date")
                ndf["cumulative_hdd_normal"] = ndf["daily_hdd_normal"].cumsum()
                fig.add_trace(go.Scatter(
                    x=ndf["date"], y=ndf["cumulative_hdd_normal"],
                    name=f"{region} (normal)", mode="lines",
                    line=dict(color=color, dash="dot", width=1),
                    opacity=0.6, showlegend=False,
                ))

    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


def make_asia_hdd_region_chart(df: pd.DataFrame, region: str,
                                normal_df: pd.DataFrame = None,
                                history_df: pd.DataFrame = None) -> go.Figure:
    """Cumulative HDD season chart for a single Asia region — mirrors make_hdd_season_chart."""
    fig = _base_fig(f"Cumulative HDD — {region}")
    rdf = df[df["region"] == region] if "region" in df.columns else df
    color = ASIA_COLORS.get(region, INK_MUTED)
    season_year = _season_start_year(rdf["delivery_date"])

    for model in rdf["model_label"].unique():
        mdf = rdf[rdf["model_label"] == model]
        dash = "solid" if "Actual" in model else "dash"
        fig.add_trace(go.Scatter(
            x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
            name=model, mode="lines",
            line=dict(color=color, dash=dash, width=2),
        ))

    # Bridge: connect last actual to first forecast
    actual_sub = rdf[rdf["model_label"].str.contains("Actual", na=False)]
    fcst_sub = rdf[~rdf["model_label"].str.contains("Actual", na=False)]
    if not actual_sub.empty and not fcst_sub.empty:
        la = actual_sub.sort_values("delivery_date").iloc[-1]
        ff = fcst_sub.sort_values("delivery_date").iloc[0]
        fig.add_trace(go.Scatter(
            x=[la["delivery_date"], ff["delivery_date"]],
            y=[la["cumulative_hdd"], ff["cumulative_hdd"]],
            mode="lines", line=dict(color=color, dash="dash", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    _add_hdd_history_overlay(fig, history_df, "region", region, season_year)
    _add_hdd_5yr_avg_line(fig, history_df, "region", region, season_year)

    if normal_df is not None and not normal_df.empty:
        ndf = normal_df[normal_df["region"] == region] if "region" in normal_df.columns else normal_df
        if not ndf.empty:
            ndf = ndf.copy()
            ndf["date"] = _doy_to_season_date(ndf["day_of_year"], season_year)
            ndf = ndf.sort_values("date")
            ndf["cumulative_hdd_normal"] = ndf["daily_hdd_normal"].cumsum()
            fig.add_trace(go.Scatter(
                x=ndf["date"], y=ndf["cumulative_hdd_normal"],
                name="Normal", mode="lines",
                line=dict(color=INK_MUTED, dash="dot", width=1.5),
            ))

    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


def make_asia_hdd_comparison_chart(df: pd.DataFrame, region: str) -> go.Figure:
    """Run comparison — daily HDD bars for the latest 3 model runs, single Asia region."""
    fig = _base_fig(f"Daily HDD Forecast — {region}", height=320)
    rdf = df[df["region"] == region] if "region" in df.columns else df
    base = ASIA_COLORS.get(region, INK_MUTED)
    opacities = {1: 1.0, 2: 0.6, 3: 0.35}

    for rank in sorted(rdf["run_rank"].dropna().unique()):
        rrdf = rdf[rdf["run_rank"] == rank]
        label = rrdf["run_label"].iloc[0] if not rrdf.empty else f"Run {rank}"
        fig.add_trace(go.Bar(
            x=rrdf["delivery_date"], y=rrdf["daily_hdd"],
            name=label, marker_color=base, opacity=opacities.get(int(rank), 0.3),
        ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 10-YEAR COMPARISON CHARTS (toggle view)
# ══════════════════════════════════════════════════════════════════════════════

def make_hdd_10yr_comparison_chart(df: pd.DataFrame, padd: str,
                                    history_df: pd.DataFrame,
                                    n_years: int = 10) -> go.Figure:
    """Current season + forecast vs last N winters' min/max band and average."""
    fig = _base_fig(f"Forecast vs {n_years}-yr Range — {padd}")
    color = PADD_COLORS.get(padd, INK_MUTED)
    padd_data = df[df["padd_region"] == padd] if "padd_region" in df.columns else df
    season_year = _season_start_year(padd_data["delivery_date"])

    hdf = history_df[history_df["padd_region"] == padd] if "padd_region" in history_df.columns else history_df
    if not hdf.empty:
        years = sorted(hdf["season_start_year"].dropna().unique())[-n_years:]
        series_list = []
        for yr in years:
            ydf = hdf[hdf["season_start_year"] == yr].copy()
            if ydf.empty:
                continue
            ydf["date"] = _doy_to_season_date(ydf["day_of_year"], season_year)
            series_list.append(ydf.groupby("date")["cumulative_hdd"].last().rename(yr))
        if series_list:
            combined = pd.concat(series_list, axis=1)
            avg = combined.mean(axis=1).dropna().sort_index()
            band_min = combined.min(axis=1).dropna().sort_index()
            band_max = combined.max(axis=1).dropna().sort_index()
            fig.add_trace(go.Scatter(
                x=list(band_max.index) + list(band_min.index[::-1]),
                y=list(band_max.values) + list(band_min.values[::-1]),
                fill="toself", fillcolor="rgba(180,180,180,0.25)",
                line=dict(color="rgba(180,180,180,0)"),
                name=f"{n_years}-yr range", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=avg.index, y=avg.values,
                name=f"{n_years}-yr avg", mode="lines",
                line=dict(color=INK_SECONDARY, dash="dashdot", width=2),
            ))

    for model in padd_data["model_label"].unique():
        mdf = padd_data[padd_data["model_label"] == model]
        dash = "solid" if "Actual" in model else "dash"
        label = "Current (Actual)" if "Actual" in model else "Current (Forecast)"
        fig.add_trace(go.Scatter(
            x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
            name=label, mode="lines",
            line=dict(color=color, dash=dash, width=2.5),
        ))
    actual_sub = padd_data[padd_data["model_label"].str.contains("Actual", na=False)]
    fcst_sub = padd_data[~padd_data["model_label"].str.contains("Actual", na=False)]
    if not actual_sub.empty and not fcst_sub.empty:
        la = actual_sub.sort_values("delivery_date").iloc[-1]
        ff = fcst_sub.sort_values("delivery_date").iloc[0]
        fig.add_trace(go.Scatter(
            x=[la["delivery_date"], ff["delivery_date"]],
            y=[la["cumulative_hdd"], ff["cumulative_hdd"]],
            mode="lines", line=dict(color=color, dash="dash", width=2.5),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


def make_asia_hdd_10yr_comparison_chart(df: pd.DataFrame, region: str,
                                        history_df: pd.DataFrame,
                                        n_years: int = 10) -> go.Figure:
    """Current season + forecast vs last N winters' min/max band and average (Asia)."""
    fig = _base_fig(f"Forecast vs {n_years}-yr Range — {region}")
    color = ASIA_COLORS.get(region, INK_MUTED)
    rdf = df[df["region"] == region] if "region" in df.columns else df
    season_year = _season_start_year(rdf["delivery_date"])

    hdf = history_df[history_df["region"] == region] if "region" in history_df.columns else history_df
    if not hdf.empty:
        years = sorted(hdf["season_start_year"].dropna().unique())[-n_years:]
        series_list = []
        for yr in years:
            ydf = hdf[hdf["season_start_year"] == yr].copy()
            if ydf.empty:
                continue
            ydf["date"] = _doy_to_season_date(ydf["day_of_year"], season_year)
            series_list.append(ydf.groupby("date")["cumulative_hdd"].last().rename(yr))
        if series_list:
            combined = pd.concat(series_list, axis=1)
            avg = combined.mean(axis=1).dropna().sort_index()
            band_min = combined.min(axis=1).dropna().sort_index()
            band_max = combined.max(axis=1).dropna().sort_index()
            fig.add_trace(go.Scatter(
                x=list(band_max.index) + list(band_min.index[::-1]),
                y=list(band_max.values) + list(band_min.values[::-1]),
                fill="toself", fillcolor="rgba(180,180,180,0.25)",
                line=dict(color="rgba(180,180,180,0)"),
                name=f"{n_years}-yr range", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=avg.index, y=avg.values,
                name=f"{n_years}-yr avg", mode="lines",
                line=dict(color=INK_SECONDARY, dash="dashdot", width=2),
            ))

    for model in rdf["model_label"].unique():
        mdf = rdf[rdf["model_label"] == model]
        dash = "solid" if "Actual" in model else "dash"
        label = "Current (Actual)" if "Actual" in model else "Current (Forecast)"
        fig.add_trace(go.Scatter(
            x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
            name=label, mode="lines",
            line=dict(color=color, dash=dash, width=2.5),
        ))
    actual_sub = rdf[rdf["model_label"].str.contains("Actual", na=False)]
    fcst_sub = rdf[~rdf["model_label"].str.contains("Actual", na=False)]
    if not actual_sub.empty and not fcst_sub.empty:
        la = actual_sub.sort_values("delivery_date").iloc[-1]
        ff = fcst_sub.sort_values("delivery_date").iloc[0]
        fig.add_trace(go.Scatter(
            x=[la["delivery_date"], ff["delivery_date"]],
            y=[la["cumulative_hdd"], ff["cumulative_hdd"]],
            mode="lines", line=dict(color=color, dash="dash", width=2.5),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


def make_asia_hdd_bar_chart(df: pd.DataFrame, selected_regions: list[str]) -> go.Figure:
    """14-day forecast HDD total by Asia region."""
    fig = _base_fig("14-Day Forecast HDD Total", height=300)

    for region in selected_regions:
        rdf = df[(df["region"] == region) & (df["model_label"].str.contains("EC-ens", na=False))]
        total = rdf["daily_hdd"].sum()
        fig.add_trace(go.Bar(
            x=[region], y=[total], name=region,
            marker_color=ASIA_COLORS.get(region, INK_MUTED), showlegend=False,
        ))

    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: INDIA PRECIPITATION ANOMALY
# ══════════════════════════════════════════════════════════════════════════════

def make_india_precip_timeseries(df: pd.DataFrame) -> go.Figure:
    """Precipitation actual + forecast vs climatology, by subregion."""
    fig = _base_fig("India Precipitation — Actual / Forecast vs Climatology")

    for region in df["subregion"].unique():
        rdf = df[df["subregion"] == region]
        color = INDIA_COLORS.get(region, INK_MUTED)
        act = rdf.dropna(subset=["precip_actual_mm"])
        if not act.empty:
            fig.add_trace(go.Scatter(
                x=act["delivery_date"], y=act["precip_actual_mm"],
                name=region, mode="lines", line=dict(color=color, width=2),
            ))
        fcst = rdf.dropna(subset=["precip_forecast_mm"])
        if not fcst.empty:
            fig.add_trace(go.Scatter(
                x=fcst["delivery_date"], y=fcst["precip_forecast_mm"],
                name=f"{region} (fcst)", mode="lines",
                line=dict(color=color, width=2, dash="dash"), showlegend=False,
            ))
        clim = rdf.dropna(subset=["precip_clim_mm"])
        if not clim.empty:
            fig.add_trace(go.Scatter(
                x=clim["delivery_date"], y=clim["precip_clim_mm"],
                name=f"{region} (normal)", mode="lines",
                line=dict(color=color, width=1, dash="dot"), opacity=0.6, showlegend=False,
            ))

    fig.update_yaxes(title_text="Precipitation (mm/day)")
    return fig


def make_india_precip_deviation_chart(df: pd.DataFrame) -> go.Figure:
    """Diverging bar of latest precipitation deviation (%) by subregion.
    
    Falls back to a daily precipitation bar chart when deviation is unavailable
    (climatology baseline still building).
    """
    latest = df[df["precip_actual_mm"].notna()].sort_values("delivery_date").groupby("subregion").last().reset_index()

    # If deviation is available, show the diverging bar
    if latest["deviation_pct"].notna().any():
        fig = _base_fig("Precipitation Deviation vs Normal (%)", height=300)
        vals = latest["deviation_pct"].fillna(0)
        fig.add_trace(go.Bar(
            x=latest["subregion"], y=vals,
            marker_color=[DIV_POS if v >= 0 else DIV_NEG for v in vals],
            showlegend=False,
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=INK_MUTED)
        fig.update_yaxes(title_text="Deviation (%)")
    else:
        # Fallback: show latest actual precip as bars
        fig = _base_fig("Latest Daily Precipitation (mm)", height=300)
        from _config import INDIA_COLORS
        fig.add_trace(go.Bar(
            x=latest["subregion"], y=latest["precip_actual_mm"],
            marker_color=[INDIA_COLORS.get(s, INK_MUTED) for s in latest["subregion"]],
            showlegend=False,
        ))
        fig.update_yaxes(title_text="Precipitation (mm/day)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PANAMA CANAL — Gatun level + forecasts, watershed precipitation
# ══════════════════════════════════════════════════════════════════════════════
# Every forecast chart shares one base — observed, official ACP forecast, and
# the two draft thresholds — so switching method changes only the projected
# curve and the reader keeps the same reference frame.

def _gatun_base_fig(title: str, height: int = 470) -> go.Figure:
    fig = _base_fig(title, height=height)
    fig.update_yaxes(title_text="Lake Level (feet)")
    return fig


def _add_gatun_observed(fig: go.Figure, hist_df: pd.DataFrame, proj_df: pd.DataFrame) -> None:
    if not hist_df.empty:
        recent = hist_df[hist_df["date"] >= pd.Timestamp.today() - pd.DateOffset(days=365)]
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["level_ft"], name="Observed", mode="lines",
            line=dict(color=CAT_BLUE, width=2.5),
        ))
    if not proj_df.empty:
        fig.add_trace(go.Scatter(
            x=proj_df["date"], y=proj_df["level_ft"], name="Official ACP Forecast", mode="lines",
            line=dict(color=CAT_RED, width=2, dash="dash"),
        ))


def _add_gatun_thresholds(fig: go.Figure) -> None:
    fig.add_hline(y=GATUN_RATE_SURCHARGE_FT, line_color=STATUS_WARNING, line_dash="dash", line_width=1.5,
                  annotation_text=f"{GATUN_RATE_SURCHARGE_FT:.0f} ft — Increased Shipping Rates",
                  annotation_font_color=STATUS_WARNING, annotation_position="bottom right")
    fig.add_hline(y=GATUN_RESTRICTION_FT, line_color=STATUS_CRITICAL, line_dash="dash", line_width=1.5,
                  annotation_text=f"{GATUN_RESTRICTION_FT:.0f} ft — Significant Restrictions",
                  annotation_font_color=STATUS_CRITICAL, annotation_position="bottom right")


def _analog_color(position: int) -> str:
    """Analog years read as history, so they take one muted ramp by draw position
    rather than a categorical hue — adding a year in the UI can't restyle the
    ones already on the chart, and none of them competes with observed/forecast."""
    return GATUN_ANALOG_RAMP[position % len(GATUN_ANALOG_RAMP)]


def make_gatun_lake_chart(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                           analog_overlays: dict[int, pd.DataFrame],
                           extensions: dict[int, pd.Series] | None = None,
                           mean_series: pd.Series | None = None) -> go.Figure:
    """Observed + official ACP forecast + El Nino analog years.

    With `extensions`/`mean_series` supplied, each analog also continues past the
    official forecast's end as a level-change extension anchored on its last
    point, and their mean is drawn as the headline projection.
    """
    titled = "Gatun Lake — Observed, Official Forecast & El Nino Analog Years"
    if extensions:
        titled += " (extended)"
    fig = _gatun_base_fig(titled)
    _add_gatun_observed(fig, hist_df, proj_df)

    for i, (year, adf) in enumerate(sorted(analog_overlays.items())):
        fig.add_trace(go.Scatter(
            x=adf["date"], y=adf["level_ft"], name=f"{year} (El Nino analog)", mode="lines",
            line=dict(color=_analog_color(i), width=1.5, dash="dot"), opacity=0.85,
        ))

    for i, (year, series) in enumerate(sorted((extensions or {}).items())):
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=f"{year} extension", mode="lines",
            line=dict(color=_analog_color(i), width=1.8), opacity=0.75, showlegend=False,
            hovertemplate=f"{year} extension<br>%{{x|%d %b %Y}} · %{{y:.2f}} ft<extra></extra>",
        ))

    if mean_series is not None and not mean_series.empty:
        fig.add_trace(go.Scatter(
            x=mean_series.index, y=mean_series.values, name="Mean of analog extensions",
            mode="lines", line=dict(color=INK_PRIMARY, width=2.8),
            hovertemplate="Analog mean<br>%{x|%d %b %Y} · %{y:.2f} ft<extra></extra>",
        ))

    _add_gatun_thresholds(fig)
    return fig


def make_gatun_enso_index_chart(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                                 central: pd.Series | None, low: pd.Series | None,
                                 high: pd.Series | None, target_index: float,
                                 band: float) -> go.Figure:
    """ENSO-regression forecast: level rolled forward under an assumed Nino3.4 level.

    The ribbon spans the same model re-run at target +/- band — it shows how much
    the projection moves with the index assumption, not a statistical interval.
    """
    fig = _gatun_base_fig(f"Gatun Lake — ENSO Index Model (Nino3.4 = {target_index:+.1f} ± {band:.1f})")
    _add_gatun_observed(fig, hist_df, proj_df)

    if central is not None and not central.empty:
        if low is not None and high is not None:
            fig.add_trace(go.Scatter(
                x=high.index, y=high.values, mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=low.index, y=low.values, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=_hex_to_rgba(CAT_GREEN, 0.16),
                name=f"Nino3.4 {target_index - band:+.1f} to {target_index + band:+.1f}",
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=central.index, y=central.values, name="ENSO model forecast", mode="lines",
            line=dict(color=CAT_GREEN, width=2.8),
            hovertemplate="ENSO model<br>%{x|%d %b %Y} · %{y:.2f} ft<extra></extra>",
        ))

    _add_gatun_thresholds(fig)
    return fig


def make_gatun_composite_chart(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                                extensions: dict[int, pd.Series],
                                mean_series: pd.Series | None,
                                std_series: pd.Series | None) -> go.Figure:
    """Mean trajectory of the strongest El Nino events, +/- 1 std across them."""
    years = ", ".join(str(y) for y in sorted(extensions)) or "n/a"
    fig = _gatun_base_fig(f"Gatun Lake — ENSO Composite Forecast ({years})")
    _add_gatun_observed(fig, hist_df, proj_df)

    for i, (year, series) in enumerate(sorted(extensions.items())):
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=str(year), mode="lines",
            line=dict(color=_analog_color(i), width=1.4, dash="dot"), opacity=0.7,
        ))

    if mean_series is not None and not mean_series.empty:
        if std_series is not None and not std_series.empty:
            fig.add_trace(go.Scatter(
                x=mean_series.index, y=(mean_series + std_series).values, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=mean_series.index, y=(mean_series - std_series).values, mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=_hex_to_rgba(CAT_ORANGE, 0.16),
                name="± 1 std across composite years", hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=mean_series.index, y=mean_series.values, name="Composite forecast",
            mode="lines", line=dict(color=CAT_ORANGE, width=2.8),
            hovertemplate="Composite<br>%{x|%d %b %Y} · %{y:.2f} ft<extra></extra>",
        ))

    _add_gatun_thresholds(fig)
    return fig


# ─── Panama Canal watershed precipitation ────────────────────────────────────────

def make_panama_precip_chart(df: pd.DataFrame) -> go.Figure:
    """Daily catchment rainfall against its day-of-year normal.

    Bars carry anomaly polarity — blue above normal, red below — so the wet/dry
    read comes straight off the bar colour, with the gray ±1σ band showing how
    much day-to-day scatter is ordinary for the season.
    """
    fig = _base_fig("Canal Watershed — Daily Precipitation vs Normal", height=330)

    act = df[df["source"] == "actual"].sort_values("delivery_date")
    fcst = df[df["source"] == "forecast"].sort_values("delivery_date")

    clim = df.dropna(subset=["precip_clim_mm"]).sort_values("delivery_date")
    if not clim.empty:
        std = clim["precip_clim_std_mm"].fillna(0)
        fig.add_trace(go.Scatter(
            x=clim["delivery_date"], y=(clim["precip_clim_mm"] + std),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=clim["delivery_date"], y=(clim["precip_clim_mm"] - std).clip(lower=0),
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor=_hex_to_rgba(INK_MUTED, 0.18), name="Normal ± 1σ", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=clim["delivery_date"], y=clim["precip_clim_mm"], name="Normal (2000-24)",
            mode="lines", line=dict(color=INK_MUTED, width=1.2, dash="dash"),
        ))

    if not act.empty:
        dev = act["precip_mm"] - act["precip_clim_mm"]
        fig.add_trace(go.Bar(
            x=act["delivery_date"], y=act["precip_mm"], name="Actual",
            marker_color=[DIV_NEG if v < 0 else DIV_POS for v in dev.fillna(0)],
            marker_line_width=0, opacity=0.85,
            hovertemplate="%{x|%d %b} · %{y:.1f} mm<extra></extra>",
        ))
    if not fcst.empty:
        fig.add_trace(go.Bar(
            x=fcst["delivery_date"], y=fcst["precip_mm"], name="Forecast",
            marker_color=CAT_ORANGE, marker_line_width=0, opacity=0.75,
            hovertemplate="%{x|%d %b} · %{y:.1f} mm (fcst)<extra></extra>",
        ))

    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title_text="mm/day")
    return fig


def make_panama_precip_departure_chart(df: pd.DataFrame) -> go.Figure:
    """Running mm ahead of / behind normal — the number the lake actually integrates.

    A single wet day moves the daily chart and nothing else; this line is what
    translates into storage, so a sustained slope down is the drought signal.
    """
    fig = _base_fig("Canal Watershed — Cumulative Departure from Normal", height=330)

    d = df.dropna(subset=["precip_mm", "precip_clim_mm"]).sort_values("delivery_date")
    act = d[d["source"] == "actual"]
    fcst = d[d["source"] == "forecast"]

    end_cum = 0.0
    if not act.empty:
        cum = (act["precip_mm"] - act["precip_clim_mm"]).cumsum()
        end_cum = float(cum.iloc[-1])
        fig.add_trace(go.Scatter(
            x=act["delivery_date"], y=cum.values, name="Cumulative departure",
            mode="lines", line=dict(color=CAT_BLUE, width=2.5),
            fill="tozeroy", fillcolor=_hex_to_rgba(CAT_BLUE, 0.10),
            hovertemplate="%{x|%d %b} · %{y:+.0f} mm vs normal<extra></extra>",
        ))

    if not fcst.empty:
        cum_f = end_cum + (fcst["precip_mm"] - fcst["precip_clim_mm"]).cumsum()
        # Bridge off the last actual so the forecast leg starts on the line
        # rather than floating at its own first value.
        xs = list(fcst["delivery_date"])
        ys = list(cum_f.values)
        if not act.empty:
            xs = [act["delivery_date"].iloc[-1]] + xs
            ys = [end_cum] + ys
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name="Forecast (extended)", mode="lines",
            line=dict(color=CAT_ORANGE, width=2, dash="dash"),
            hovertemplate="%{x|%d %b} · %{y:+.0f} mm vs normal (fcst)<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=BASELINE, line_width=1.2)
    fig.update_yaxes(title_text="Departure (mm)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PADD 3 COASTAL TEMPERATURES
# ══════════════════════════════════════════════════════════════════════════════

def make_padd3_temp_chart(df: pd.DataFrame, cities: list[str] | None = None) -> go.Figure:
    """Temperature timeseries for PADD 3 coastal export terminals."""
    fig = _base_fig("PADD 3 Coastal Temperatures", height=440)

    if cities is None:
        cities = df["city"].unique().tolist()

    for city in cities:
        cdf = df[df["city"] == city]
        color = PADD3_COASTAL_COLORS.get(city, INK_MUTED)
        act = cdf[cdf["source"] == "actual"]
        if not act.empty:
            fig.add_trace(go.Scatter(
                x=act["delivery_date"], y=act["temperature_c"], name=city, mode="lines",
                line=dict(color=color, width=2),
            ))
        fct = cdf[cdf["source"] == "forecast"]
        if not fct.empty:
            fig.add_trace(go.Scatter(
                x=fct["delivery_date"], y=fct["temperature_c"], name=f"{city} (fcst)", mode="lines",
                line=dict(color=color, width=2, dash="dash"), showlegend=False,
            ))

    fig.add_hline(y=FROST_THRESHOLD_C, line_dash="dot", line_color=STATUS_WARNING,
                  annotation_text=f"Frost advisory ({FROST_THRESHOLD_C:.0f}°C)", annotation_position="bottom right")
    fig.add_hline(y=FREEZE_THRESHOLD_C, line_dash="dash", line_color=STATUS_CRITICAL,
                  annotation_text=f"Freeze ({FREEZE_THRESHOLD_C:.0f}°C)", annotation_position="bottom right")
    fig.update_yaxes(title_text="Temperature (°C)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: HURRICANES
# ══════════════════════════════════════════════════════════════════════════════

def make_hurricane_map(storms: list[dict], region_name: str, bbox: dict, pad_deg: float = 8.0) -> go.Figure:
    """Map of active tropical cyclones, zoomed to a single basin's bbox.

    A world-view map renders any single storm as a barely visible dot at
    this zoom level traders actually care about (is something bearing down
    on the Gulf coast / NW Australia right now); one focused map per basin
    reads far better than one global map with two faraway hot spots.
    """
    fig = go.Figure()
    region_storms = [s for s in storms if s["basin"] == region_name]

    if not region_storms:
        # Plotly only renders the geo subplot when a trace references it —
        # with zero storms there'd be no trace at all and the map would
        # silently fall back to a blank cartesian grid. An invisible marker
        # at the bbox center keeps the actual map (coastlines, ocean) showing.
        fig.add_trace(go.Scattergeo(
            lon=[(bbox["lon_min"] + bbox["lon_max"]) / 2],
            lat=[(bbox["lat_min"] + bbox["lat_max"]) / 2],
            mode="markers", marker=dict(size=0.001, opacity=0),
            showlegend=False, hoverinfo="skip",
        ))

    for storm in region_storms:
        color = STORM_RAMP.get(storm["category"], INK_MUTED)

        observed = storm.get("observed_track", [])
        if observed:
            fig.add_trace(go.Scattergeo(
                lon=[p["lon"] for p in observed] + [storm["lon"]],
                lat=[p["lat"] for p in observed] + [storm["lat"]],
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                opacity=0.55,
                showlegend=False, hoverinfo="skip",
            ))

        track = storm.get("forecast_track", [])
        if track:
            fig.add_trace(go.Scattergeo(
                lon=[storm["lon"]] + [p["lon"] for p in track],
                lat=[storm["lat"]] + [p["lat"] for p in track],
                mode="lines+markers",
                line=dict(color=color, width=1.5, dash="dot"),
                marker=dict(size=[10] + [6] * len(track), color=color, opacity=0.75),
                text=["Now"] + [f"+{p['hours']}h" for p in track],
                hovertemplate="%{text}<br>%{lat:.1f}, %{lon:.1f}<extra></extra>",
                showlegend=False,
            ))
        wind = storm["wind_kt"]
        size = max(14, min(28, 10 + wind // 5))
        fig.add_trace(go.Scattergeo(
            lon=[storm["lon"]], lat=[storm["lat"]],
            mode="markers+text",
            marker=dict(size=size, color=color, line=dict(width=1.5, color="#0b0b0b")),
            text=[storm["name"]], textposition="top center",
            textfont=dict(size=10, color="#0b0b0b"),
            name=f"{storm['name']} — {storm['category']}, {wind} kt [{storm['basin']}]",
            hovertemplate=(
                f"<b>{storm['name']}</b> ({storm['classification']})<br>"
                f"Basin: {storm['basin']}<br>Category: {storm['category']}<br>"
                f"Wind: {wind} kt<br>Pressure: {storm.get('pressure_mb', 'N/A')} mb<br>"
                f"Last update: {storm.get('last_update', 'N/A')}<extra></extra>"
            ),
        ))

    geo = dict(GEO_LAYOUT)
    geo.update(
        projection_type="mercator",
        lonaxis=dict(range=[bbox["lon_min"] - pad_deg, bbox["lon_max"] + pad_deg]),
        lataxis=dict(range=[bbox["lat_min"] - pad_deg, bbox["lat_max"] + pad_deg]),
    )
    fig.update_layout(
        geo=geo,
        title=dict(text=region_name, font=dict(size=13, color="#0b0b0b")),
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=0, r=0, t=36, b=0),
        font=dict(family="Inter, sans-serif", color="#0b0b0b", size=11),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)", bordercolor=GRIDLINE, borderwidth=1,
            font=dict(size=10, color="#374151"), x=0.01, y=0.01, xanchor="left", yanchor="bottom",
        ),
    )
    if not region_storms:
        fig.add_annotation(
            text="No active tropical cyclones in this region",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=INK_MUTED),
        )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: NORTH AFRICA WIND
# ══════════════════════════════════════════════════════════════════════════════

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r, g, b, alpha)' for Plotly fill colors."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def make_wind_corridor_chart(df: pd.DataFrame, cities: list[str] | None = None) -> go.Figure:
    """ENS P50 wind speed per city with P10–P90 shaded spread band.

    Solid line = actual (past delivery dates from latest ENS run).
    Dashed line = forecast (future delivery dates). The shaded band covers the
    10th–90th percentile of the 50-member ensemble — wider band = more uncertainty.
    """
    fig = _base_fig("North Africa Shipping Corridor — Wind Speed (100m ENS P10/P50/P90)", height=440)

    if cities is None:
        cities = df["city"].unique().tolist()

    for city in cities:
        cdf = df[df["city"] == city].sort_values("delivery_date")
        if cdf.empty:
            continue
        color = NAFRICA_WIND_COLORS.get(city, INK_MUTED)
        fill = _hex_to_rgba(color, 0.12)

        # P10–P90 band: invisible lower bound, filled upper bound
        fig.add_trace(go.Scatter(
            x=cdf["delivery_date"], y=cdf["wind_p10"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip", name=f"_{city}_p10",
        ))
        fig.add_trace(go.Scatter(
            x=cdf["delivery_date"], y=cdf["wind_p90"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill,
            showlegend=False, hoverinfo="skip", name=f"_{city}_p90",
        ))

        # P50 median — split actual (solid) vs forecast (dashed)
        act = cdf[cdf["source"] == "actual"]
        fct = cdf[cdf["source"] == "forecast"]
        if not act.empty:
            fig.add_trace(go.Scatter(
                x=act["delivery_date"], y=act["wind_p50"],
                name=city, mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{city}</b><br>%{{x|%d %b}}<br>"
                    "P50: %{y:.1f} m/s<extra></extra>"
                ),
            ))
        if not fct.empty:
            fig.add_trace(go.Scatter(
                x=fct["delivery_date"], y=fct["wind_p50"],
                name=f"{city} (fcst)", mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False,
                hovertemplate=(
                    f"<b>{city}</b><br>%{{x|%d %b}}<br>"
                    "P50: %{y:.1f} m/s<extra></extra>"
                ),
            ))

    fig.add_hline(y=WIND_CAUTION_MS, line_dash="dot", line_color=INK_MUTED,
                  annotation_text=f"Caution ({WIND_CAUTION_MS:.0f} m/s)")
    fig.add_hline(y=WIND_WARNING_MS, line_dash="dash", line_color=STATUS_WARNING,
                  annotation_text=f"Warning ({WIND_WARNING_MS:.0f} m/s)")
    fig.add_hline(y=WIND_GALE_MS, line_dash="solid", line_color=STATUS_CRITICAL,
                  annotation_text=f"Gale ({WIND_GALE_MS:.0f} m/s)")

    fig.update_yaxes(title_text="Wind Speed (m/s)")
    return fig


def make_wind_exceedance_chart(df: pd.DataFrame) -> go.Figure:
    """Ensemble risk: % of members ≥ warning threshold per city per forecast day.

    A line above 50% means the majority of the ensemble predicts disruptive winds.
    Even 10–25% signals meaningful risk for loading/scheduling decisions.
    """
    fig = _base_fig(
        f"Ensemble Risk ≥{WIND_WARNING_MS:.0f} m/s (‰ of members)", height=300
    )

    fcst = df[df["source"] == "forecast"]
    for city in fcst["city"].unique():
        cdf = fcst[fcst["city"] == city].sort_values("delivery_date")
        if cdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=cdf["delivery_date"], y=cdf["pct_above_warning"],
            name=city, mode="lines+markers",
            line=dict(color=NAFRICA_WIND_COLORS.get(city, INK_MUTED), width=1.5),
            marker=dict(size=5),
            hovertemplate=(
                f"<b>{city}</b><br>%{{x|%d %b}}<br>Risk: %{{y:.0f}}%<extra></extra>"
            ),
        ))

    fig.add_hline(y=50, line_dash="dot", line_color=INK_MUTED,
                  annotation_text="50% (majority)", annotation_position="right")
    fig.update_yaxes(
        title_text=f"% Members ≥ {WIND_WARNING_MS:.0f} m/s",
        range=[0, 100],
    )
    return fig
