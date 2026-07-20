"""Chart builders — LPG Desk Weather Dashboard.

All functions return Plotly Figure objects styled with the dark-navy theme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from _config import (
    PADD_COLORS, ASIA_COLORS, PADD3_COASTAL_COLORS,
    WIND_CAUTION_MS, WIND_WARNING_MS, WIND_GALE_MS,
    FREEZE_THRESHOLD_C, FROST_THRESHOLD_C,
    HDD_BASE_TEMP,
)
from _style import PLOTLY_LAYOUT


def _base_fig(title: str = "", height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: US HDD SEASON CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def make_hdd_season_chart(df: pd.DataFrame, padd: str, normal_df: pd.DataFrame = None) -> go.Figure:
    """Cumulative HDD season chart for a single PADD."""
    fig = _base_fig(f"Cumulative HDD — {padd}")

    padd_data = df[df["padd_region"] == padd] if "padd_region" in df.columns else df

    for model in padd_data["model_label"].unique():
        mdf = padd_data[padd_data["model_label"] == model]
        color = PADD_COLORS.get(padd, "#94a3b8")
        dash = "solid" if "Actual" in model else "dash"
        fig.add_trace(go.Scatter(
            x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
            name=model, mode="lines",
            line=dict(color=color, dash=dash, width=2),
        ))

    if normal_df is not None and not normal_df.empty:
        ndf = normal_df[normal_df["padd_region"] == padd] if "padd_region" in normal_df.columns else normal_df
        if not ndf.empty:
            fig.add_trace(go.Scatter(
                x=ndf["delivery_date"] if "delivery_date" in ndf.columns else ndf.index,
                y=ndf["cumulative_hdd_normal"] if "cumulative_hdd_normal" in ndf.columns else ndf["daily_hdd_normal"].cumsum(),
                name="Normal", mode="lines",
                line=dict(color="#6b7280", dash="dot", width=1.5),
            ))

    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


def make_hdd_comparison_chart(df: pd.DataFrame, padd: str) -> go.Figure:
    """Run comparison — daily HDD bars for latest runs."""
    fig = _base_fig(f"Daily HDD Forecast — {padd}", height=350)
    padd_data = df[df["padd_region"] == padd] if "padd_region" in df.columns else df

    for rank in sorted(padd_data["run_rank"].dropna().unique()):
        rdf = padd_data[padd_data["run_rank"] == rank]
        label = rdf["run_label"].iloc[0] if not rdf.empty else f"Run {rank}"
        opacity = 1.0 if rank == 1 else 0.5
        fig.add_trace(go.Bar(
            x=rdf["delivery_date"], y=rdf["daily_hdd"],
            name=label, opacity=opacity,
        ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASIA HDD
# ══════════════════════════════════════════════════════════════════════════════

def make_asia_hdd_season_chart(df: pd.DataFrame, selected_regions: list[str]) -> go.Figure:
    """Cumulative HDD for selected Asia regions on one chart."""
    fig = _base_fig("Asia HDD Season — Cumulative")

    for region in selected_regions:
        rdf = df[df["region"] == region]
        # Only plot actuals + latest forecast
        for model in rdf["model_label"].unique():
            mdf = rdf[rdf["model_label"] == model]
            dash = "solid" if "Actual" in model else "dash"
            fig.add_trace(go.Scatter(
                x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
                name=f"{region} ({model})", mode="lines",
                line=dict(color=ASIA_COLORS.get(region, "#94a3b8"), dash=dash, width=2),
            ))

    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


def make_asia_hdd_bar_chart(df: pd.DataFrame, selected_regions: list[str]) -> go.Figure:
    """14-day forecast HDD total bar chart by Asia region."""
    fig = _base_fig("14-Day Forecast HDD Total", height=320)

    for region in selected_regions:
        rdf = df[(df["region"] == region) & (~df["model_label"].str.contains("Actual", na=False))]
        total = rdf["daily_hdd"].sum()
        fig.add_trace(go.Bar(
            x=[region], y=[total],
            name=region,
            marker_color=ASIA_COLORS.get(region, "#94a3b8"),
        ))

    fig.update_yaxes(title_text="HDD (°C-days)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: INDIA PRECIPITATION ANOMALY
# ══════════════════════════════════════════════════════════════════════════════

def make_india_precip_timeseries(df: pd.DataFrame) -> go.Figure:
    """Precipitation actual vs climatology timeseries by subregion."""
    fig = _base_fig("India Precipitation — Actual vs Climatology")

    colors = ["#dc2626", "#2563eb", "#d97706", "#059669", "#7c3aed"]
    for i, region in enumerate(df["subregion"].unique()):
        rdf = df[df["subregion"] == region]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=rdf["delivery_date"], y=rdf["precip_actual_mm"],
            name=f"{region} (actual)", mode="lines",
            line=dict(color=color, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=rdf["delivery_date"], y=rdf["precip_clim_mm"],
            name=f"{region} (normal)", mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False,
        ))

    fig.update_yaxes(title_text="Precipitation (mm/day)")
    return fig


def make_india_precip_deviation_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of precipitation deviation % by subregion."""
    fig = _base_fig("India Precipitation Deviation (%)", height=320)

    # Aggregate to latest available period
    latest = df.groupby("subregion").agg({"deviation_pct": "mean"}).reset_index()

    fig.add_trace(go.Bar(
        x=latest["subregion"], y=latest["deviation_pct"],
        marker_color=["#059669" if v >= 0 else "#dc2626" for v in latest["deviation_pct"]],
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_yaxes(title_text="Deviation (%)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GATUN LAKE ANALOG
# ══════════════════════════════════════════════════════════════════════════════

def make_gatun_analog_chart(hist_df: pd.DataFrame, analog_df: pd.DataFrame,
                           current_year: int) -> go.Figure:
    """Gatun Lake catchment cumulative precipitation — current year vs analogs."""
    fig = _base_fig("Gatun Lake Catchment — Cumulative Precipitation")

    # Current year (bold)
    cur = hist_df[hist_df["year"] == current_year]
    if not cur.empty:
        fig.add_trace(go.Scatter(
            x=cur["day_of_year"], y=cur["cumulative_precip_mm"],
            name=str(current_year), mode="lines",
            line=dict(color="#f59e0b", width=3),
        ))

    # Analog years (thinner, semi-transparent)
    analog_years = analog_df["analog_year"].unique() if not analog_df.empty else []
    palette = ["#60a5fa", "#34d399", "#a78bfa", "#fb923c", "#f87171"]
    for i, yr in enumerate(analog_years[:5]):
        adf = analog_df[analog_df["analog_year"] == yr]
        fig.add_trace(go.Scatter(
            x=adf["day_of_year"], y=adf["cumulative_precip_mm"],
            name=f"{yr} (analog)", mode="lines",
            line=dict(color=palette[i % len(palette)], width=1.5, dash="dash"),
            opacity=0.7,
        ))

    # Climatological mean
    clim = hist_df.groupby("day_of_year")["cumulative_precip_mm"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=clim["day_of_year"], y=clim["cumulative_precip_mm"],
        name="Climatology", mode="lines",
        line=dict(color="#6b7280", width=2, dash="dot"),
    ))

    fig.update_xaxes(title_text="Day of Year")
    fig.update_yaxes(title_text="Cumulative Precip (mm)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PADD 3 COASTAL TEMPERATURES
# ══════════════════════════════════════════════════════════════════════════════

def make_padd3_temp_chart(df: pd.DataFrame, cities: list[str] | None = None) -> go.Figure:
    """Temperature timeseries for PADD 3 coastal terminals."""
    fig = _base_fig("PADD 3 Coastal Temperatures")

    if cities is None:
        cities = df["city"].unique().tolist()

    for city in cities:
        cdf = df[df["city"] == city]
        color = PADD3_COASTAL_COLORS.get(city, "#94a3b8")
        # Actuals (solid)
        act = cdf[cdf["source"] == "actual"]
        if not act.empty:
            fig.add_trace(go.Scatter(
                x=act["delivery_date"], y=act["temperature_c"],
                name=city, mode="lines",
                line=dict(color=color, width=2),
            ))
        # Forecast (dashed, same color)
        fct = cdf[cdf["source"] == "forecast"]
        if not fct.empty:
            fig.add_trace(go.Scatter(
                x=fct["delivery_date"], y=fct["temperature_c"],
                name=f"{city} (fcst)", mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False,
            ))

    # Freeze threshold
    fig.add_hline(y=FREEZE_THRESHOLD_C, line_dash="dot", line_color="#93c5fd",
                  annotation_text="Freeze (0°C)", annotation_position="bottom right")
    fig.update_yaxes(title_text="Temperature (°C)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: HURRICANES
# ══════════════════════════════════════════════════════════════════════════════

def make_hurricane_track_map(df: pd.DataFrame) -> go.Figure:
    """Map of active tropical cyclone tracks."""
    fig = go.Figure()

    for storm in df["storm_name"].unique():
        sdf = df[df["storm_name"] == storm].sort_values("timestamp")
        fig.add_trace(go.Scattergeo(
            lon=sdf["longitude"], lat=sdf["latitude"],
            name=storm, mode="lines+markers",
            marker=dict(size=sdf["max_wind_kt"].clip(upper=100) / 10, opacity=0.7),
            line=dict(width=2),
        ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#1e293b",
        showocean=True, oceancolor="#0f172a",
        showcoastlines=True, coastlinecolor="#475569",
    )
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        title="Active Tropical Cyclones",
        height=480,
    )
    return fig


def make_ace_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Season ACE index comparison bar chart."""
    fig = _base_fig("Accumulated Cyclone Energy (ACE) — Season to Date", height=320)

    for basin in df["basin"].unique():
        bdf = df[df["basin"] == basin]
        latest = bdf[bdf["season_year"] == bdf["season_year"].max()]
        if latest.empty:
            continue
        fig.add_trace(go.Bar(
            x=[basin], y=latest["ace_index"].values,
            name=f"{basin} (actual)",
        ))
        fig.add_trace(go.Bar(
            x=[basin], y=latest["ace_normal"].values,
            name=f"{basin} (normal)",
            marker_color="#6b7280", opacity=0.5,
        ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="ACE Index")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: NORTH AFRICA WIND
# ══════════════════════════════════════════════════════════════════════════════

def make_wind_corridor_chart(df: pd.DataFrame, cities: list[str] | None = None) -> go.Figure:
    """Wind speed timeseries along North Africa corridor."""
    fig = _base_fig("North Africa Shipping Corridor — Wind Speed")

    if cities is None:
        cities = df["city"].unique().tolist()

    for city in cities:
        cdf = df[df["city"] == city]
        fig.add_trace(go.Scatter(
            x=cdf["delivery_date"], y=cdf["wind_speed_ms"],
            name=city, mode="lines",
            line=dict(width=1.5),
        ))

    # Threshold lines
    fig.add_hline(y=WIND_CAUTION_MS, line_dash="dot", line_color="#fbbf24",
                  annotation_text="Caution (10 m/s)")
    fig.add_hline(y=WIND_WARNING_MS, line_dash="dash", line_color="#f97316",
                  annotation_text="Warning (15 m/s)")
    fig.add_hline(y=WIND_GALE_MS, line_dash="solid", line_color="#dc2626",
                  annotation_text="Gale (20 m/s)")

    fig.update_yaxes(title_text="Wind Speed (m/s)")
    return fig


def make_wind_exceedance_chart(df: pd.DataFrame) -> go.Figure:
    """Days exceeding warning threshold in next 14d, by city."""
    fig = _base_fig("Forecast Days Above Warning Threshold (15 m/s)", height=320)

    fcst = df[df["source"] == "forecast"]
    exceed = fcst[fcst["wind_speed_ms"] >= WIND_WARNING_MS]
    counts = exceed.groupby("city").size().reset_index(name="days")

    fig.add_trace(go.Bar(
        x=counts["city"], y=counts["days"],
        marker_color="#f97316",
    ))
    fig.update_yaxes(title_text="Days")
    return fig
