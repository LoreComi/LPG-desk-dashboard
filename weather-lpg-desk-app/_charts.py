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
    NAFRICA_WIND_COLORS, GATUN_ANALOG_COLORS,
    GATUN_RATE_SURCHARGE_FT, GATUN_RESTRICTION_FT,
    WIND_CAUTION_MS, WIND_WARNING_MS, WIND_GALE_MS,
    FREEZE_THRESHOLD_C, FROST_THRESHOLD_C,
)
from _style import (
    PLOTLY_LAYOUT, GEO_LAYOUT, INK_MUTED, GRIDLINE,
    DIV_POS, DIV_NEG, STATUS_WARNING, STATUS_CRITICAL,
    STORM_RAMP, CAT_ORANGE,
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: US HDD SEASON CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def make_hdd_season_chart(df: pd.DataFrame, padd: str, normal_df: pd.DataFrame = None) -> go.Figure:
    """Cumulative HDD season chart for a single PADD — actual, forecast, normal."""
    fig = _base_fig(f"Cumulative HDD — {padd}")
    padd_data = df[df["padd_region"] == padd] if "padd_region" in df.columns else df
    color = PADD_COLORS.get(padd, INK_MUTED)

    for model in padd_data["model_label"].unique():
        mdf = padd_data[padd_data["model_label"] == model]
        dash = "solid" if "Actual" in model else "dash"
        fig.add_trace(go.Scatter(
            x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
            name=model, mode="lines",
            line=dict(color=color, dash=dash, width=2),
        ))

    if normal_df is not None and not normal_df.empty:
        ndf = normal_df[normal_df["padd_region"] == padd] if "padd_region" in normal_df.columns else normal_df
        if not ndf.empty:
            season_year = _season_start_year(padd_data["delivery_date"])
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
                                normal_df: pd.DataFrame = None) -> go.Figure:
    """Cumulative HDD for selected Asia regions — actual, forecast, normal."""
    fig = _base_fig("Asia HDD Season — Cumulative (population-weighted)")

    for region in selected_regions:
        rdf = df[df["region"] == region]
        color = ASIA_COLORS.get(region, INK_MUTED)
        for model in rdf["model_label"].unique():
            mdf = rdf[rdf["model_label"] == model]
            dash = "solid" if "Actual" in model else "dash"
            fig.add_trace(go.Scatter(
                x=mdf["delivery_date"], y=mdf["cumulative_hdd"],
                name=f"{region} ({model})", mode="lines",
                line=dict(color=color, dash=dash, width=2),
            ))
        if normal_df is not None and not normal_df.empty:
            ndf = normal_df[normal_df["region"] == region]
            if not ndf.empty:
                season_year = _season_start_year(rdf["delivery_date"])
                ndf = ndf.copy()
                ndf["date"] = _doy_to_season_date(ndf["day_of_year"], season_year)
                ndf = ndf.sort_values("date")   # season order (Oct1->Mar31), not raw day_of_year
                ndf["cumulative_hdd_normal"] = ndf["daily_hdd_normal"].cumsum()
                fig.add_trace(go.Scatter(
                    x=ndf["date"], y=ndf["cumulative_hdd_normal"],
                    name=f"{region} (normal)", mode="lines",
                    line=dict(color=color, dash="dot", width=1),
                    opacity=0.6, showlegend=False,
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
    """Diverging bar of latest precipitation deviation (%) by subregion."""
    fig = _base_fig("Precipitation Deviation vs Normal (%)", height=300)

    latest = df.sort_values("delivery_date").groupby("subregion").last().reset_index()
    fig.add_trace(go.Bar(
        x=latest["subregion"], y=latest["deviation_pct"],
        marker_color=[DIV_POS if v >= 0 else DIV_NEG for v in latest["deviation_pct"]],
        showlegend=False,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=INK_MUTED)
    fig.update_yaxes(title_text="Deviation (%)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GATUN LAKE — observed, official forecast, El Nino analog years
# ══════════════════════════════════════════════════════════════════════════════

def make_gatun_lake_chart(hist_df: pd.DataFrame, proj_df: pd.DataFrame,
                           analog_overlays: dict[int, pd.DataFrame]) -> go.Figure:
    """Observed level + official ACP forecast + El Nino analog-year overlays."""
    fig = _base_fig("Gatun Lake — Observed, Official Forecast & El Nino Analog Years", height=460)

    if not hist_df.empty:
        recent = hist_df[hist_df["date"] >= pd.Timestamp.today() - pd.DateOffset(days=365)]
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["level_ft"], name="Observed", mode="lines",
            line=dict(color=PADD_COLORS.get("PADD 1 - Northeast", "#2a78d6"), width=2.5),
        ))

    if not proj_df.empty:
        fig.add_trace(go.Scatter(
            x=proj_df["date"], y=proj_df["level_ft"], name="Official ACP Forecast", mode="lines",
            line=dict(color="#e34948", width=2, dash="dash"),
        ))

    for year, adf in sorted(analog_overlays.items()):
        fig.add_trace(go.Scatter(
            x=adf["date"], y=adf["level_ft"], name=f"{year} (El Nino analog)", mode="lines",
            line=dict(color=GATUN_ANALOG_COLORS.get(year, INK_MUTED), width=1.5, dash="dot"),
            opacity=0.85,
        ))

    fig.add_hline(y=GATUN_RATE_SURCHARGE_FT, line_color=STATUS_WARNING, line_dash="dash", line_width=1.5,
                  annotation_text=f"{GATUN_RATE_SURCHARGE_FT:.0f} ft — Increased Shipping Rates",
                  annotation_font_color="#7a5a00", annotation_position="bottom right")
    fig.add_hline(y=GATUN_RESTRICTION_FT, line_color=STATUS_CRITICAL, line_dash="dash", line_width=1.5,
                  annotation_text=f"{GATUN_RESTRICTION_FT:.0f} ft — Significant Restrictions",
                  annotation_font_color="#7a1f1f", annotation_position="bottom right")

    fig.update_yaxes(title_text="Lake Level (feet)")
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

def make_wind_corridor_chart(df: pd.DataFrame, cities: list[str] | None = None) -> go.Figure:
    """Wind speed timeseries along the North Africa shipping corridor."""
    fig = _base_fig("North Africa Shipping Corridor — Wind Speed", height=440)

    if cities is None:
        cities = df["city"].unique().tolist()

    for city in cities:
        cdf = df[df["city"] == city]
        color = NAFRICA_WIND_COLORS.get(city, INK_MUTED)
        act = cdf[cdf["source"] == "actual"]
        if not act.empty:
            fig.add_trace(go.Scatter(
                x=act["delivery_date"], y=act["wind_speed_ms"], name=city, mode="lines",
                line=dict(color=color, width=2),
            ))
        fct = cdf[cdf["source"] == "forecast"]
        if not fct.empty:
            fig.add_trace(go.Scatter(
                x=fct["delivery_date"], y=fct["wind_speed_ms"], name=f"{city} (fcst)", mode="lines",
                line=dict(color=color, width=2, dash="dash"), showlegend=False,
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
    """Forecast days above the warning threshold in the next 14 days, by port."""
    fig = _base_fig(f"Forecast Days Above Warning Threshold ({WIND_WARNING_MS:.0f} m/s)", height=300)

    fcst = df[df["source"] == "forecast"]
    exceed = fcst[fcst["wind_speed_ms"] >= WIND_WARNING_MS]
    counts = exceed.groupby("city").size().reset_index(name="days")

    fig.add_trace(go.Bar(x=counts["city"], y=counts["days"], marker_color=CAT_ORANGE, showlegend=False))
    fig.update_yaxes(title_text="Days")
    return fig
