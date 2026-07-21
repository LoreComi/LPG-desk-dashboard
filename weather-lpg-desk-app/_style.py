"""Shared visual styling — LPG Desk Weather Dashboard.

Design direction: dark, high-contrast institutional trading-desk theme.
Inspired by Bloomberg Terminal / Refinitiv Eikon aesthetics. Color is assigned
by job — categorical for series identity, sequential for magnitude, diverging
for anomaly polarity, status for alert state.
Never mix jobs: a category never borrows a status color and vice versa.
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS — DARK INSTITUTIONAL THEME
# ══════════════════════════════════════════════════════════════════════════════

# Chart / page surfaces
SURFACE = "#0f1117"          # deep charcoal chart bg
PAGE_PLANE = "#0a0c10"       # app background (near-black)
INK_PRIMARY = "#e8eaed"      # primary text (off-white)
INK_SECONDARY = "#9aa0a6"    # secondary text (silver)
INK_MUTED = "#5f6368"        # muted labels (mid-gray)
GRIDLINE = "#1e2128"         # subtle grid on dark
BASELINE = "#2d323b"         # zero-line / baseline
BORDER = "rgba(255,255,255,0.06)"
BORDER_BRIGHT = "rgba(255,255,255,0.12)"

# Categorical — tuned for dark backgrounds (brighter, more saturated)
CAT_BLUE = "#4d9ff7"
CAT_GREEN = "#34d399"
CAT_MAGENTA = "#f472b6"
CAT_YELLOW = "#fbbf24"
CAT_AQUA = "#22d3ee"
CAT_ORANGE = "#fb923c"
CAT_VIOLET = "#a78bfa"
CAT_RED = "#f87171"
CATEGORICAL = [CAT_BLUE, CAT_GREEN, CAT_MAGENTA, CAT_YELLOW, CAT_AQUA, CAT_ORANGE, CAT_VIOLET, CAT_RED]

# Sequential (magnitude) — single hue, dark bg optimized
SEQ_BLUE = ["#1e3a5f", "#2563a8", "#3b82f6", "#60a5fa", "#93c5fd", "#dbeafe"]

# Diverging (polarity) — blue (wet/positive) <-> red (dry/negative)
DIV_NEG = "#f87171"       # dry / below normal
DIV_MID = "#1e2128"
DIV_POS = "#4d9ff7"       # wet / above normal

# Status — vibrant on dark, never reused for series identity
STATUS_GOOD = "#34d399"
STATUS_WARNING = "#fbbf24"
STATUS_SERIOUS = "#fb923c"
STATUS_CRITICAL = "#ef4444"

# Ordinal storm-severity ramp (TD -> Cat 5): one hue family, monotone lightness
STORM_RAMP = {
    "TD": "#5f6368", "TS": "#94a3b8",
    "Cat 1": "#fca5a5", "Cat 2": "#f87171", "Cat 3": CAT_RED,
    "Cat 4": "#dc2626", "Cat 5": "#991b1b",
}


CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg-app: {PAGE_PLANE};
    --bg-surface: {SURFACE};
    --bg-card: #f8f8f6;
    --border: {BORDER};
    --border-bright: {BORDER_BRIGHT};
    --text-primary: {INK_PRIMARY};
    --text-secondary: {INK_SECONDARY};
    --text-muted: {INK_MUTED};
    --accent: {CAT_BLUE};
    --good: {STATUS_GOOD};
    --warning: {STATUS_WARNING};
    --serious: {STATUS_SERIOUS};
    --critical: {STATUS_CRITICAL};
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-card: 0 1px 6px rgba(11,11,11,0.06), 0 0 0 1px rgba(11,11,11,0.05);
    --shadow-pop: 0 6px 24px rgba(11,11,11,0.12);
}}

.stApp {{
    background: var(--bg-app) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}}
.stApp > header {{ background: transparent !important; }}

h1 {{
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    font-size: 1.7rem !important;
    margin-bottom: 0 !important;
}}
h1::after {{
    content: "";
    display: inline-block;
    width: 7px; height: 7px;
    margin-left: 8px;
    background: var(--accent);
    border-radius: 2px;
    vertical-align: 5px;
}}
h2, h3 {{ color: var(--text-primary) !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }}
h4 {{
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 22px !important;
    margin-bottom: 4px !important;
}}
p, .stMarkdown, span, label {{ color: var(--text-primary); }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--text-muted) !important; font-size: 0.80rem !important; }}

.main .block-container {{
    background: var(--bg-surface);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-card);
    padding: 24px 28px 32px !important;
    margin-top: 16px !important;
}}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: rgba(11,11,11,0.05);
    border-radius: var(--radius-lg);
    padding: 4px;
    gap: 2px;
}}
[data-testid="stTabs"] button {{
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    border-radius: var(--radius-md) !important;
    background: transparent !important;
    border: none !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    background: var(--bg-surface) !important;
    color: var(--accent) !important;
    box-shadow: var(--shadow-card) !important;
    font-weight: 600 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: transparent !important; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none !important; }}

/* KPI cards */
.kpi-card {{
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: var(--shadow-card);
}}
.kpi-card-warm  {{ border-top: 2px solid {CAT_ORANGE}; }}
.kpi-card-cool  {{ border-top: 2px solid {CAT_BLUE}; }}
.kpi-card-neutral {{ border-top: 2px solid var(--text-muted); }}
.kpi-card-good     {{ border-top: 2px solid var(--good); }}
.kpi-card-warning  {{ border-top: 2px solid var(--warning); }}
.kpi-card-critical {{ border-top: 2px solid var(--critical); }}
.kpi-label {{
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text-primary) !important;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
}}
.kpi-delta {{ font-size: 0.72rem; margin-top: 4px; font-weight: 500; }}
.kpi-delta-up    {{ color: {CAT_RED} !important; }}
.kpi-delta-down  {{ color: {CAT_BLUE} !important; }}
.kpi-delta-flat  {{ color: var(--text-muted) !important; }}

.kpi-rank {{
    display: inline-block;
    margin-top: 8px;
    padding: 2px 9px;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-secondary);
    background: rgba(11,11,11,0.05);
    border-radius: 999px;
}}

/* Groups a KPI row + its chart(s) into one visual card per entity */
.section-block {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 14px 16px 6px;
    margin-bottom: 18px;
}}

/* Status alert banners — icon + label, never color alone */
.status-banner {{
    border-radius: var(--radius-md);
    padding: 8px 14px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0 10px 0;
}}
.status-banner-critical {{ background: #fdecec; color: #7a1f1f; border: 1px solid rgba(208,59,59,0.35); }}
.status-banner-warning  {{ background: #fef6e6; color: #7a5a00; border: 1px solid rgba(250,178,25,0.45); }}
.status-banner-good     {{ background: #eafbea; color: #0f5c0f; border: 1px solid rgba(12,163,12,0.30); }}

[data-testid="stDataFrame"] {{
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}}
.stPlotlyChart {{
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-card);
    border: 1px solid var(--border);
    background: var(--bg-surface);
}}
hr {{ border-color: var(--border) !important; }}

.stMultiSelect [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div {{
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-md) !important;
}}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, sans-serif", color=INK_PRIMARY, size=12),
    margin=dict(l=55, r=20, t=50, b=70),
    legend=dict(
        bgcolor="rgba(22,25,32,0.92)",
        bordercolor=GRIDLINE,
        borderwidth=1,
        font=dict(size=11, color=INK_SECONDARY),
        orientation="h",
        yanchor="top",
        y=-0.22,
        xanchor="center",
        x=0.5,
    ),
    xaxis=dict(
        gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=GRIDLINE,
        tickfont=dict(color=INK_MUTED, size=11), title_font=dict(color=INK_SECONDARY),
    ),
    yaxis=dict(
        gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=GRIDLINE,
        tickfont=dict(color=INK_MUTED, size=11), title_font=dict(color=INK_SECONDARY),
    ),
)

# Geo (map) chrome — dark theme for Scattergeo charts
GEO_LAYOUT = dict(
    projection_type="natural earth",
    showland=True, landcolor="#1a1d24",
    showocean=True, oceancolor="#0d1117",
    showcountries=True, countrycolor="#2d323b",
    showcoastlines=True, coastlinecolor=INK_MUTED,
    showlakes=False,
    showframe=False,
    resolution=50,
    bgcolor="rgba(0,0,0,0)",
)
