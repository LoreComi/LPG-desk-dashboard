"""CSS & Plotly theming — LPG Desk Weather Dashboard."""
from __future__ import annotations

CUSTOM_CSS = """
<style>
/* Dark navy background */
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stSidebar"] { background: #1e293b; }
[data-testid="stHeader"] { background: transparent; }

/* Text */
body, .stMarkdown, .stText, p, span, label, h1, h2, h3, h4 {
    color: #e2e8f0 !important;
}

/* KPI cards */
.kpi-card {
    background: #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid #334155;
    min-height: 100px;
}
.kpi-card-warm { border-left: 4px solid #ef4444; }
.kpi-card-cool { border-left: 4px solid #3b82f6; }
.kpi-card-neutral { border-left: 4px solid #6b7280; }
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9 !important;
    margin-top: 4px;
}
.kpi-delta { font-size: 0.8rem; margin-top: 6px; }
.kpi-delta-up { color: #f87171 !important; }   /* warmer = red */
.kpi-delta-down { color: #60a5fa !important; } /* cooler = blue */
.kpi-delta-flat { color: #6b7280 !important; }

/* Tabs */
[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f1f5f9 !important;
    border-bottom: 2px solid #3b82f6;
}

/* Warning badges */
.wind-warning {
    background: #7c2d12;
    border-radius: 6px;
    padding: 4px 10px;
    color: #fbbf24 !important;
    font-weight: 600;
    font-size: 0.8rem;
    display: inline-block;
}
.freeze-alert {
    background: #1e3a5f;
    border-radius: 6px;
    padding: 4px 10px;
    color: #93c5fd !important;
    font-weight: 600;
    font-size: 0.8rem;
    display: inline-block;
}
</style>
"""

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1e293b",
    font=dict(color="#e2e8f0", size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
)
