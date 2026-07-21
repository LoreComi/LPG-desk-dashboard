"""
LPG market AI brief — 6-agent architecture.

Flow
----
  Python data collectors build compact, structured context documents.
  Five independent specialist LLM calls each analyse one domain.
  One synthesis LLM call reads all five findings and writes 4-6 trader bullets.

Agents
------
  1. hurricane    — storm threats to LPG export terminals (PADD 3 Gulf Coast)
  2. us_hdd       — US propane/LPG heating demand by PADD
  3. asia_hdd     — Asia LPG import demand (Japan, Korea, China heating)
  4. india_precip — India monsoon and LPG import logistics
  5. panama_canal — Gatun Lake / Panama Canal VLGC transit constraints
  6. synthesis    — combines all five findings into an LPG market brief
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ── System prompts ─────────────────────────────────────────────────────────────

_SYS_HURRICANE = """\
You are an LPG market analyst specialising in supply-chain disruption from tropical storms.

Key LPG export terminals and infrastructure (US Gulf Coast / PADD 3):
  Houston Ship Channel — Enterprise Products Partners (world's largest LPG export terminal)
  Freeport, TX — multiple NGL fractionators and export docks
  Sabine Pass / Port Arthur — Targa Resources, Energy Transfer
  Morgan City / Cameron, LA — LPG barge and vessel loading
  Corpus Christi, TX — EPIC NGL pipeline terminus, VLGC loading

Critical LPG shipping lanes:
  US Gulf → Panama Canal → NE Asia (the dominant VLGC route, ~60% of US LPG exports)
  US Gulf → Cape of Good Hope → Asia (alternative when Canal is congested)
  Middle East → India / NE Asia (competing supply from Qatar, Saudi Arabia)

Given the list of active tropical cyclones:
1. Identify any storm threatening an LPG export terminal or Gulf shipping lanes.
2. State the directional impact on Mont Belvieu propane, FEI (Far East Index), and ARB spreads.
3. If no relevant LPG-market threat exists, say so in ONE concise line.

Format: each bullet starts with "- ". Name the storm, intensity, location,
and the specific terminal or route at risk.

Last line only — write exactly one of:
SIGNAL: BULLISH  (storm threatens LPG export terminal or key shipping lane — bullish for prices)
SIGNAL: BEARISH  (no relevant LPG-market threat from current storms)
SIGNAL: NEUTRAL  (minor or uncertain LPG-market impact)
"""

_SYS_US_HDD = """\
You are a US propane/LPG demand analyst. HDD = max(18°C − T_mean, 0).
High HDD = cold weather = more residential propane demand for heating.

Key context:
  PADD 1 (Northeast): largest residential propane market in the US (>60% of winter propane demand)
  PADD 2 (Midwest): second-largest propane heating market (rural heating)
  PADD 3 (South Central): export hub — cold snaps can shift propane from export to domestic
  PADD 4/5: marginal heating demand

Market impact:
  Above-normal HDD (cold) → propane stock draws → bullish Mont Belvieu, narrows ARB
  Below-normal HDD (mild) → propane stock builds → bearish domestic, widens ARB (more for export)

Given cumulative season HDD data by PADD (actual vs normal, rank among last 10 winters):
- Focus on PADD 1 and PADD 2 (heating demand centers)
- State whether the current winter is running above/below normal and by how much
- If off-season (Apr-Sep), note that heating demand is minimal

Write 1-2 bullet points. Format: each bullet starts with "- ".

Last line only — write exactly one of:
SIGNAL: BULLISH  (above-normal cold driving propane demand — bullish Mont Belvieu)
SIGNAL: BEARISH  (below-normal cold / mild winter — bearish propane demand)
SIGNAL: NEUTRAL  (near-normal or off-season, marginal heating demand signal)
"""

_SYS_ASIA_HDD = """\
You are an Asia-Pacific LPG demand analyst.
HDD = max(18°C − T_mean, 0). In NE Asia, LPG is used for residential heating
and petrochemical feedstock. Japan, South Korea, and China together import ~50 Mt/year LPG.

Key dynamics:
  Japan: 2nd largest LPG importer; residential + city gas supplement (winter peak)
  South Korea: large petchem sector (PDH plants) + residential heating
  China North: residential LPG heating in cities without piped gas; PDH capacity booming
  China East/South: PDH feedstock demand (propane → propylene)

Market impact:
  High HDD in NE Asia → residential LPG demand up → FEI (Far East Index) bid up
    → wider US-Asia ARB → more US exports → bullish Mont Belvieu
  Low HDD → reduced Asian spot demand → FEI soft → ARB narrows

Given Asia HDD season data (cumulative, vs normal, rank):
- Treat Japan/Korea separately from China regions
- Focus on the forward signal (is it getting colder or milder?)

Write 2-3 bullet points. Format: each bullet starts with "- ".

Last line only — write exactly one of:
SIGNAL: BULLISH  (above-normal cold driving Asian LPG heating demand — bullish FEI)
SIGNAL: BEARISH  (below-normal cold — Asian LPG demand soft)
SIGNAL: NEUTRAL  (near-normal or off-season, marginal signal)
"""

_SYS_INDIA_PRECIP = """\
You are an India LPG market analyst specialising in monsoon logistics.

India context:
  World's #1 LPG importer (~20 Mt/year, mostly Middle East + US Gulf)
  Import terminals: Kandla, Mumbai (JNPT), Mangalore, Vizag, Haldia
  Monsoon season (Jun–Sep): heavy rainfall disrupts port operations and inland transport
  India subsidises LPG (PMUY programme) — demand is price-inelastic, supply-constrained

Key relationships:
  Above-normal monsoon (wet) → port congestion, vessel demurrage, supply chain delays
    → bullish India CFR premiums, can slow VLGC turnaround
  Below-normal monsoon (dry) → smoother port ops, but agricultural stress can lift rural LPG demand
  Off-monsoon: precipitation has minimal LPG market impact

Given India subregion precipitation anomaly data (actual vs climatology, % deviation):
- Northwest and West Coast subregions cover Kandla/Mumbai (key import terminals)
- South covers Mangalore/Kochi terminals
- State the directional impact on India CFR premiums and VLGC scheduling

Write 1-2 bullet points. Format: each bullet starts with "- ".

Last line only — write exactly one of:
SIGNAL: BULLISH  (monsoon disruption tightening India LPG supply — bullish premiums)
SIGNAL: BEARISH  (smooth logistics / weak demand signal)
SIGNAL: NEUTRAL  (off-monsoon or mixed signal, minimal LPG market impact)
"""

_SYS_PANAMA = """\
You are a shipping analyst specialising in Panama Canal transit for LPG VLGCs.

Panama Canal context for LPG:
  ~40% of US LPG exports transit the Panama Canal (US Gulf → NE Asia)
  VLGCs (Very Large Gas Carriers, 84,000 cbm) are the standard LPG vessel
  VLGCs transit through Neopanamax locks — draft-sensitive to Gatun Lake level
  Alternative route: Cape of Good Hope (+15-20 days voyage, +$2-4/mt freight)

Gatun Lake thresholds:
  ≥ 85 ft: normal operations, full VLGC draft
  82–85 ft: rate surcharges, some booking delays
  < 82 ft: significant draft restrictions — VLGCs may need to lighten or reroute

Market impact:
  Low Gatun → VLGC transit delays/surcharges → higher US→Asia freight
    → narrows ARB (makes US exports less competitive) → bullish FEI
  Normal Gatun → smooth transit, low freight → wide ARB, bearish FEI

Given current Gatun Lake level and trend:
- State the current level and whether it's at restriction/surcharge thresholds
- Assess impact on VLGC scheduling and US→Asia LPG freight

Write 1-2 bullet points. Format: each bullet starts with "- ".

Last line only — write exactly one of:
SIGNAL: BULLISH  (Canal restrictions tightening VLGC transit — bullish freight/FEI)
SIGNAL: BEARISH  (normal Canal ops — low freight, supportive ARB for US exports)
SIGNAL: NEUTRAL  (Canal situation stable, marginal LPG market impact)
"""

_SYS_SYNTHESIS = """\
You are the head LPG trader at a major energy company writing the daily brief.
You have received analyses from five specialist agents:
  1. Hurricane/storm risk to US Gulf LPG export terminals
  2. US HDD — propane/LPG heating demand signal
  3. Asia HDD — NE Asia LPG import demand signal
  4. India monsoon — India CFR premium and VLGC logistics
  5. Panama Canal — Gatun Lake VLGC transit constraints and freight

Write exactly 5-6 bullet points for the morning LPG market brief.

Requirements:
  - LEAD with the single most market-moving signal right now.
  - Cover BOTH the supply/logistics angle (freight, terminals, Canal) and the
    demand angle (US heating, Asia heating, India imports) — at least one bullet each.
  - One bullet must describe a cross-market interaction where two or more signals
    reinforce or offset each other (e.g., "Cold NE Asia + Canal restrictions = bullish FEI,
    compressed ARB").
  - One stability bullet: primary uncertainty or what would flip this view.
  - Reference specific price benchmarks: Mont Belvieu propane, FEI, US-Asia ARB.
  - No intro sentence. No headers. No conclusion. No numbering.
  - Bullets only, each starting with "- ".

Last line only — write: SIGNAL: BULLISH, SIGNAL: BEARISH, or SIGNAL: NEUTRAL
(net LPG market direction synthesising all five agent signals).
"""


# ── Data formatters (pure Python, no LLM) ─────────────────────────────────────

def _fmt_hurricanes(storms: list) -> str:
    ts = datetime.utcnow().strftime("%d %b %Y %H:%M UTC")
    if not storms:
        return f"No active tropical cyclones in any monitored basin. ({ts})"

    lines = [f"Active tropical cyclones — {ts}", f"Total storms: {len(storms)}", ""]
    for s in storms:
        name = s.get("name", "Unknown")
        basin = s.get("basin", "Unknown")
        cat = s.get("category", "TS")
        wind = s.get("wind_kt", 0)
        lat = s.get("lat", 0.0)
        lon = s.get("lon", 0.0)

        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        line = (f"  {name} ({basin}) — {cat}, {wind} kt, "
                f"{abs(lat):.1f}°{ns} {abs(lon):.1f}°{ew}")
        lines.append(line)

        track = s.get("forecast_track", [])
        for pt in track[:4]:
            h = pt.get("hours", 0)
            if h == 0:
                continue
            plat, plon = pt.get("lat", 0.0), pt.get("lon", 0.0)
            pns = "N" if plat >= 0 else "S"
            pew = "E" if plon >= 0 else "W"
            lines.append(f"    +{h:02d}h: {abs(plat):.1f}°{pns} {abs(plon):.1f}°{pew}")

    return "\n".join(lines)


def _fmt_us_hdd(padd_kpis: dict) -> str:
    """Format US HDD KPIs for the LLM.
    
    padd_kpis: {padd_name: {cumulative_to_date, delta_vs_normal, rank, rank_pool_size, as_of}}
    """
    ts = datetime.utcnow().strftime("%d %b %Y")
    lines = [f"US HDD Season Summary — {ts}", ""]
    
    if not padd_kpis:
        month = datetime.utcnow().month
        if month in (4, 5, 6, 7, 8, 9):
            lines.append("OFF-SEASON (Apr–Sep): HDD heating demand minimal. No winter signal.")
        else:
            lines.append("No HDD data available.")
        return "\n".join(lines)
    
    lines.append(f"{'PADD Region':<25}{'Cum HDD':>10}{'vs Normal':>12}{'Rank':>15}")
    for padd, kpi in sorted(padd_kpis.items()):
        if kpi is None:
            continue
        cum = kpi.get("cumulative_to_date", 0)
        delta = kpi.get("delta_vs_normal")
        rank = kpi.get("rank")
        pool = kpi.get("rank_pool_size", 0)
        
        delta_str = f"{delta:+.0f}" if delta is not None else "N/A"
        rank_str = f"{rank}/{pool} coldest" if rank and pool else "N/A"
        lines.append(f"  {padd:<23}{cum:>10.0f}{delta_str:>12}{rank_str:>15}")
    
    return "\n".join(lines)


def _fmt_asia_hdd(region_kpis: dict) -> str:
    """Format Asia HDD KPIs for the LLM."""
    ts = datetime.utcnow().strftime("%d %b %Y")
    lines = [f"Asia HDD Season Summary (population-weighted) — {ts}", ""]
    
    if not region_kpis:
        month = datetime.utcnow().month
        if month in (4, 5, 6, 7, 8, 9):
            lines.append("OFF-SEASON (Apr–Sep): NE Asia heating demand minimal.")
        else:
            lines.append("No Asia HDD data available.")
        return "\n".join(lines)
    
    lines.append(f"{'Region':<18}{'Cum HDD':>10}{'vs Normal':>12}{'Rank':>15}")
    for region, kpi in sorted(region_kpis.items()):
        if kpi is None:
            continue
        cum = kpi.get("cumulative_to_date", 0)
        delta = kpi.get("delta_vs_normal")
        rank = kpi.get("rank")
        pool = kpi.get("rank_pool_size", 0)
        
        delta_str = f"{delta:+.0f}" if delta is not None else "N/A"
        rank_str = f"{rank}/{pool} coldest" if rank and pool else "N/A"
        lines.append(f"  {region:<16}{cum:>10.0f}{delta_str:>12}{rank_str:>15}")
    
    return "\n".join(lines)


def _fmt_india_precip(precip_summary: dict) -> str:
    """Format India precipitation anomaly for the LLM.
    
    precip_summary: {subregion: {avg_actual_mm, avg_clim_mm, avg_deviation_pct, n_days}}
    """
    ts = datetime.utcnow().strftime("%d %b %Y")
    month = datetime.utcnow().month
    
    monsoon_active = month in (6, 7, 8, 9)
    season_ctx = ("MONSOON ACTIVE (Jun–Sep) — port and logistics disruption risk elevated"
                  if monsoon_active else
                  "Off-monsoon — precipitation has minimal LPG logistics impact")
    
    lines = [
        f"India Precipitation Anomaly — {ts}",
        f"Seasonal context: {season_ctx}",
        "",
    ]
    
    if not precip_summary:
        lines.append("No precipitation data available.")
        return "\n".join(lines)
    
    lines.append(f"{'Subregion':<15}{'Actual mm/d':>12}{'Normal mm/d':>12}{'Deviation':>12}")
    for region, d in sorted(precip_summary.items()):
        actual = d.get("avg_actual_mm", 0)
        clim = d.get("avg_clim_mm")
        dev = d.get("avg_deviation_pct")
        clim_str = f"{clim:.1f}" if clim is not None else "N/A"
        dev_str = f"{dev:+.0f}%" if dev is not None else "N/A"
        lines.append(f"  {region:<13}{actual:>12.1f}{clim_str:>12}{dev_str:>12}")
    
    lines.extend([
        "",
        "Key terminals by subregion:",
        "  Northwest / West Coast: Kandla, Mumbai (JNPT) — India's primary LPG import ports",
        "  South: Mangalore, Kochi — secondary import terminals",
        "  Northeast / Central: inland demand centres",
    ])
    return "\n".join(lines)


def _fmt_panama_canal(current_level_ft: float | None, trend_text: str = "") -> str:
    """Format Gatun Lake level for the LLM."""
    ts = datetime.utcnow().strftime("%d %b %Y")
    lines = [f"Panama Canal — Gatun Lake Status — {ts}", ""]
    
    if current_level_ft is None:
        lines.append("Gatun Lake level data unavailable.")
        return "\n".join(lines)
    
    lines.append(f"Current level: {current_level_ft:.1f} ft")
    
    if current_level_ft < 82.0:
        lines.append("Status: CRITICAL — below 82 ft, significant VLGC draft restrictions")
        lines.append("Impact: VLGCs cannot transit fully laden; lightering or Cape reroute required")
    elif current_level_ft < 85.0:
        lines.append("Status: CAUTION — 82-85 ft, rate surcharges and booking delays")
        lines.append("Impact: VLGC transit possible but with surcharges; scheduling uncertainty")
    else:
        lines.append("Status: NORMAL — ≥85 ft, full VLGC draft operations")
        lines.append("Impact: smooth transit, standard booking and freight rates")
    
    if trend_text:
        lines.append(f"Trend: {trend_text}")
    
    lines.extend([
        "",
        "Reference:",
        "  ~40% of US LPG exports transit Panama Canal (Gulf → NE Asia)",
        "  Cape of Good Hope alternative: +15-20 days, +$2-4/mt freight",
        "  VLGC capacity: 84,000 cbm (~44,000 mt LPG)",
    ])
    return "\n".join(lines)


# ── Signal extractor ──────────────────────────────────────────────────────────

def _extract_signal(text: str) -> tuple[str, str]:
    """Remove SIGNAL: line(s) from agent output and return (signal, cleaned_text)."""
    signal = "NEUTRAL"
    clean = []
    for line in text.strip().split("\n"):
        m = re.match(r"^SIGNAL:\s*(BULLISH|BEARISH|NEUTRAL)\s*$", line.strip(), re.IGNORECASE)
        if m:
            signal = m.group(1).upper()
        else:
            clean.append(line)
    while clean and not clean[-1].strip():
        clean.pop()
    return signal, "\n".join(clean)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_client(tenant_id: str, client_id: str, client_secret: str):
    from azure.identity import ClientSecretCredential
    from openai import AzureOpenAI

    cred = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )
    return AzureOpenAI(
        azure_endpoint="https://azure-oai-prod.openai.azure.com/",
        api_version="2024-12-01-preview",
        azure_ad_token_provider=lambda: cred.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token,
    )


def _llm(client, system: str, user_doc: str, model: str, max_tokens: int = 400) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_doc},
        ],
        temperature=0.3,
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def generate_lpg_brief(
    storms: list,
    padd_kpis: dict,
    asia_kpis: dict,
    india_precip_summary: dict,
    gatun_level_ft: float | None,
    gatun_trend: str,
    azure_tenant_id: str,
    azure_client_id: str,
    azure_client_secret: str,
    model: str = "gpt-4o",
    progress_cb=None,
) -> dict:
    """Run the 6-agent LPG brief pipeline.

    Parameters
    ----------
    storms               list of storm dicts from load_hurricane_data()
    padd_kpis            {padd_name: {cumulative_to_date, delta_vs_normal, rank, ...}}
    asia_kpis            {region: {cumulative_to_date, delta_vs_normal, rank, ...}}
    india_precip_summary {subregion: {avg_actual_mm, avg_clim_mm, avg_deviation_pct}}
    gatun_level_ft       current Gatun Lake level in feet (or None)
    gatun_trend          text describing recent trend (e.g., "rising +0.3 ft/week")
    azure_*              service-principal credentials for Azure OpenAI
    model                Azure OpenAI deployment name (default "gpt-4o")
    progress_cb          optional callable(str) for Streamlit progress messages

    Returns
    -------
    dict — hurricane, us_hdd, asia_hdd, india_precip, panama_canal, synthesis,
           signals, gatun_level_ft, generated_at
    """
    def _prog(msg: str):
        if progress_cb:
            progress_cb(msg)

    client = _build_client(azure_tenant_id, azure_client_id, azure_client_secret)

    # ── Format context documents ─────────────────────────────────────────────
    doc_hurricane   = _fmt_hurricanes(storms)
    doc_us_hdd      = _fmt_us_hdd(padd_kpis)
    doc_asia_hdd    = _fmt_asia_hdd(asia_kpis)
    doc_india       = _fmt_india_precip(india_precip_summary)
    doc_panama      = _fmt_panama_canal(gatun_level_ft, gatun_trend)

    # ── Specialist agents (independent, sequential) ───────────────────────────
    _prog("Hurricane analyst…")
    sig_hurricane, brief_hurricane = _extract_signal(
        _llm(client, _SYS_HURRICANE, doc_hurricane, model))

    _prog("US HDD / propane demand analyst…")
    sig_us_hdd, brief_us_hdd = _extract_signal(
        _llm(client, _SYS_US_HDD, doc_us_hdd, model))

    _prog("Asia HDD / LPG import demand analyst…")
    sig_asia_hdd, brief_asia_hdd = _extract_signal(
        _llm(client, _SYS_ASIA_HDD, doc_asia_hdd, model))

    _prog("India monsoon / LPG logistics analyst…")
    sig_india, brief_india = _extract_signal(
        _llm(client, _SYS_INDIA_PRECIP, doc_india, model))

    _prog("Panama Canal / VLGC transit analyst…")
    sig_panama, brief_panama = _extract_signal(
        _llm(client, _SYS_PANAMA, doc_panama, model))

    # ── Synthesis ─────────────────────────────────────────────────────────────
    _prog("Synthesis agent — writing LPG market brief…")
    synthesis_input = "\n\n".join([
        "=== HURRICANE / STORM SUPPLY RISK ===",           brief_hurricane,
        "=== US HDD / PROPANE HEATING DEMAND ===",         brief_us_hdd,
        "=== ASIA HDD / LPG IMPORT DEMAND ===",            brief_asia_hdd,
        "=== INDIA MONSOON / LPG LOGISTICS ===",           brief_india,
        "=== PANAMA CANAL / VLGC TRANSIT ===",             brief_panama,
    ])
    sig_synthesis, brief_synthesis = _extract_signal(
        _llm(client, _SYS_SYNTHESIS, synthesis_input, model, max_tokens=700))

    return {
        "hurricane":     brief_hurricane,
        "us_hdd":        brief_us_hdd,
        "asia_hdd":      brief_asia_hdd,
        "india_precip":  brief_india,
        "panama_canal":  brief_panama,
        "synthesis":     brief_synthesis,
        "signals": {
            "hurricane":    sig_hurricane,
            "us_hdd":       sig_us_hdd,
            "asia_hdd":     sig_asia_hdd,
            "india_precip": sig_india,
            "panama_canal": sig_panama,
            "synthesis":    sig_synthesis,
        },
        "gatun_level_ft": gatun_level_ft,
        "generated_at":   datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
    }


# ── Display helper ─────────────────────────────────────────────────────────────

def brief_to_html_bullets(brief_text: str) -> str:
    """Convert LLM markdown bullets to styled HTML divs."""
    parts = []
    for line in brief_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        parts.append(f'<div class="brief-bullet">{line}</div>')
    return "\n".join(parts)
