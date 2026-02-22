"""
Octavian Professional Theme
Navy blue, white, black, gold, lavender color scheme with micro-animations.
Author: APB - Octavian Team
"""

import streamlit as st

COLORS = {
    "navy": "#0a1628",
    "navy_light": "#132240",
    "navy_mid": "#1a2d4a",
    "gold": "#c9a84c",
    "gold_light": "#d4b86a",
    "gold_dark": "#a08030",
    "white": "#f0f2f6",
    "white_soft": "#e0e4ec",
    "black": "#060d18",
    "lavender": "#9b8ec4",
    "lavender_light": "#b8aed6",
    "lavender_dark": "#7b6ea4",
    "success": "#4caf50",
    "danger": "#ef5350",
    "neutral": "#78909c",
    "text_primary": "#e8eaf0",
    "text_secondary": "#a0a8b8",
    "border": "#1e3050",
    "glass_bg": "rgba(19, 34, 64, 0.65)",
    "glass_border": "rgba(201, 168, 76, 0.12)",
}

PLOTLY_TEMPLATE = {
    "template": "plotly_dark",
    "paper_bgcolor": COLORS["navy"],
    "plot_bgcolor": COLORS["navy_light"],
    "font_color": COLORS["text_primary"],
}


def apply_theme():
    """Apply the full Octavian professional theme."""
    st.markdown(_get_theme_css(), unsafe_allow_html=True)


def _get_theme_css() -> str:
    return f"""
    <style>
    /* ========================================
       OCTAVIAN PROFESSIONAL THEME v4.0
       Micro-animations | Glassmorphism | Polish
       ======================================== */

    /* --- TYPOGRAPHY: Google Fonts --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* --- GLOBAL --- */
    .stApp {{
        background: linear-gradient(160deg, {COLORS['black']} 0%, {COLORS['navy']} 40%, {COLORS['navy_light']} 100%);
        color: {COLORS['text_primary']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* --- ANIMATED BACKGROUND GLOW --- */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(155,142,196,0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(201,168,76,0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(26,45,74,0.06) 0%, transparent 50%);
        animation: octavian-glow 12s ease-in-out infinite alternate;
    }}

    @keyframes octavian-glow {{
        0%   {{ opacity: 0.6; transform: scale(1.0); }}
        50%  {{ opacity: 1.0; transform: scale(1.02); }}
        100% {{ opacity: 0.7; transform: scale(1.0); }}
    }}

    /* Subtle grid overlay */
    .stApp::after {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        background-image:
            linear-gradient(rgba(201,168,76,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(201,168,76,0.015) 1px, transparent 1px);
        background-size: 60px 60px;
        animation: grid-drift 40s linear infinite;
    }}

    @keyframes grid-drift {{
        0%   {{ transform: translate(0, 0); }}
        100% {{ transform: translate(60px, 60px); }}
    }}

    /* --- MICRO-ANIMATION KEYFRAMES --- */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}

    @keyframes shimmer {{
        0%   {{ background-position: -200% center; }}
        100% {{ background-position: 200% center; }}
    }}

    @keyframes pulse-border {{
        0%, 100% {{ border-color: {COLORS['border']}; }}
        50%      {{ border-color: {COLORS['gold_dark']}; }}
    }}

    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-8px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['black']} 0%, {COLORS['navy']} 100%);
        border-right: 1px solid {COLORS['border']};
    }}

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {COLORS['gold']};
    }}

    /* --- HEADERS --- */
    h1 {{
        color: {COLORS['gold']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        border-bottom: 1px solid {COLORS['gold_dark']};
        padding-bottom: 8px;
        animation: fadeInUp 0.5s ease-out;
    }}

    h2 {{
        color: {COLORS['white']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        animation: fadeInUp 0.4s ease-out;
    }}

    h3 {{
        color: {COLORS['lavender_light']} !important;
        font-weight: 500 !important;
        animation: fadeInUp 0.35s ease-out;
    }}

    h4 {{
        color: {COLORS['gold_light']} !important;
        font-weight: 500 !important;
    }}

    /* --- METRICS with micro-animation --- */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, {COLORS['glass_bg']}, {COLORS['navy_mid']});
        border: 1px solid {COLORS['glass_border']};
        border-radius: 10px;
        padding: 14px 18px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.4s ease-out both;
    }}

    [data-testid="stMetric"]:hover {{
        border-color: {COLORS['gold']};
        box-shadow: 0 4px 20px rgba(201,168,76,0.15), 0 0 0 1px rgba(201,168,76,0.1);
        transform: translateY(-3px) scale(1.01);
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS['text_secondary']} !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS['white']} !important;
        font-family: 'JetBrains Mono', 'Consolas', monospace !important;
        font-weight: 600;
    }}

    [data-testid="stMetricDelta"] > div {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem;
    }}

    /* --- BUTTONS with micro-animations --- */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['navy_mid']}, {COLORS['navy_light']});
        color: {COLORS['gold_light']};
        border: 1px solid {COLORS['gold_dark']};
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.3px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    .stButton > button::after {{
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        width: 0; height: 0;
        background: rgba(201,168,76,0.15);
        border-radius: 50%;
        transition: width 0.4s ease, height 0.4s ease;
        transform: translate(-50%, -50%);
    }}

    .stButton > button:hover::after {{
        width: 300px;
        height: 300px;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']});
        color: {COLORS['black']};
        border-color: {COLORS['gold']};
        box-shadow: 0 4px 18px rgba(201,168,76,0.3);
        transform: translateY(-1px);
    }}

    .stButton > button:active {{
        transform: translateY(0px) scale(0.98);
        transition: transform 0.1s;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']});
        color: {COLORS['black']};
        border: none;
        font-weight: 600;
        box-shadow: 0 2px 12px rgba(201,168,76,0.2);
    }}

    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 6px 24px rgba(201,168,76,0.4);
        transform: translateY(-2px);
    }}

    /* --- TABS with slide animation --- */
    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS['navy']};
        border-bottom: 1px solid {COLORS['border']};
        gap: 0;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_secondary']};
        border-bottom: 2px solid transparent;
        padding: 10px 22px;
        font-size: 0.85rem;
        letter-spacing: 0.3px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['gold_light']};
        background: rgba(201,168,76,0.04);
    }}

    .stTabs [aria-selected="true"] {{
        color: {COLORS['gold']} !important;
        border-bottom-color: {COLORS['gold']} !important;
        background: transparent !important;
        font-weight: 600;
    }}

    /* --- DATAFRAMES --- */
    .stDataFrame {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        overflow: hidden;
        animation: fadeIn 0.3s ease-out;
    }}

    /* --- INPUTS with focus animation --- */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {{
        background: {COLORS['navy_light']} !important;
        color: {COLORS['text_primary']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }}

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {COLORS['gold']} !important;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
        outline: none;
    }}

    .stSelectbox > div > div {{
        background: {COLORS['navy_light']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px;
        transition: border-color 0.3s ease;
    }}

    .stSelectbox > div > div:hover {{
        border-color: {COLORS['gold_dark']} !important;
    }}

    /* --- EXPANDERS with smooth open --- */
    .streamlit-expanderHeader {{
        background: {COLORS['navy_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['gold_light']} !important;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {COLORS['gold_dark']};
        background: {COLORS['navy_mid']};
    }}

    details[open] .streamlit-expanderContent {{
        animation: fadeInUp 0.3s ease-out;
    }}

    /* --- CHAT --- */
    .stChatMessage {{
        background: {COLORS['navy_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        animation: fadeInUp 0.3s ease-out;
        transition: border-color 0.3s ease;
    }}

    .stChatMessage:hover {{
        border-color: {COLORS['glass_border']};
    }}

    /* --- RADIO SIDEBAR NAV with slide animation --- */
    .stRadio > div {{
        gap: 2px;
    }}

    .stRadio > div > label {{
        padding: 8px 14px;
        border-radius: 6px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 0.88rem;
        font-family: 'Inter', sans-serif;
        border-left: 2px solid transparent;
    }}

    .stRadio > div > label:hover {{
        background: {COLORS['navy_mid']};
        border-left-color: {COLORS['gold_dark']};
        padding-left: 18px;
    }}

    /* --- DIVIDERS --- */
    hr {{
        border-color: {COLORS['border']};
        opacity: 0.5;
    }}

    /* --- CAPTIONS --- */
    .stCaption {{
        color: {COLORS['text_secondary']} !important;
        font-size: 0.75rem;
    }}

    /* --- SCROLLBAR --- */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS['navy']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['gold_dark']};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['gold']};
    }}

    /* --- PLOTLY CHART CONTAINERS --- */
    .js-plotly-plot {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        animation: fadeIn 0.4s ease-out;
    }}

    /* --- LOADING SPINNER --- */
    .stSpinner > div {{
        border-color: {COLORS['gold']} transparent transparent transparent !important;
    }}

    /* --- FILE UPLOADER --- */
    [data-testid="stFileUploader"] {{
        border: 1px dashed {COLORS['border']};
        border-radius: 10px;
        padding: 14px;
        transition: all 0.3s ease;
    }}

    [data-testid="stFileUploader"]:hover {{
        border-color: {COLORS['gold_dark']};
        background: rgba(201,168,76,0.02);
    }}

    /* --- ALERTS --- */
    .stAlert {{
        border-radius: 8px;
        animation: fadeInUp 0.3s ease-out;
        backdrop-filter: blur(4px);
    }}

    /* --- COLUMNS: stagger animation --- */
    [data-testid="stHorizontalBlock"] > div {{
        animation: fadeInUp 0.4s ease-out both;
    }}

    [data-testid="stHorizontalBlock"] > div:nth-child(1) {{ animation-delay: 0.02s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(2) {{ animation-delay: 0.06s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(3) {{ animation-delay: 0.10s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(4) {{ animation-delay: 0.14s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(5) {{ animation-delay: 0.18s; }}

    /* --- DOWNLOAD BUTTON --- */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, {COLORS['navy_mid']}, {COLORS['lavender_dark']}) !important;
        color: {COLORS['white']} !important;
        border: 1px solid {COLORS['lavender_dark']} !important;
        border-radius: 8px;
        transition: all 0.3s ease;
    }}

    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['lavender_dark']}, {COLORS['lavender']}) !important;
        box-shadow: 0 4px 16px rgba(155,142,196,0.25);
        transform: translateY(-1px);
    }}

    /* --- SLIDER --- */
    .stSlider [data-testid="stThumbValue"] {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }}

    /* --- MARKDOWN LINKS --- */
    a {{
        color: {COLORS['gold_light']} !important;
        text-decoration: none;
        transition: color 0.2s ease;
    }}
    a:hover {{
        color: {COLORS['gold']} !important;
        text-decoration: underline;
    }}

    /* --- TOOLTIPS & POPOVERS --- */
    [data-baseweb="tooltip"] {{
        background: {COLORS['navy_mid']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px;
    }}

    /* --- PROGRESS BAR --- */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {COLORS['gold_dark']}, {COLORS['gold']}, {COLORS['gold_light']});
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
        border-radius: 4px;
    }}

    /* --- EMPTY STATE TEXT --- */
    .stMarkdown p {{
        line-height: 1.6;
    }}

    </style>
    """


def apply_glass_card(content_html: str, accent: str = "gold") -> str:
    """Return HTML for a glassmorphism card with optional accent color."""
    accent_color = COLORS.get(accent, COLORS["gold"])
    return (
        f'<div style="'
        f'background:{COLORS["glass_bg"]};'
        f'border:1px solid {COLORS["glass_border"]};'
        f'border-top:2px solid {accent_color};'
        f'border-radius:12px;'
        f'padding:20px;'
        f'backdrop-filter:blur(12px);'
        f'-webkit-backdrop-filter:blur(12px);'
        f'animation:fadeInUp 0.4s ease-out;'
        f'">{content_html}</div>'
    )


def render_header(title: str, subtitle: str = ""):
    """Render a professional header bar with shimmer accent."""
    sub_html = (
        f'<p style="color:{COLORS["lavender"]};font-size:0.85rem;'
        f'margin:6px 0 0 0;font-weight:300;letter-spacing:1px;">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="text-align:center;padding:28px 24px 22px;'
        f'background:linear-gradient(135deg,{COLORS["black"]},{COLORS["navy"]},{COLORS["navy_mid"]});'
        f'border:1px solid {COLORS["border"]};border-radius:12px;margin-bottom:24px;'
        f'box-shadow:0 8px 32px rgba(0,0,0,0.4);'
        f'animation:fadeInUp 0.5s ease-out;">'
        f'<h1 style="color:{COLORS["gold"]};margin:0;font-family:Inter,sans-serif;'
        f'letter-spacing:3px;font-size:1.8rem;border:none;padding:0;'
        f'font-weight:700;">OCTAVIAN</h1>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str):
    """Render a section header with gold accent and fade animation."""
    st.markdown(
        f'<h3 style="color:{COLORS["gold_light"]};border-left:3px solid {COLORS["gold"]};'
        f'padding-left:14px;margin:24px 0 12px 0;font-weight:600;'
        f'animation:slideInLeft 0.4s ease-out;">{text}</h3>',
        unsafe_allow_html=True,
    )


def status_badge(text: str, variant: str = "neutral") -> str:
    """Return inline HTML for a colored status badge. variant: success|danger|neutral|gold|lavender"""
    color_map = {
        "success": (COLORS["success"], "rgba(76,175,80,0.12)"),
        "danger": (COLORS["danger"], "rgba(239,83,80,0.12)"),
        "neutral": (COLORS["neutral"], "rgba(120,144,156,0.12)"),
        "gold": (COLORS["gold"], "rgba(201,168,76,0.12)"),
        "lavender": (COLORS["lavender"], "rgba(155,142,196,0.12)"),
    }
    fg, bg = color_map.get(variant, color_map["neutral"])
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'font-size:0.72rem;font-weight:600;letter-spacing:0.5px;'
        f'color:{fg};background:{bg};border:1px solid {fg};">{text}</span>'
    )
