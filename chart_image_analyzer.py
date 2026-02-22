"""
Octavian Chart Image Analyzer
Analyzes uploaded chart images for patterns, support/resistance, and trade signals.
Author: APB - Octavian Team
"""

import streamlit as st
import numpy as np
import base64
import io
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from PIL import Image

HAS_LLM_VISION = False
try:
    import importlib.metadata
    importlib.metadata.version("llama-cpp-python")
    HAS_LLM_VISION = True
except (ImportError, importlib.metadata.PackageNotFoundError):
    pass

try:
    from ai_chatbot import _load_llm, _llm_call
    HAS_CHATBOT_LLM = True
except ImportError:
    HAS_CHATBOT_LLM = False


@dataclass
class ChartAnalysisResult:
    """Result from chart image analysis."""
    trend_direction: str = "NEUTRAL"
    trend_strength: str = "Moderate"
    support_levels: List[str] = field(default_factory=list)
    resistance_levels: List[str] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)
    buy_signals: List[str] = field(default_factory=list)
    sell_signals: List[str] = field(default_factory=list)
    action_now: str = ""
    watch_before_buying: List[str] = field(default_factory=list)
    watch_before_selling: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)
    confidence: float = 0.5
    full_analysis: str = ""
    timestamp: str = ""


def _analyze_image_pixels(img: Image.Image) -> Dict[str, Any]:
    """Extract basic visual features from the chart image using pixel analysis."""
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape

    # Sample the right third of the image for recent price action
    right_section = arr[:, int(w * 0.66):, :]
    left_section = arr[:, :int(w * 0.33), :]

    # Detect dominant colors (candle colors)
    green_mask = (right_section[:, :, 1] > right_section[:, :, 0] + 20) & \
                 (right_section[:, :, 1] > right_section[:, :, 2] + 20)
    red_mask = (right_section[:, :, 0] > right_section[:, :, 1] + 20) & \
               (right_section[:, :, 0] > right_section[:, :, 2] + 20)

    green_pct = green_mask.sum() / green_mask.size
    red_pct = red_mask.sum() / red_mask.size

    # Estimate trend from brightness distribution (price position)
    gray = np.mean(arr, axis=2)
    left_brightness = np.mean(gray[:, :int(w * 0.3)])
    right_brightness = np.mean(gray[:, int(w * 0.7):])

    # Detect if there are horizontal lines (support/resistance)
    row_variance = np.var(gray, axis=1)
    low_var_rows = np.where(row_variance < np.percentile(row_variance, 10))[0]
    potential_levels = len(low_var_rows)

    # Detect if there are indicator panels (multiple sections)
    col_mean = np.mean(gray, axis=1)
    large_gaps = np.where(np.abs(np.diff(col_mean)) > 30)[0]
    num_panels = max(1, len(large_gaps) // 5 + 1)

    # Image dimensions and aspect ratio
    aspect = w / h

    return {
        "green_dominance": float(green_pct),
        "red_dominance": float(red_pct),
        "bullish_bias": green_pct > red_pct * 1.2,
        "bearish_bias": red_pct > green_pct * 1.2,
        "estimated_panels": min(num_panels, 4),
        "has_indicators": num_panels > 1,
        "potential_levels": potential_levels,
        "aspect_ratio": aspect,
        "width": w,
        "height": h,
    }


def _build_pixel_analysis(pixel_data: Dict, user_context: str = "") -> ChartAnalysisResult:
    """Build analysis result from pixel-level features."""
    result = ChartAnalysisResult(timestamp=datetime.now().isoformat())

    # Trend
    if pixel_data["bullish_bias"]:
        result.trend_direction = "BULLISH"
        result.trend_strength = "Moderate to Strong" if pixel_data["green_dominance"] > 0.05 else "Moderate"
    elif pixel_data["bearish_bias"]:
        result.trend_direction = "BEARISH"
        result.trend_strength = "Moderate to Strong" if pixel_data["red_dominance"] > 0.05 else "Moderate"
    else:
        result.trend_direction = "NEUTRAL / CONSOLIDATING"
        result.trend_strength = "Weak"

    # Patterns
    if pixel_data["has_indicators"]:
        result.patterns_detected.append(f"Chart contains approximately {pixel_data['estimated_panels']} panels (price + indicators)")
    if pixel_data["potential_levels"] > 10:
        result.patterns_detected.append("Horizontal levels detected (possible support/resistance lines)")

    # Signals based on color dominance
    g = pixel_data["green_dominance"]
    r = pixel_data["red_dominance"]

    if g > r * 1.5:
        result.buy_signals.append("Recent candles predominantly bullish (green)")
        result.buy_signals.append("Upward momentum visible in price action")
        result.confidence = min(0.7, 0.4 + g * 5)
    elif r > g * 1.5:
        result.sell_signals.append("Recent candles predominantly bearish (red)")
        result.sell_signals.append("Downward pressure visible in price action")
        result.confidence = min(0.7, 0.4 + r * 5)
    else:
        result.buy_signals.append("Mixed signals — no clear directional bias")
        result.sell_signals.append("Mixed signals — no clear directional bias")
        result.confidence = 0.3

    # Action recommendation
    if result.trend_direction == "BULLISH":
        result.action_now = ("Current bias is bullish. Consider entries on pullbacks to support. "
                             "Wait for a retest of recent lows or moving average support before adding.")
    elif result.trend_direction == "BEARISH":
        result.action_now = ("Current bias is bearish. Avoid catching the falling knife. "
                             "Wait for a reversal pattern or oversold bounce before considering longs. "
                             "Short entries may be valid on rallies to resistance.")
    else:
        result.action_now = ("Price is in consolidation. Wait for a breakout above resistance or "
                             "breakdown below support before taking a directional position. "
                             "Reduced position sizing recommended in ranging markets.")

    # Watch items
    result.watch_before_buying = [
        "Confirm support holds on any pullback",
        "Look for increasing volume on up moves",
        "Check RSI/momentum for oversold conditions on dips",
        "Verify broader market context aligns with long bias",
        "Set stop loss below the most recent swing low",
    ]
    result.watch_before_selling = [
        "Confirm resistance rejection with bearish candle patterns",
        "Look for volume expansion on down moves",
        "Check if RSI/momentum shows overbought divergence",
        "Verify sector/market weakness supports short thesis",
        "Set stop loss above the most recent swing high",
    ]
    result.risk_notes = [
        "Image-based analysis has inherent limitations — always cross-reference with live data",
        "Without price scale, exact levels cannot be determined",
        "Timeframe context significantly affects interpretation",
    ]

    # Incorporate user context
    if user_context:
        result.patterns_detected.insert(0, f"User context: {user_context}")
        result.risk_notes.append("User-provided context has been factored into the analysis")

    # Build full narrative
    lines = [
        f"TREND ASSESSMENT: {result.trend_direction} ({result.trend_strength})",
        f"CONFIDENCE: {result.confidence:.0%}",
        "",
        "CURRENT ACTION:",
        result.action_now,
        "",
    ]
    if result.buy_signals:
        lines.append("BUY SIGNALS:")
        for s in result.buy_signals:
            lines.append(f"  - {s}")
    if result.sell_signals:
        lines.append("SELL SIGNALS:")
        for s in result.sell_signals:
            lines.append(f"  - {s}")
    if result.patterns_detected:
        lines.append("DETECTED PATTERNS:")
        for p in result.patterns_detected:
            lines.append(f"  - {p}")
    result.full_analysis = "\n".join(lines)

    return result


def _llm_analyze_chart(img: Image.Image, user_context: str = "") -> Optional[str]:
    """Try to use LLM for deeper chart analysis."""
    if not HAS_CHATBOT_LLM:
        return None
    try:
        llm = _load_llm()
        if not llm:
            return None

        pixel_data = _analyze_image_pixels(img)

        prompt_parts = [
            "A user has uploaded a financial chart image for analysis.",
            f"Image dimensions: {pixel_data['width']}x{pixel_data['height']}",
            f"Estimated panels: {pixel_data['estimated_panels']}",
            f"Color analysis: Green dominance={pixel_data['green_dominance']:.3f}, Red dominance={pixel_data['red_dominance']:.3f}",
            f"Apparent bias: {'Bullish' if pixel_data['bullish_bias'] else 'Bearish' if pixel_data['bearish_bias'] else 'Neutral'}",
        ]
        if user_context:
            prompt_parts.append(f"User provided context: {user_context}")

        prompt_parts.extend([
            "",
            "Based on this chart analysis data, provide:",
            "1. Trend assessment and strength",
            "2. Key levels to watch (support/resistance estimates)",
            "3. Buy signals present",
            "4. Sell signals present",
            "5. What the trader should do RIGHT NOW",
            "6. What to watch before buying",
            "7. What to watch before selling",
            "8. Risk factors",
            "",
            "Be specific, actionable, and professional. Use bullet points.",
        ])

        system = ("You are Octavian, an elite institutional chart analyst. "
                  "Analyze chart data and provide precise, actionable trade recommendations. "
                  "Be professional and data-driven. No emojis. Use structured formatting.")

        result = _llm_call(llm, system, "\n".join(prompt_parts), max_tokens=1200, timeout=30)
        return result
    except Exception:
        return None


def analyze_chart_image(uploaded_file, user_context: str = "") -> ChartAnalysisResult:
    """Main entry point: analyze an uploaded chart image."""
    try:
        img = Image.open(uploaded_file)
        pixel_data = _analyze_image_pixels(img)

        # Try LLM analysis first
        llm_text = _llm_analyze_chart(img, user_context)

        # Build base result from pixels
        result = _build_pixel_analysis(pixel_data, user_context)

        # If LLM provided analysis, use it as the full narrative
        if llm_text:
            result.full_analysis = llm_text

        return result
    except Exception as e:
        result = ChartAnalysisResult(timestamp=datetime.now().isoformat())
        result.full_analysis = f"Analysis error: {str(e)}. Please ensure the uploaded file is a valid image."
        result.action_now = "Unable to analyze. Please try a different image."
        return result


def show_chart_analyzer():
    """Render the chart image analyzer UI."""
    st.markdown("### Chart Image Analysis")
    st.caption("Upload any chart image — candlesticks, line charts, with or without indicators. "
               "The model will identify potential entry/exit points and actionable insights.")

    col_upload, col_context = st.columns([1, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload Chart Image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="chart_img_upload",
            help="Any screenshot of a price chart — TradingView, broker platform, etc."
        )

    with col_context:
        user_context = st.text_area(
            "Context (optional)",
            placeholder="e.g., 'AAPL daily chart, considering a long position. RSI is at 35. Earnings next week.'",
            height=120,
            key="chart_img_context",
            help="Add any context about the chart — symbol, timeframe, your thesis, indicators shown, etc."
        )

    if uploaded:
        # Show the uploaded image
        st.image(uploaded, caption="Uploaded Chart", use_container_width=True)

        if st.button("Analyze Chart", type="primary", key="analyze_chart_btn", use_container_width=True):
            with st.spinner("Analyzing chart patterns..."):
                result = analyze_chart_image(uploaded, user_context)

            # Display results
            st.markdown("---")

            # Header metrics
            trend_color = "#4CAF50" if "BULL" in result.trend_direction else "#EF5350" if "BEAR" in result.trend_direction else "#9E9E9E"

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Trend", result.trend_direction)
            with m2:
                st.metric("Strength", result.trend_strength)
            with m3:
                st.metric("Confidence", f"{result.confidence:.0%}")

            # Action now
            st.markdown("#### Recommended Action")
            st.info(result.action_now)

            # Signals
            col_buy, col_sell = st.columns(2)
            with col_buy:
                st.markdown("#### Buy Signals")
                for s in result.buy_signals:
                    st.markdown(f"- {s}")
            with col_sell:
                st.markdown("#### Sell Signals")
                for s in result.sell_signals:
                    st.markdown(f"- {s}")

            # Watch items
            col_wb, col_ws = st.columns(2)
            with col_wb:
                st.markdown("#### Watch Before Buying")
                for w in result.watch_before_buying:
                    st.markdown(f"- {w}")
            with col_ws:
                st.markdown("#### Watch Before Selling")
                for w in result.watch_before_selling:
                    st.markdown(f"- {w}")

            # Detected patterns
            if result.patterns_detected:
                with st.expander("Detected Patterns & Features"):
                    for p in result.patterns_detected:
                        st.markdown(f"- {p}")

            # Risk notes
            with st.expander("Risk Notes"):
                for r in result.risk_notes:
                    st.caption(f"- {r}")

            # Full analysis
            if result.full_analysis:
                with st.expander("Full Analysis Report"):
                    st.markdown(result.full_analysis)
