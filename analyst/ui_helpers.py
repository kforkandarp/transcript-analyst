"""Reusable Streamlit UI rendering components and interactive state helpers."""

from __future__ import annotations

import html
import re
from typing import Sequence
import streamlit as st

from analyst.models import Evidence


def render_coverage_badge(coverage: str) -> None:
    """Render a colored badge for claim or cell coverage."""
    colors = {
        "direct": "#2e7d32",        # Green
        "indirect": "#f57c00",      # Orange / Amber
        "not_discussed": "#757575",  # Muted Grey
        "unverified": "#c62828",    # Red
    }
    cov = coverage.lower().strip()
    color = colors.get(cov, "#616161")
    label = cov.replace("_", " ").title()

    st.markdown(
        f'<span style="background-color: {color}; color: white; padding: 2px 8px; '
        f'border-radius: 4px; font-size: 0.78rem; font-weight: 600; display: inline-block;">'
        f'{label}</span>',
        unsafe_allow_html=True,
    )


def jump_to_transcript(chunk_id: str, quote: str) -> None:
    """Set the active citation highlight, switch to the Transcripts view, and rerun."""
    tid = chunk_id.split("-")[0]
    st.session_state["selected_transcript_id"] = tid
    st.session_state["highlight"] = {
        "chunk_id": chunk_id,
        "quote": quote,
    }
    st.session_state["view"] = "Transcripts"
    st.rerun()


def render_evidence(
    evidence_list: Sequence[Evidence],
    key_prefix: str,
    expert_name: str = "",
) -> None:
    """Render the standard evidence expander with quotes, verification badges, and jump buttons."""
    if not evidence_list:
        return

    with st.expander(f"Evidence ({len(evidence_list)} verified quote{'s' if len(evidence_list) > 1 else ''})"):
        for idx, ev in enumerate(evidence_list):
            name_display = f"{expert_name} • " if expert_name else ""
            st.markdown(
                f'> "{html.escape(ev.quote)}"\n\n'
                f'<span style="color: #2e7d32; font-weight: 600;">✓ verified verbatim</span> '
                f'<span style="color: #616161; font-size: 0.85rem;">— {name_display}[{ev.ts}] ({ev.chunk_id})</span>',
                unsafe_allow_html=True,
            )
            st.button(
                "View in transcript",
                key=f"{key_prefix}_jump_{idx}_{ev.chunk_id}",
                on_click=jump_to_transcript,
                args=(ev.chunk_id, ev.quote),
            )


def highlight_quote_in_text(text: str, quote_to_highlight: str) -> str:
    """Safely highlight a verbatim quote substring inside source text with a yellow mark."""
    if not quote_to_highlight:
        return html.escape(text)

    clean_text = html.escape(text)
    clean_quote = html.escape(quote_to_highlight)

    # Replace first occurrence with styled mark
    pattern = re.escape(clean_quote)
    highlighted = re.sub(
        pattern,
        lambda m: (
            f'<mark style="background-color: #fff176; padding: 2px 4px; '
            f'border-radius: 2px; font-weight: 500;">{m.group(0)}</mark>'
        ),
        clean_text,
        count=1,
    )
    return highlighted