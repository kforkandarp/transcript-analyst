"""Streamlit UI for verified expert transcript analysis and synthesis."""

from __future__ import annotations

import html
import logging
from pathlib import Path
import streamlit as st

from analyst.chat import ask, topics_covered
from analyst.config import settings
from analyst.index import IndexStore
from analyst.models import Guide, GuideCell, Theme, Transcript
from analyst.pipeline import build_store, load_cached
from analyst.ui_helpers import (
    highlight_quote_in_text,
    render_coverage_badge,
    render_evidence,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyst.app")

DATA_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"

# Page configuration
st.set_page_config(
    page_title="Analyst — Expert Transcripts",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- 1. Authentication Gate ---

def check_password() -> bool:
    """Optional password gate; skipped if APP_PASSWORD is not set or empty."""
    app_pwd = getattr(settings, "app_password", "").strip()
    if not app_pwd:
        return True

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.markdown("### Authentication Required")
    pwd_input = st.text_input("Enter Access Password", type="password")
    if st.button("Log In"):
        if pwd_input == app_pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid password.")
    return False


if not check_password():
    st.stop()


# --- 2. Data Loading & Index Setup ---

cached_bundle = load_cached(DATA_PROCESSED_DIR, DATA_RAW_DIR)

if cached_bundle is None:
    st.error(
        "⚠️ No valid precomputed cache found in `data/processed/`. "
        "Please run `python -m analyst.build_cache --force` in your terminal to process the transcripts."
    )
    st.stop()

transcripts: list[Transcript] = cached_bundle["transcripts"]
guide: Guide = cached_bundle["guide"]
guide_cells: list[GuideCell] = cached_bundle["guide_cells"]
themes: list[Theme] = cached_bundle["themes"]
meta: dict = cached_bundle["meta"]
store: IndexStore = cached_bundle["store"]


# --- 3. Sidebar Setup ---

with st.sidebar:
    st.title("🔬 Analyst Intelligence")
    st.caption("Verbatim-grounded transcript extraction & synthesis")
    st.markdown("---")

    st.subheader("Loaded Experts")
    for t in transcripts:
        st.markdown(
            f"**{t.expert_name}** (`{t.id}`)\n\n"
            f"<small style='color: #616161;'>{t.expert_role} • {t.market}</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Models & Integrity")
    models = meta.get("models", {})
    st.markdown(
        f"- **Synthesis**: `{models.get('large', 'unknown')}`\n"
        f"- **Decomposition**: `{models.get('small', 'unknown')}`\n"
        f"- **Embeddings**: `{models.get('embed', 'unknown')}`\n"
        f"- **Prompt Version**: `{meta.get('prompt_version', '1.0')}`"
    )
    st.caption("🛡️ Quotes are machine-verified verbatim; timestamps are turn start times.")


# --- 4. Navigation & State Setup ---

if "view" not in st.session_state:
    st.session_state["view"] = "Transcripts"

if "highlight" not in st.session_state:
    st.session_state["highlight"] = None

if "selected_transcript_id" not in st.session_state:
    st.session_state["selected_transcript_id"] = transcripts[0].id

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Main top horizontal navigation
nav_views = ["Transcripts", "Interview Guide", "Themes", "Ask"]
active_view = st.radio(
    "Navigation",
    options=nav_views,
    horizontal=True,
    key="view",
    label_visibility="collapsed",
)


# ==============================================================================
# VIEW 1: TRANSCRIPTS
# ==============================================================================
if active_view == "Transcripts":
    st.header("📄 Expert Transcripts")

    transcript_lookup = {t.id: t for t in transcripts}
    tid_options = list(transcript_lookup.keys())
    current_tid = st.session_state.get("selected_transcript_id", tid_options[0])
    selected_idx = tid_options.index(current_tid) if current_tid in tid_options else 0

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        chosen_tid = st.selectbox(
            "Select Expert Transcript",
            options=tid_options,
            index=selected_idx,
            format_func=lambda x: f"{x} - {transcript_lookup[x].expert_name} ({transcript_lookup[x].market})",
            key="transcript_picker",
        )
        st.session_state["selected_transcript_id"] = chosen_tid

    selected_t = transcript_lookup[chosen_tid]

    with col_info:
        st.markdown(
            f"**{selected_t.expert_name}** | {selected_t.expert_role} | {selected_t.market}\n\n"
            f"<small style='color: #616161;'>Total turns: {len(selected_t.turns)} | Transcript ID: {selected_t.id}</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    active_highlight = st.session_state.get("highlight")
    if active_highlight:
        hl_chunk = active_highlight.get("chunk_id", "")
        if hl_chunk.startswith(chosen_tid):
            st.info(f"Target citation active: chunk **{hl_chunk}** highlighted below.")
            if st.button("Clear highlight", key="clear_hl_btn"):
                st.session_state["highlight"] = None
                st.rerun()

    # Render turns
    for turn in selected_t.turns:
        ts_str = f"[{turn.ts}]"
        is_interviewer = turn.role.lower() == "interviewer"

        if is_interviewer:
            st.markdown(
                f"<div style='color: #757575; font-style: italic; margin-bottom: 8px;'>"
                f"<strong>Interviewer</strong> {ts_str}: {html.escape(turn.text)}</div>",
                unsafe_allow_html=True,
            )
        else:
            text_to_show = turn.text
            if active_highlight and active_highlight.get("quote"):
                hl_quote = active_highlight["quote"]
                if hl_quote in text_to_show:
                    text_to_show = highlight_quote_in_text(text_to_show, hl_quote)
                else:
                    text_to_show = html.escape(text_to_show)
            else:
                text_to_show = html.escape(text_to_show)

            st.markdown(
                f"<div style='margin-bottom: 12px; line-height: 1.5;'>"
                f"<strong>{selected_t.expert_name}</strong> "
                f"<span style='color: #616161; font-size: 0.85rem;'>{ts_str}</span>: "
                f"{text_to_show}</div>",
                unsafe_allow_html=True,
            )


    
# ==============================================================================
# VIEW 2: INTERVIEW GUIDE
# ==============================================================================
elif active_view == "Interview Guide":
    st.header("📋 Cross-Transcript Interview Guide")
    st.caption("Decomposed sub-questions synthesized across all 3 experts.")

    cell_map: dict[tuple[int, str], GuideCell] = {
        (c.question_id, c.transcript_id): c for c in guide_cells
    }

    for q in guide.questions:
        st.markdown(f"### Q{q.id}: {q.text}")
        cols = st.columns(len(transcripts))

        for idx, t in enumerate(transcripts):
            with cols[idx]:
                st.markdown(
                    f"<div style='background-color: #f5f5f5; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px;'>"
                    f"<strong>{t.expert_name}</strong><br>"
                    f"<small style='color: #616161;'>{t.expert_role} ({t.market})</small></div>",
                    unsafe_allow_html=True,
                )

                cell = cell_map.get((q.id, t.id))
                if not cell:
                    st.write("No data available.")
                    continue

                show_part_labels = len(cell.parts) > 1

                for p_idx, part in enumerate(cell.parts):
                    if show_part_labels:
                        st.markdown(f"**Sub-Q {p_idx+1}:** *{part.sub_question}*")

                    render_coverage_badge(part.result.coverage)

                    if part.result.note:
                        st.caption(f"_{part.result.note}_")

                    for c_idx, claim in enumerate(part.result.claims):
                        st.markdown(f"• {claim.text}")
                        render_evidence(
                            claim.evidence,
                            key_prefix=f"g_q{q.id}_{t.id}_p{p_idx}_c{c_idx}",
                            expert_name=t.expert_name,
                        )

                    if p_idx < len(cell.parts) - 1:
                        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

        st.markdown("---")


# ==============================================================================
# VIEW 3: THEMES
# ==============================================================================
elif active_view == "Themes":
    st.header("💡 Synthesized Market Themes")
    st.caption("Based on 3 experts, not the market")

    disagreements = [th for th in themes if th.kind == "disagreement"]
    commons = [th for th in themes if th.kind == "common"]
    uniques = [th for th in themes if th.kind == "unique"]

    def _render_theme_card(th: Theme, group_prefix: str, idx: int) -> None:
        total_experts = len(transcripts)
        verdict_str = f" • Verdict: **{th.verdict.replace('_', ' ').title()}**" if th.verdict else ""

        with st.container(border=True):
            st.markdown(f"#### {th.title}")
            st.caption(f"Raised by **{th.expert_count} of {total_experts}** experts{verdict_str}")

            if th.summary:
                st.info(th.summary)

            st.markdown("**Expert Positions:**")
            for pos_idx, pos in enumerate(th.positions):
                val_parts = []
                if pos.value:
                    val_parts.append(f"Value: `{pos.value}`")
                if pos.scope:
                    val_parts.append(f"Scope: `{pos.scope}`")
                if pos.qualifier:
                    val_parts.append(f"Qualifier: `{pos.qualifier}`")

                val_annotation = f" ({', '.join(val_parts)})" if val_parts else ""
                st.markdown(
                    f"• **{pos.expert_name}** ({pos.market}): {pos.position}{val_annotation}"
                )
                render_evidence(
                    pos.evidence,
                    key_prefix=f"{group_prefix}_{idx}_p{pos_idx}",
                    expert_name=pos.expert_name,
                )

    if disagreements:
        st.subheader("⚡ Disagreements & Divergent Projections")
        for i, th in enumerate(disagreements):
            _render_theme_card(th, "disagree", i)

    if commons:
        st.subheader("🤝 Common Themes & Market Consensus")
        for i, th in enumerate(commons):
            _render_theme_card(th, "common", i)

    if uniques:
        st.subheader("🔍 Unique Angles & Local Realities")
        for i, th in enumerate(uniques):
            _render_theme_card(th, "unique", i)

# ==============================================================================
# VIEW 4: ASK
# ==============================================================================
elif active_view == "Ask":
    st.header("💬 Interactive Expert Query")
    st.caption("Ask questions across all transcripts. Quotes are verified before surfacing.")

    st.markdown("**Suggested queries:**")
    ex_cols = st.columns(3)
    suggested_q = None

    if ex_cols[0].button("Which experts say finance decides the purchase?", key="q1_chip"):
        suggested_q = "Which experts say finance decides the purchase?"
    if ex_cols[1].button("How long does a purchase decision take?", key="q2_chip"):
        suggested_q = "How long does a purchase decision take?"
    if ex_cols[2].button("What did the UK expert say about training?", key="q3_chip"):
        suggested_q = "What did the UK expert say about training?"

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                res = msg["result"]
                debug = msg.get("debug", {})
                render_coverage_badge(res.coverage)

                if res.note:
                    st.info(res.note)

                for c_idx, claim in enumerate(res.claims):
                    st.markdown(f"• {claim.text}")
                    render_evidence(
                        claim.evidence,
                        key_prefix=f"hist_{msg['id']}_{c_idx}",
                    )

                if res.coverage == "not_discussed":
                    st.markdown("**Topics covered in the transcripts:**")
                    for top in topics_covered(guide):
                        st.markdown(f"- {top}")

                with st.expander("How this was answered"):
                    st.markdown(f"- **Standalone Question**: `{debug.get('standalone_question', 'N/A')}`")
                    st.markdown(f"- **Target Transcripts**: `{debug.get('target_ids', [])}`")
                    st.markdown(f"- **Retrieved Chunks**: `{debug.get('retrieved_chunk_ids', [])}`")
                    st.markdown(
                        f"- **Quotes Verified**: {res.stats.get('verified_quotes', 0)} / {res.stats.get('raw_quotes', 0)} "
                        f"({res.stats.get('dropped_quotes', 0)} dropped, {res.stats.get('retries', 0)} retries)"
                    )

    user_query = st.chat_input("Ask a question about robotic surgery adoption...")
    final_query = suggested_q or user_query

    if final_query:
        st.session_state["chat_history"].append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.write(final_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, retrieving exchanges, and verifying quotes..."):
                try:
                    clean_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["chat_history"][:-1]
                    ]
                    result, debug = ask(store, transcripts, guide, final_query, clean_history)

                    render_coverage_badge(result.coverage)

                    if result.note:
                        st.info(result.note)

                    for c_idx, claim in enumerate(result.claims):
                        st.markdown(f"• {claim.text}")
                        render_evidence(
                            claim.evidence,
                            key_prefix=f"live_{len(st.session_state['chat_history'])}_{c_idx}",
                        )

                    if result.coverage == "not_discussed":
                        st.markdown("**Topics covered in the transcripts:**")
                        for top in topics_covered(guide):
                            st.markdown(f"- {top}")

                    with st.expander("How this was answered"):
                        st.markdown(f"- **Standalone Question**: `{debug.get('standalone_question', 'N/A')}`")
                        st.markdown(f"- **Target Transcripts**: `{debug.get('target_ids', [])}`")
                        st.markdown(f"- **Retrieved Chunks**: `{debug.get('retrieved_chunk_ids', [])}`")
                        st.markdown(
                            f"- **Quotes Verified**: {result.stats.get('verified_quotes', 0)} / {result.stats.get('raw_quotes', 0)} "
                            f"({result.stats.get('dropped_quotes', 0)} dropped, {result.stats.get('retries', 0)} retries)"
                        )

                    st.session_state["chat_history"].append(
                        {
                            "id": len(st.session_state["chat_history"]),
                            "role": "assistant",
                            "content": "\n".join(c.text for c in result.claims),
                            "result": result,
                            "debug": debug,
                        }
                    )

                except Exception as exc:
                    logger.error("Error executing query: %s", exc)
                    st.error(f"An unexpected error occurred while processing your request: {exc}")