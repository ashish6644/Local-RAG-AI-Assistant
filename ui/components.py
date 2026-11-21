"""Shared UI helpers: CSS, formatting, source rendering."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import streamlit as st

STATUS_BADGES = {"ready": "✅ Ready", "processing": "⏳ Processing", "failed": "❌ Failed"}

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 980px;}
[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
.app-title {font-size: 1.25rem; font-weight: 700; margin-bottom: .2rem;}
.welcome {text-align: center; padding: 2.5rem 1rem 1rem 1rem;}
.welcome h1 {font-size: 2rem; margin-bottom: .25rem;}
.welcome p {opacity: .75;}
.msg-time {font-size: .75rem; opacity: .55;}
[data-testid="stSidebar"] .stButton button {text-align: left; justify-content: flex-start;}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def to_local(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc).astimezone()


def format_dt(dt: Optional[datetime]) -> str:
    return to_local(dt).strftime("%d %b %Y, %H:%M") if dt else ""


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def show_error(message: str) -> None:
    st.error(message, icon="🚨")


def _quote(text: str) -> str:
    safe = text.replace("$", "\\$").strip()
    return "> " + safe.replace("\n", "\n> ")


def render_sources(payload: Optional[dict]) -> None:
    """Show cited chunks (filename, location, score, text) plus retrieval details."""
    if not payload:
        return
    chunks = payload.get("chunks") or []
    if not chunks:
        return
    with st.expander(f"📚 Sources & retrieval details ({len(chunks)})"):
        for number, c in enumerate(chunks, start=1):
            where = f" · {c['location']}" if c.get("location") else ""
            st.markdown(f"**[{number}] {c.get('filename', 'unknown')}**{where} · similarity {c.get('score', 0):.2f}")
            st.markdown(_quote(c.get("text", "")))
        queries = payload.get("queries") or []
        rewritten = payload.get("rewritten_query")
        if rewritten:
            st.caption(f"Search query: {rewritten}")
        if len(queries) > 1:
            st.caption("Query variants: " + " | ".join(queries))
