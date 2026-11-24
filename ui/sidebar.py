"""Sidebar: branding, navigation, chat history, document overview."""
from __future__ import annotations

import streamlit as st

from core.errors import AppError
from core.ollama_client import OllamaClient
from services.container import Services
from ui.components import STATUS_BADGES

PAGES = ["💬 Chat", "📄 Documents", "⚙️ Settings"]


@st.cache_data(ttl=10, show_spinner=False)
def _ollama_up(base_url: str) -> bool:
    return OllamaClient(base_url).is_running(timeout=1.5)


def _new_chat() -> None:
    st.session_state["conversation_id"] = None
    st.session_state["page"] = PAGES[0]


def _open_chat(conversation_id: int) -> None:
    st.session_state["conversation_id"] = conversation_id
    st.session_state["page"] = PAGES[0]


def _delete_chat(services: Services, conversation_id: int) -> None:
    try:
        services.chat.delete_conversation(conversation_id)
    except AppError as exc:
        st.session_state["flash_error"] = str(exc)
    if st.session_state.get("conversation_id") == conversation_id:
        st.session_state["conversation_id"] = None


def render_sidebar(services: Services) -> str:
    settings = st.session_state["settings"]
    with st.sidebar:
        st.markdown('<div class="app-title">📚 Local RAG AI Assistant</div>', unsafe_allow_html=True)
        up = _ollama_up(services.settings.ollama_base_url)
        st.caption(("🟢 Ollama running" if up else "🔴 Ollama offline") + f" · model: `{settings.chat_model}`")
        page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.button("➕ New chat", use_container_width=True, type="primary", on_click=_new_chat)

        st.markdown("**Chats**")
        try:
            conversations = services.chat.list_conversations()
        except AppError as exc:
            st.error(str(exc))
            conversations = []
        if not conversations:
            st.caption("No conversations yet.")
        current = st.session_state.get("conversation_id")
        for conv in conversations[:40]:
            col_open, col_del = st.columns([5, 1])
            title = conv.title if len(conv.title) <= 28 else conv.title[:27] + "…"
            col_open.button(
                title, key=f"open_{conv.conversation_id}", use_container_width=True,
                type="primary" if conv.conversation_id == current else "secondary",
                on_click=_open_chat, args=(conv.conversation_id,),
            )
            col_del.button("🗑", key=f"del_{conv.conversation_id}", help="Delete conversation",
                           on_click=_delete_chat, args=(services, conv.conversation_id))

        try:
            docs = services.documents.list_documents()
        except AppError as exc:
            st.error(str(exc))
            docs = []
        with st.expander(f"📄 Documents ({len(docs)})"):
            if not docs:
                st.caption("No documents uploaded. Open the Documents page to add some.")
            for d in docs:
                badge = STATUS_BADGES.get(d.processing_status, d.processing_status).split()[0]
                st.caption(f"{badge} {d.filename} · {d.chunk_count} chunks")
    return page
