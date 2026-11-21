"""Main chat area: history, welcome screen, streaming answers, stop button."""
from __future__ import annotations

import logging

import streamlit as st

from core.errors import AppError
from core.rag_chain import RAGParams
from services.container import Services
from ui.components import format_dt, render_sources, show_error

logger = logging.getLogger(__name__)

EXAMPLES = [
    "What does this document explain?",
    "Summarize the main points.",
    "What are the important findings?",
    "Compare the information in these documents.",
]


def _set_pending(prompt: str) -> None:
    st.session_state["pending_prompt"] = prompt


def _render_message(msg) -> None:
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        st.markdown(f'<span class="msg-time">{format_dt(msg.timestamp)}</span>', unsafe_allow_html=True)
        if msg.role == "assistant":
            render_sources(msg.sources)


def _render_header(services: Services, conv_id, ready_docs) -> None:
    conv = services.chat.get_conversation(conv_id) if conv_id else None
    left, mid, right = st.columns([5, 1.2, 1.4])
    left.subheader(conv.title if conv else "New chat")
    if conv:
        with mid.popover("✏️ Rename"):
            new_title = st.text_input("Conversation title", value=conv.title, key=f"rename_{conv_id}")
            if st.button("Save title", key=f"save_title_{conv_id}"):
                services.chat.rename_conversation(conv_id, new_title)
                st.rerun()
        if right.button("🧹 Clear chat", key=f"clear_{conv_id}"):
            services.chat.clear_conversation(conv_id)
            st.rerun()
    if ready_docs:
        names = {d.document_id: d.filename for d in ready_docs}
        st.session_state["doc_filter"] = [i for i in st.session_state.get("doc_filter", []) if i in names]
        with st.popover("🔎 Search scope"):
            st.multiselect(
                "Limit answers to these documents (empty = all documents)",
                options=list(names), format_func=lambda i: names[i], key="doc_filter",
            )


def _welcome(ready_docs) -> None:
    st.markdown(
        '<div class="welcome"><h1>Ask your documents anything</h1>'
        "<p>Answers are generated locally with Ollama and grounded in your uploaded files, "
        "with sources you can inspect.</p></div>",
        unsafe_allow_html=True,
    )
    if not ready_docs:
        st.info("No documents are indexed yet. Open the **📄 Documents** page to upload PDF, DOCX, TXT, CSV or XLSX files.")
        return
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        cols[i % 2].button(example, key=f"example_{i}", use_container_width=True, on_click=_set_pending, args=(example,))


def _stream_answer(services: Services, conv_id: int, prompt: str, params: RAGParams) -> bool:
    """Render a streaming answer. Returns True if it finished without an error."""
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("⏳ _Thinking…_")
        st.button("⏹ Stop generating", key=f"stop_{conv_id}")
        box = st.empty()
        text = ""
        generator = services.chat.ask(conv_id, prompt, params)
        try:
            for event in generator:
                if event["type"] == "status":
                    status.markdown(f"⏳ _{event['text']}_")
                elif event["type"] == "token":
                    if not text:
                        status.empty()
                    text += event["text"]
                    box.markdown(text + "▌")
            status.empty()
            box.markdown(text)
            return True
        except AppError as exc:
            status.empty()
            if text:
                box.markdown(text)
            show_error(str(exc))
            return False
        except Exception:
            logger.exception("Unexpected error while answering")
            status.empty()
            if text:
                box.markdown(text)
            show_error("Something went wrong while generating the answer. Details were written to logs/app.log.")
            return False
        finally:
            generator.close()  # saves a partial answer if the stop button interrupted us


def render_chat(services: Services) -> None:
    settings = st.session_state["settings"]
    if st.session_state.get("flash_error"):
        show_error(st.session_state.pop("flash_error"))

    conv_id = st.session_state.get("conversation_id")
    if conv_id and services.chat.get_conversation(conv_id) is None:
        conv_id = st.session_state["conversation_id"] = None

    ready_docs = services.documents.list_documents("ready")
    _render_header(services, conv_id, ready_docs)
    messages = services.chat.get_messages(conv_id) if conv_id else []

    if not messages:
        _welcome(ready_docs)
    for msg in messages:
        _render_message(msg)

    prompt = st.chat_input("Ask a question about your documents…")
    if not prompt:
        prompt = st.session_state.pop("pending_prompt", None)
    if not prompt:
        return

    ok, message = services.models.preflight(
        settings.chat_model, settings.embedding_model, need_embedding=bool(ready_docs)
    )
    if not ok:
        show_error(message)
        st.code("\n".join(l for l in message.splitlines() if l.startswith("ollama pull")) or "ollama serve")
        return

    if conv_id is None:
        conv_id = services.chat.create_conversation().conversation_id
        st.session_state["conversation_id"] = conv_id
    scope = st.session_state.get("doc_filter") or None
    params = RAGParams.from_settings(settings, document_ids=scope)
    if _stream_answer(services, conv_id, prompt, params):
        st.rerun()
