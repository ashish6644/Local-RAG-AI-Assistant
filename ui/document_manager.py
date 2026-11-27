"""Documents page: upload, status, re-index, delete."""
from __future__ import annotations

import logging

import streamlit as st

from config.settings import ALLOWED_EXTENSIONS
from core.errors import AppError
from services.container import Services
from services.ingestion_service import IngestResult
from ui.components import STATUS_BADGES, format_dt, format_size, show_error

logger = logging.getLogger(__name__)


def _show_result(result: IngestResult) -> None:
    if result.status == "ready":
        st.success(result.message, icon="✅")
    elif result.status == "duplicate":
        st.info(result.message, icon="♻️")
    else:
        st.error(f"{result.filename}: {result.message}", icon="❌")
    for warning in result.warnings:
        st.warning(f"{result.filename}: {warning}")


def _request_delete(document_id: int) -> None:
    st.session_state["confirm_delete"] = document_id


def _cancel_delete() -> None:
    st.session_state["confirm_delete"] = None


def _request_rebuild() -> None:
    st.session_state["confirm_rebuild"] = True


def _cancel_rebuild() -> None:
    st.session_state["confirm_rebuild"] = False


def render_document_manager(services: Services) -> None:
    settings = st.session_state["settings"]
    st.header("Documents")
    st.caption(
        f"Supported: {', '.join(e.upper().lstrip('.') for e in ALLOWED_EXTENSIONS)} · max {settings.max_upload_mb} MB per file · "
        f"chunk size {settings.chunk_size}, overlap {settings.chunk_overlap}"
    )
    if message := st.session_state.pop("flash_success", None):
        st.success(message)

    with st.expander("🛠 Maintenance"):
        st.caption(
            "ChromaDB fixes the vector size when the index is created, so after changing the "
            "embedding model a per-document re-index can fail. Rebuild the whole index instead."
        )
        st.button("Rebuild vector index", key="rebuild_index", on_click=_request_rebuild)
        if st.session_state.get("confirm_rebuild"):
            st.warning("Delete and rebuild the search index for every uploaded document?")
            yes, no, _ = st.columns([1, 1, 4])
            if yes.button("Yes, rebuild", key="confirm_rebuild_yes", type="primary"):
                ok, preflight_message = services.models.preflight(
                    settings.chat_model, settings.embedding_model, need_chat=False
                )
                if not ok:
                    show_error(preflight_message)
                else:
                    with st.spinner("Rebuilding the vector index…"):
                        try:
                            indexed, failed = services.documents.rebuild_index(settings)
                        except AppError as exc:
                            show_error(str(exc))
                        else:
                            st.session_state["confirm_rebuild"] = False
                            st.session_state["flash_success"] = (
                                f"Rebuilt the index for {indexed} document(s)."
                                + (f" {failed} could not be rebuilt." if failed else "")
                            )
                            st.rerun()
            no.button("Cancel", key="confirm_rebuild_no", on_click=_cancel_rebuild)

    with st.form("upload_form", clear_on_submit=True):
        files = st.file_uploader(
            "Upload one or more documents", type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Upload & index", type="primary")

    if submitted:
        if not files:
            st.warning("Choose at least one file first.")
        else:
            ok, message = services.models.preflight(
                settings.chat_model, settings.embedding_model, need_chat=False, need_embedding=True
            )
            if not ok:
                show_error(message)
            else:
                progress = st.progress(0.0)
                for index, uploaded in enumerate(files, start=1):
                    with st.spinner(f"Processing {uploaded.name}…"):
                        try:
                            result = services.ingestion.ingest_bytes(uploaded.name, uploaded.getvalue(), settings)
                        except AppError as exc:
                            st.error(f"{uploaded.name}: {exc}", icon="❌")
                        except OSError as exc:
                            logger.exception("Could not store upload %s", uploaded.name)
                            st.error(f"{uploaded.name}: the file could not be saved ({exc}).", icon="❌")
                        else:
                            _show_result(result)
                    progress.progress(index / len(files))

    try:
        docs = services.documents.list_documents()
    except AppError as exc:
        show_error(str(exc))
        return
    st.subheader(f"Uploaded documents ({len(docs)})")
    if not docs:
        st.info("Nothing here yet. Upload a document above to get started.")
        return

    header = st.columns([3.2, 1, 1.6, 1.4, 1, 2])
    for col, label in zip(header, ["File", "Size", "Uploaded", "Status", "Chunks", "Actions"], strict=True):
        col.markdown(f"**{label}**")

    pending = st.session_state.get("confirm_delete")
    for doc in docs:
        cols = st.columns([3.2, 1, 1.6, 1.4, 1, 2])
        cols[0].markdown(f"{doc.filename}  \n<span class='msg-time'>{doc.file_type.upper()}</span>", unsafe_allow_html=True)
        cols[1].write(format_size(doc.file_size))
        cols[2].write(format_dt(doc.uploaded_at))
        cols[3].write(STATUS_BADGES.get(doc.processing_status, doc.processing_status))
        cols[4].write(str(doc.chunk_count))
        with cols[5]:
            a, b = st.columns(2)
            if a.button("🔄", key=f"reindex_{doc.document_id}", help="Re-index with the current chunking / embedding settings"):
                with st.spinner(f"Re-indexing {doc.filename}…"):
                    try:
                        result = services.documents.reindex(doc.document_id, settings)
                        if result:
                            _show_result(result)
                    except AppError as exc:
                        show_error(str(exc))
            b.button("🗑", key=f"delete_{doc.document_id}", help="Delete document", on_click=_request_delete, args=(doc.document_id,))
        if doc.processing_status == "failed" and doc.error_message:
            st.caption(f"⚠️ {doc.error_message}")
        if pending == doc.document_id:
            st.warning(f"Delete '{doc.filename}' and remove it from the search index?")
            yes, no, _ = st.columns([1, 1, 4])
            if yes.button("Yes, delete", key=f"confirm_{doc.document_id}", type="primary"):
                try:
                    services.documents.delete_document(doc.document_id)
                    st.session_state["confirm_delete"] = None
                    st.rerun()
                except AppError as exc:
                    show_error(str(exc))
            no.button("Cancel", key=f"cancel_{doc.document_id}", on_click=_cancel_delete)
