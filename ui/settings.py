"""Settings page: Ollama status, model selection, generation / chunking / retrieval parameters."""
from __future__ import annotations

import streamlit as st

from config.settings import save_user_overrides, with_updates
from core.errors import AppError
from core.text_splitter import STRATEGIES
from services.container import Services
from services.model_service import pull_command
from ui.components import show_error


def _model_select(label: str, options: list[str], current: str, key: str) -> str:
    choices = list(options)
    if current not in choices:
        choices.insert(0, current)
    return st.selectbox(
        label, choices, index=choices.index(current), key=key,
        format_func=lambda m: m if m in options else f"{m} (not installed)",
    )


def render_settings(services: Services) -> None:
    current = st.session_state["settings"]
    st.header("Settings")

    running = services.models.is_running()
    installed = services.models.installed_models() if running else []
    c1, c2 = st.columns([3, 1])
    if running:
        c1.success(f"Ollama is running at {services.settings.ollama_base_url} · {len(installed)} model(s) installed")
    else:
        c1.error("Ollama is not reachable. Install it from https://ollama.com/download and start it (open the app or run `ollama serve`).")
    if c2.button("🔄 Refresh"):
        st.rerun()

    st.subheader("Models")
    chat_options = services.models.chat_models() or installed
    chat_model = _model_select("Chat model", chat_options, current.chat_model, "set_chat_model")
    embed_model = _model_select("Embedding model", services.models.embedding_models(), current.embedding_model, "set_embed_model")
    missing = [m for m in dict.fromkeys([chat_model, embed_model]) if running and not services.client.model_installed(m, installed)]
    if missing:
        st.warning("Missing models. Download them in a terminal:")
        st.code("\n".join(pull_command(m) for m in missing), language="powershell")
    if embed_model != current.embedding_model:
        st.info("Changing the embedding model requires re-indexing existing documents (🔄 on the Documents page).")

    st.subheader("Generation")
    g1, g2 = st.columns(2)
    temperature = g1.slider("Temperature", 0.0, 2.0, float(current.temperature), 0.05)
    top_p = g2.slider("Top-p", 0.05, 1.0, float(current.top_p), 0.05)
    max_tokens = g1.number_input("Max output tokens", 16, 32768, int(current.max_output_tokens), 64)
    num_ctx = g2.number_input("Context window (tokens)", 512, 131072, int(current.context_window), 512)

    st.subheader("Chunking (applies when documents are indexed)")
    k1, k2, k3 = st.columns(3)
    chunk_size = k1.number_input("Chunk size (characters)", 100, 8000, int(current.chunk_size), 50)
    chunk_overlap = k2.number_input("Chunk overlap", 0, 4000, int(current.chunk_overlap), 10)
    strategy = k3.selectbox("Chunking strategy", list(STRATEGIES), index=list(STRATEGIES).index(current.chunking_strategy))

    st.subheader("Retrieval")
    r1, r2 = st.columns(2)
    top_k = r1.slider("Top K", 1, 20, int(current.top_k))
    threshold = r2.slider("Similarity threshold", 0.0, 1.0, float(current.similarity_threshold), 0.01)
    search = r1.selectbox("Search strategy", ["similarity", "mmr"], index=["similarity", "mmr"].index(current.search_strategy),
                          help="MMR diversifies results to reduce near-duplicate chunks.")
    history_window = r2.slider("Chat history window (messages)", 0, 50, int(current.history_window))
    rewrite = st.toggle("History-aware question rewriting", value=current.use_query_rewriting)
    multi = st.toggle("Multi-query retrieval (extra LLM call)", value=current.use_multi_query)

    if st.button("💾 Save settings", type="primary"):
        try:
            updated = with_updates(
                current, chat_model=chat_model, embedding_model=embed_model, temperature=temperature,
                top_p=top_p, max_output_tokens=int(max_tokens), context_window=int(num_ctx),
                chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap), chunking_strategy=strategy,
                top_k=top_k, similarity_threshold=threshold, search_strategy=search,
                history_window=history_window, use_query_rewriting=rewrite, use_multi_query=multi,
            )
            save_user_overrides(updated)
            st.session_state["settings"] = updated
            st.success("Settings saved.")
        except (AppError, OSError) as exc:
            show_error(str(exc))

    st.subheader("Connection test")
    if st.button("🧪 Test models"):
        with st.spinner("Contacting Ollama (the first call may take a while while the model loads)…"):
            result = services.models.test_connection(chat_model, embed_model)
        if not result["running"]:
            show_error(result.get("error", "Ollama is not reachable."))
        else:
            st.write(f"Ollama version: `{result['version']}`")
            for label, key in (("Chat model", "chat"), ("Embedding model", "embedding")):
                item = result[key]
                (st.success if item["ok"] else st.error)(f"{label}: {item['detail']}")
