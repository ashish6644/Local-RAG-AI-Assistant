"""Local RAG AI Assistant - Streamlit entry point.  Run with:  streamlit run app.py"""
from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(page_title="Local RAG AI Assistant", page_icon="📚", layout="wide")

from config.settings import load_settings, setup_logging  # noqa: E402
from core.errors import AppError, ConfigError  # noqa: E402
from services.container import Services, build_services  # noqa: E402
from ui.chat_interface import render_chat  # noqa: E402
from ui.components import inject_css  # noqa: E402
from ui.document_manager import render_document_manager  # noqa: E402
from ui.settings import render_settings  # noqa: E402
from ui.sidebar import PAGES, render_sidebar  # noqa: E402

logger = logging.getLogger("app")


@st.cache_resource(show_spinner="Starting up…")
def get_services() -> Services:
    settings = load_settings()
    setup_logging(settings)
    return build_services(settings)


def main() -> None:
    inject_css()
    try:
        services = get_services()
    except ConfigError as exc:
        st.error(f"Configuration problem.\n\n{exc}\n\nFix your `.env` file and reload the page.")
        st.stop()
    except AppError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        logger.exception("Startup failed")
        st.error("The application could not start. Details were written to logs/app.log.")
        st.stop()

    if "settings" not in st.session_state:
        st.session_state["settings"] = load_settings()  # picks up saved UI overrides
    st.session_state.setdefault("page", PAGES[0])
    page = render_sidebar(services)
    if page == PAGES[0]:
        render_chat(services)
    elif page == PAGES[1]:
        render_document_manager(services)
    else:
        render_settings(services)


main()
