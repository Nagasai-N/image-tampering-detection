"""
styles_injector.py
──────────────────
Drop this next to styles.css.  In your Streamlit app just do:

    from styles_injector import inject_styles
    inject_styles()          # call once, near the top of app.py

It reads styles.css from the same folder, wraps it in <style> tags,
and pushes it into the page.  The file is read only once per session
thanks to st.cache_resource.
"""

from pathlib import Path
import streamlit as st

# ── resolve the CSS file path relative to THIS file ──────────
_CSS_PATH = Path(__file__).resolve().parent / "styles.css"


@st.cache_resource
def _load_css() -> str:
    """Read the stylesheet once and cache it for the whole session."""
    return _CSS_PATH.read_text(encoding="utf-8")


def inject_styles() -> None:
    """Inject styles.css into the current Streamlit page."""
    css = _load_css()
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
