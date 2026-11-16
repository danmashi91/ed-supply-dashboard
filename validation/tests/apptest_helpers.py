"""Helpers for driving the Streamlit app headlessly with streamlit.testing.v1.AppTest.

The app uses st.navigation, which AppTest cannot switch pages on. To exercise one page
we write a tiny wrapper copy of app.py (next to app.py so APP_ROOT resolves) that
marks the requested page as the default, run it, and delete the wrapper afterwards.
The application logic is unchanged; only the default page differs.
"""
from __future__ import annotations
import re, os, uuid
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
PAGES = {"overview": "page_overview", "item_explorer": "page_item_explorer", "trends": "page_trends",
         "scenarios": "page_scenarios", "predictive": "page_predictive_models", "data": "page_data"}


def open_page(page: str, timeout: int = 120) -> AppTest:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    src = src.replace(", default=True", "")
    fn = PAGES[page]
    src = re.sub(rf"(st\.Page\({fn},[^\n]*?)\)(,?\n)", lambda m: m.group(1) + ", default=True)" + m.group(2), src, count=1)
    assert "default=True" in src
    wrapper = ROOT / f"_apptest_{uuid.uuid4().hex[:8]}.py"
    wrapper.write_text(src, encoding="utf-8")
    try:
        at = AppTest.from_file(str(wrapper), default_timeout=timeout)
        at.run()
    finally:
        pass
    at._wrapper = wrapper
    return at


def close(at: AppTest) -> None:
    w = getattr(at, "_wrapper", None)
    if w and w.exists():
        w.unlink()


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}
