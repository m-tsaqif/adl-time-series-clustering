"""Aplikasi utama dashboard Streamlit Ruang Sales."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Ruang Sales · Pola Penjualan",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from scripts.dashboard_pages import apply_dashboard_css, render_brand_sidebar, render_dashboard_footer

PAGE_SPECS = [
    ("Ringkasan Penjualan", "ringkasan", "dashboard.01_ringkasan", True),
    ("Eksplorasi Data", "eksplorasi", "dashboard.02_eksplorasi", False),
    ("Segmentasi Produk", "segmentasi", "dashboard.03_segmentasi", False),
    ("Perbandingan Metode", "perbandingan", "dashboard.04_perbandingan", False),
    ("Metodologi", "metodologi", "dashboard.05_metodologi", False),
]

PAGE_OBJECTS = {}
for title, slug, module_name, is_default in PAGE_SPECS:
    page_module = import_module(module_name)
    PAGE_OBJECTS[title] = st.Page(
        page_module.render_page,
        title=title,
        url_path=slug,
        default=is_default,
    )

selected_page = st.navigation(list(PAGE_OBJECTS.values()), position="hidden")
apply_dashboard_css()

with st.sidebar:
    render_brand_sidebar(selected_page.title, PAGE_OBJECTS)

selected_page.run()
render_dashboard_footer()
