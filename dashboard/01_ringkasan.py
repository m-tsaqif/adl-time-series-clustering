"""01 Ringkasan; entry point halaman dashboard."""

from scripts.dashboard_data import load_dashboard_bundle
from scripts.dashboard_pages import render_overview_page


def render_page() -> None:
    """Memuat data dan merender halaman 01 ringkasan."""
    render_overview_page(load_dashboard_bundle())
