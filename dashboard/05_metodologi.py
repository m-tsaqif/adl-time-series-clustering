"""05 Metodologi; entry point halaman dashboard."""

from scripts.dashboard_data import load_dashboard_bundle
from scripts.dashboard_pages import render_methodology_page


def render_page() -> None:
    """Memuat data dan merender halaman 05 metodologi."""
    render_methodology_page(load_dashboard_bundle())
