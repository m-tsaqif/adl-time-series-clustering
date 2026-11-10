"""02 Eksplorasi; entry point halaman dashboard."""

from scripts.dashboard_data import load_dashboard_bundle
from scripts.dashboard_pages import render_exploration_page


def render_page() -> None:
    """Memuat data dan merender halaman 02 eksplorasi."""
    render_exploration_page(load_dashboard_bundle())
