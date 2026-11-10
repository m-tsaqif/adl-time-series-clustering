"""03 Segmentasi; entry point halaman dashboard."""

from scripts.dashboard_data import load_dashboard_bundle
from scripts.dashboard_pages import render_segmentation_page


def render_page() -> None:
    """Memuat data dan merender halaman 03 segmentasi."""
    render_segmentation_page(load_dashboard_bundle())
