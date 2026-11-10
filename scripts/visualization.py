"""Seluruh definisi visualisasi Plotly untuk notebook dan integrasi dashboard berikutnya."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def style_figure(fig: go.Figure, title: str | None = None, height: int = 360, config: object | None = None) -> go.Figure:
    """Menerapkan gaya grafik Plotly yang sama dengan dashboard Ruang Sales asli."""
    primary = getattr(config, "PLOT_PRIMARY", "#1F6B4F") if config else "#1F6B4F"
    accent = getattr(config, "PLOT_ACCENT", "#C39231") if config else "#C39231"
    palette = getattr(
        config,
        "PLOT_PALETTE",
        [primary, accent, "#6285A0", "#9A6775", "#8A78A1"],
    ) if config else [primary, accent, "#6285A0", "#9A6775", "#8A78A1"]

    # Pengaturan berikut mengikuti helper plot() pada dashboard sumber:
    # latar transparan, font Plus Jakarta Sans, margin rapat, dan legend horizontal.
    fig.update_layout(
        template="none",
        title=title,
        height=height,
        margin=dict(l=8, r=25, t=56 if title else 20, b=15),
        font=dict(family="Plus Jakarta Sans, Arial", size=11, color="#53664F"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=palette,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor="#FFFDF7", font_size=12),
        transition_duration=0,
    )
    fig.update_layout(separators=",.")
    fig.update_xaxes(showgrid=False, automargin=True, zeroline=False)
    fig.update_yaxes(gridcolor="#E8E5D8", zerolinecolor="#C9D2BE", automargin=True)
    return fig


def _hex_to_rgba(color: str, alpha: float = 0.12) -> str:
    """Mengubah warna HEX menjadi RGBA untuk pita rentang antar-persentil."""
    value = color.lstrip("#")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _weekly_axis(fig: go.Figure, ylabel: str) -> go.Figure:
    """Merapikan sumbu minggu agar label W0–W51 mudah dibaca."""
    fig.update_xaxes(
        title="Minggu", tickmode="array", tickvals=[0, 8, 16, 24, 32, 40, 51],
        ticktext=["W0", "W8", "W16", "W24", "W32", "W40", "W51"],
    )
    fig.update_yaxes(title=ylabel, rangemode="tozero" if "asli" in ylabel.lower() or "volume" in ylabel.lower() else "normal")
    return fig


def create_weekly_sales_figure(weekly_stats: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Membuat grafik tren total volume penjualan untuk seluruh produk per minggu."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_stats["Minggu"], y=weekly_stats["Total_Penjualan"], mode="lines",
        line=dict(width=3), fill="tozeroy", name="Total penjualan",
        hovertemplate="W%{x}<br>Volume: %{y:,.0f}<extra></extra>",
    ))
    max_idx = int(weekly_stats["Total_Penjualan"].idxmax())
    min_idx = int(weekly_stats["Total_Penjualan"].idxmin())
    fig.add_trace(go.Scatter(
        x=weekly_stats.loc[[max_idx, min_idx], "Minggu"],
        y=weekly_stats.loc[[max_idx, min_idx], "Total_Penjualan"],
        mode="markers+text", text=["Tertinggi", "Terendah"], textposition="top center",
        marker=dict(size=10, color=["#C39231", "#9A6775"]), name="Titik ekstrem",
        hovertemplate="W%{x}<br>Volume: %{y:,.0f}<extra></extra>",
    ))
    _weekly_axis(fig, "Volume penjualan asli")
    return style_figure(fig, "Total penjualan seluruh produk · W0–W51", 380, config)


def create_top_products_figure(product_stats: pd.DataFrame, top_n: int = 10, config: object | None = None) -> go.Figure:
    """Membuat bar chart horizontal produk dengan total volume tertinggi."""
    top = product_stats.nlargest(int(top_n), "Total volume").sort_values("Total volume")
    fig = go.Figure(go.Bar(
        x=top["Total volume"], y=top["Produk"], orientation="h",
        marker_color="#1F6B4F", text=top["Total volume"], texttemplate="%{text:,.0f}",
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Total volume: %{x:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(title="Total volume penjualan · 52 minggu")
    fig.update_yaxes(title="Produk")
    return style_figure(fig, f"{top_n} produk dengan volume tertinggi", 390, config)


def create_zero_weeks_figure(product_stats: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Menampilkan distribusi banyaknya minggu tanpa penjualan pada setiap produk."""
    counts = product_stats["Minggu nol"].value_counts().sort_index().reindex(range(53), fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color="#C39231",
        hovertemplate="Minggu nol: %{x}<br>Jumlah produk: %{y}<extra></extra>",
    ))
    fig.update_xaxes(title="Jumlah minggu tanpa penjualan", dtick=5, range=[-0.5, 52.5])
    fig.update_yaxes(title="Jumlah produk")
    return style_figure(fig, "Distribusi minggu nol per produk", 340, config)


def create_exploration_figures(
    product_stats: pd.DataFrame,
    W_original: np.ndarray,
    selected_product: str,
    config: object | None = None,
) -> dict[str, go.Figure]:
    """Membuat histogram, scatter aktivitas-volume, dan detail seri satu produk."""
    values = np.asarray(W_original, dtype=float)
    if selected_product not in set(product_stats["Produk"].astype(str)):
        selected_product = str(product_stats.iloc[0]["Produk"])
    position = int(np.flatnonzero(product_stats["Produk"].astype(str).to_numpy() == selected_product)[0])

    hist = px.histogram(product_stats, x="Total volume", nbins=35, color_discrete_sequence=["#1F6B4F"])
    hist.update_xaxes(title="Total volume per produk · 52 minggu")
    hist.update_yaxes(title="Jumlah produk")
    hist = style_figure(hist, "Sebaran total penjualan", 330, config)

    scatter = px.scatter(
        product_stats, x="Minggu aktif (%)", y="Total volume", hover_name="Produk",
        color_discrete_sequence=["#1F6B4F"], opacity=0.60,
        hover_data={"Minggu nol": True, "Puncak mingguan": True},
    )
    scatter.update_traces(marker_size=7)
    scatter.update_xaxes(title="Minggu aktif (%)")
    scatter.update_yaxes(title="Total volume · 52 minggu")
    scatter = style_figure(scatter, "Hubungan aktivitas dan volume produk", 330, config)

    series = values[position]
    detail = go.Figure(go.Scatter(
        x=np.arange(values.shape[1]), y=series, mode="lines+markers",
        line=dict(width=2.5, color="#1F6B4F"), marker=dict(size=5), fill="tozeroy",
        fillcolor="rgba(31,107,79,.08)",
        hovertemplate="W%{x}<br>Volume: %{y:,.0f}<extra></extra>",
    ))
    _weekly_axis(detail, "Penjualan asli")
    detail = style_figure(detail, f"Perjalanan penjualan produk {selected_product}", 360, config)
    return {"sales_distribution": hist, "activity_vs_volume": scatter, "product_detail": detail}


def create_cluster_pattern_figure(
    W_or_X: np.ndarray,
    labels: np.ndarray,
    scale_label: str,
    config: object | None = None,
) -> go.Figure:
    """Membuat profil median mingguan tiap cluster dari data asli atau fitur Min-Max."""
    values = np.asarray(W_or_X, dtype=float)
    y = np.asarray(labels, dtype=int)
    fig = go.Figure()
    palette = getattr(config, "PLOT_PALETTE", ["#1F6B4F", "#C39231", "#6285A0"]) if config else ["#1F6B4F", "#C39231", "#6285A0"]
    for index, cluster_id in enumerate(sorted(np.unique(y))):
        group = values[y == cluster_id]
        median = np.median(group, axis=0)
        low = np.percentile(group, 10, axis=0)
        high = np.percentile(group, 90, axis=0)
        color = "#879080" if cluster_id == -1 else palette[index % len(palette)]
        band_color = _hex_to_rgba(color, 0.12)
        fig.add_trace(go.Scatter(
            x=np.arange(values.shape[1]), y=high, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip", name=f"C{cluster_id} P90",
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(values.shape[1]), y=low, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=band_color,
            showlegend=False, hoverinfo="skip", name=f"C{cluster_id} P10",
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(values.shape[1]), y=median, mode="lines",
            line=dict(width=3, color=color, dash="dot" if cluster_id == -1 else "solid"),
            name="Noise" if cluster_id == -1 else f"C{cluster_id} · n={len(group)}",
            hovertemplate="W%{x}<br>Median: %{y:.3f}<extra></extra>",
        ))
    _weekly_axis(fig, scale_label)
    return style_figure(fig, f"Profil mingguan cluster · {scale_label}", 390, config)


def create_cluster_contribution_figure(profiles: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Membandingkan proporsi jumlah produk dan proporsi volume setiap cluster."""
    labels = ["Noise" if value == -1 else f"C{value}" for value in profiles["Cluster"]]
    fig = go.Figure()
    fig.add_bar(y=labels, x=profiles["Persen_Produk"], orientation="h", name="Porsi produk",
                marker_color="#1F6B4F", text=profiles["Persen_Produk"], texttemplate="%{text:.1f}%",
                textposition="outside", cliponaxis=False)
    fig.add_bar(y=labels, x=profiles["Porsi_Volume"], orientation="h", name="Porsi volume",
                marker_color="#C39231", text=profiles["Porsi_Volume"], texttemplate="%{text:.1f}%",
                textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Persentase dari keseluruhan", ticksuffix="%", range=[0, 118])
    fig.update_yaxes(title="Cluster", autorange="reversed")
    return style_figure(fig, "Porsi produk dibandingkan kontribusi volume", 390, config)


def create_metrics_comparison_figure(metrics: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Menyajikan empat indeks evaluasi metode dalam panel perbandingan Plotly."""
    metric_specs = [
        ("Silhouette_FastDTW", "Silhouette FastDTW ↑", True),
        ("DBI_Medoid_FastDTW", "DBI medoid FastDTW ↓", False),
        ("CHI_MinMax_Euclidean", "CHI Euclidean Min-Max ↑", True),
        ("Stabilitas_ARI", "Stabilitas ARI ↑", True),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[spec[1] for spec in metric_specs], horizontal_spacing=0.16, vertical_spacing=0.20)
    for index, (metric, label, higher) in enumerate(metric_specs):
        row, col = index // 2 + 1, index % 2 + 1
        ordered = metrics.sort_values(metric, ascending=not higher, na_position="last")
        fig.add_trace(go.Bar(
            x=ordered[metric], y=ordered["Metode"], orientation="h",
            marker_color=["#C39231" if method == getattr(config, "DEFAULT_METHOD", "FuzzyCMeans") else "#1F6B4F" for method in ordered["Metode"]],
            text=ordered[metric].map(lambda value: "tidak diuji" if pd.isna(value) else f"{value:.3f}"),
            textposition="outside", cliponaxis=False, showlegend=False,
            hovertemplate="%{y}<br>Nilai: %{x:.4f}<extra></extra>",
        ), row=row, col=col)
        fig.update_xaxes(title_text=label, row=row, col=col, showgrid=True, gridcolor="#E8E5D8")
        fig.update_yaxes(autorange="reversed", row=row, col=col)
    return style_figure(fig, "Evaluasi enam metode clustering", 690, config)


def create_coverage_figure(metrics: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Menampilkan coverage dan jumlah cluster aktual untuk keenam metode."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=metrics["Metode"], y=100 * metrics["Coverage"], name="Coverage (%)", marker_color="#1F6B4F", secondary_y=False)
    fig.add_scatter(x=metrics["Metode"], y=metrics["Jumlah_Cluster"], name="Jumlah cluster", mode="lines+markers", line=dict(color="#C39231", width=2.5), secondary_y=True)
    fig.update_yaxes(title_text="Produk terkelompok (%)", range=[0, 110], secondary_y=False)
    fig.update_yaxes(title_text="Jumlah cluster aktual", rangemode="tozero", secondary_y=True)
    fig.update_xaxes(title="Metode", tickangle=-20)
    return style_figure(fig, "Coverage dan jumlah cluster aktual", 430, config)


def create_similarity_heatmap(matrix: pd.DataFrame, metric_name: str = "ARI", config: object | None = None) -> go.Figure:
    """Membuat heatmap kesepakatan partisi berdasarkan ARI atau NMI."""
    fig = px.imshow(
        matrix, zmin=0, zmax=1, text_auto=".2f", aspect="auto",
        color_continuous_scale=["#F5F0DD", "#A9BE9F", "#1F6B4F"],
        labels={"color": metric_name},
    )
    return style_figure(fig, f"Kesamaan hasil clustering antar-metode · {metric_name}", 480, config)


def create_tuning_k_figure(tuning_per_k: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Membandingkan perubahan Silhouette dan DBI terhadap jumlah cluster aktual."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Silhouette FastDTW", "DBI medoid FastDTW"])
    for method, subset in tuning_per_k.groupby("Metode"):
        subset = subset.sort_values("k_Aktual")
        fig.add_trace(go.Scatter(x=subset["k_Aktual"], y=subset["Silhouette_FastDTW"], mode="lines+markers", name=method, legendgroup=method), row=1, col=1)
        fig.add_trace(go.Scatter(x=subset["k_Aktual"], y=subset["DBI_Medoid_FastDTW"], mode="lines+markers", name=method, legendgroup=method, showlegend=False), row=1, col=2)
    fig.update_xaxes(title="Jumlah cluster aktual (k)", dtick=1, row=1, col=1)
    fig.update_xaxes(title="Jumlah cluster aktual (k)", dtick=1, row=1, col=2)
    fig.update_yaxes(title="Silhouette (lebih tinggi lebih baik)", row=1, col=1)
    fig.update_yaxes(title="DBI (lebih rendah lebih baik)", row=1, col=2)
    return style_figure(fig, "Tuning jumlah cluster · selain DBSCAN", 520, config)


def create_dbscan_tuning_figure(dbscan_tuning: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Menampilkan trade-off Silhouette non-noise dan proporsi noise terhadap eps DBSCAN."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Silhouette non-noise vs eps", "Proporsi noise vs eps"])
    for min_samples, subset in dbscan_tuning.groupby("min_samples"):
        subset = subset.sort_values("eps")
        fig.add_trace(go.Scatter(x=subset["eps"], y=subset["Silhouette_FastDTW"], mode="lines+markers", name=f"min_samples={min_samples}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=subset["eps"], y=100 * (1 - subset["Coverage"]), mode="lines+markers", name=f"min_samples={min_samples}", showlegend=False), row=1, col=2)
    fig.update_xaxes(title="eps (skala FastDTW)", row=1, col=1)
    fig.update_xaxes(title="eps (skala FastDTW)", row=1, col=2)
    fig.update_yaxes(title="Silhouette FastDTW", row=1, col=1)
    fig.update_yaxes(title="Produk noise (%)", row=1, col=2)
    return style_figure(fig, "Tuning parameter DBSCAN", 500, config)


def create_silhouette_distribution_figure(detail: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Membuat histogram silhouette per produk untuk melihat anggota yang salah tempat."""
    fig = px.histogram(detail, x="Silhouette", color=detail["Cluster"].astype(str), nbins=35, barmode="overlay", opacity=0.65)
    fig.add_vline(x=0, line_dash="dash", line_color="#9A6775")
    fig.update_xaxes(title="Silhouette individual · FastDTW", range=[-1, 1])
    fig.update_yaxes(title="Jumlah produk")
    return style_figure(fig, "Distribusi Silhouette per produk", 380, config)


def create_silhouette_box_figure(detail: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Membuat box plot silhouette individual menurut cluster."""
    frame = detail.copy()
    frame["Cluster_Label"] = frame["Cluster"].map(lambda value: f"C{value}")
    fig = px.box(frame, x="Cluster_Label", y="Silhouette", color="Cluster_Label", points="outliers")
    fig.add_hline(y=0, line_dash="dash", line_color="#9A6775")
    fig.update_yaxes(title="Silhouette individual", range=[-1, 1])
    fig.update_xaxes(title="Cluster")
    return style_figure(fig, "Sebaran silhouette pada setiap cluster", 380, config)


def create_pca_scatter_figure(pca_data: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Membuat scatter PCA dua dimensi untuk visualisasi tambahan struktur data."""
    frame = pca_data.copy()
    frame["Cluster_Label"] = frame["Cluster"].map(lambda value: "Noise" if value == -1 else f"C{value}")
    fig = px.scatter(frame, x="PC1", y="PC2", color="Cluster_Label", opacity=0.65,
                     hover_data={"Cluster": True}, color_discrete_sequence=getattr(config, "PLOT_PALETTE", None) if config else None)
    fig.update_xaxes(title="PC1 · visualisasi saja")
    fig.update_yaxes(title="PC2 · visualisasi saja")
    return style_figure(fig, "Proyeksi PCA · bukan jarak yang digunakan clustering", 460, config)


def create_weekly_diagnostics_figure(weekly_stats: pd.DataFrame, config: object | None = None) -> go.Figure:
    """Menampilkan total penjualan mingguan dan persentase produk aktif pada panel terpisah."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=["Total penjualan mingguan", "Produk yang aktif (%)"])
    fig.add_trace(go.Scatter(x=weekly_stats["Minggu"], y=weekly_stats["Total_Penjualan"], mode="lines+markers", name="Total penjualan", line=dict(color="#1F6B4F", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=weekly_stats["Minggu"], y=weekly_stats["Persen_Produk_Aktif"], mode="lines+markers", name="Produk aktif", line=dict(color="#C39231", width=2.5)), row=2, col=1)
    fig.update_xaxes(title="Minggu", tickmode="array", tickvals=[0, 8, 16, 24, 32, 40, 51], ticktext=["W0", "W8", "W16", "W24", "W32", "W40", "W51"], row=2, col=1)
    fig.update_yaxes(title="Volume", row=1, col=1)
    fig.update_yaxes(title="Produk aktif (%)", range=[0, 100], row=2, col=1)
    for week in [24, 26]:
        fig.add_vline(x=week, line_dash="dash", line_color="#9A6775", opacity=0.45)
    return style_figure(fig, "Audit aktivitas penjualan per minggu", 600, config)


def build_dashboard_figures(
    W_original: np.ndarray,
    product_codes: np.ndarray,
    product_stats: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    metrics: pd.DataFrame,
    ari: pd.DataFrame,
    nmi: pd.DataFrame,
    tuning_per_k: pd.DataFrame,
    tuning_dbscan: pd.DataFrame,
    silhouette_detail: pd.DataFrame,
    pca_data: pd.DataFrame,
    selected_product: str,
    config: object,
) -> dict[str, go.Figure]:
    """Membuat kumpulan grafik lengkap yang dapat langsung dipakai ulang pada dashboard."""
    figures = {
        "01_weekly_sales": create_weekly_sales_figure(weekly_stats, config),
        "02_top_products": create_top_products_figure(product_stats, 10, config),
        "03_zero_weeks": create_zero_weeks_figure(product_stats, config),
    }
    figures.update({f"04_{name}": figure for name, figure in create_exploration_figures(
        product_stats, W_original, selected_product, config
    ).items()})
    figures["05_cluster_original"] = create_cluster_pattern_figure(W_original, labels, "Median volume · penjualan asli", config)
    figures["06_cluster_minmax"] = create_cluster_pattern_figure(X_scaled, labels, "Median nilai Min-Max · pola ternormalisasi", config)
    figures["07_cluster_contribution"] = create_cluster_contribution_figure(profiles, config)
    figures["08_metrics_comparison"] = create_metrics_comparison_figure(metrics, config)
    figures["09_coverage"] = create_coverage_figure(metrics, config)
    figures["10_ari_heatmap"] = create_similarity_heatmap(ari, "ARI", config)
    figures["11_nmi_heatmap"] = create_similarity_heatmap(nmi, "NMI", config)
    figures["12_tuning_k"] = create_tuning_k_figure(tuning_per_k, config)
    figures["13_dbscan_tuning"] = create_dbscan_tuning_figure(tuning_dbscan, config)
    figures["14_silhouette_distribution"] = create_silhouette_distribution_figure(silhouette_detail, config)
    figures["15_silhouette_by_cluster"] = create_silhouette_box_figure(silhouette_detail, config)
    figures["16_pca_scatter"] = create_pca_scatter_figure(pca_data, config)
    figures["17_weekly_diagnostics"] = create_weekly_diagnostics_figure(weekly_stats, config)
    return figures
