"""Fungsi renderer untuk halaman-halaman dashboard Streamlit Ruang Sales."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import config
from scripts.dashboard_data import (
    attach_product_codes_to_silhouette,
    build_results_zip,
    calculate_silhouette_for_method,
    format_cluster_label,
    format_method_name,
    get_membership_columns,
    get_method_labels,
    get_method_profiles,
    get_method_tuning,
)
from scripts.visualization import (
    create_cluster_contribution_figure,
    create_cluster_pattern_figure,
    create_coverage_figure,
    create_dbscan_tuning_figure,
    create_exploration_figures,
    create_metrics_comparison_figure,
    create_pca_scatter_figure,
    create_similarity_heatmap,
    create_silhouette_box_figure,
    create_silhouette_distribution_figure,
    create_top_products_figure,
    create_tuning_k_figure,
    create_weekly_diagnostics_figure,
    create_weekly_sales_figure,
    create_zero_weeks_figure,
)


def apply_dashboard_css() -> None:
    """Memuat stylesheet dashboard dari folder assets bila tersedia."""
    css_path = config.PROJECT_ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_brand_sidebar(active_page: str, page_objects: dict[str, Any]) -> None:
    """Menampilkan sidebar dengan identitas, tombol navigasi, dan gaya asli Ruang Sales."""
    page_names = [
        "Ringkasan Penjualan",
        "Eksplorasi Data",
        "Segmentasi Produk",
        "Perbandingan Metode",
        "Metodologi",
    ]
    icons = ["space_dashboard", "query_stats", "category", "equalizer", "menu_book"]
    st.markdown(
        '<div class="brand"><div class="brand-icon">↗</div><div>'
        '<div class="brand-name">Ruang Sales</div>'
        '<div class="brand-sub">CERITA DI BALIK ANGKA</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("JELAJAHI ANALISIS")
    for page_name, icon in zip(page_names, icons):
        st.button(
            page_name,
            key=f"nav_{page_name}",
            icon=f":material/{icon}:",
            width="stretch",
            type="primary" if active_page == page_name else "secondary",
            on_click=lambda target=page_name: st.switch_page(page_objects[target]),
        )
    st.markdown(
        '<div class="side-note"><b>811 produk · 52 minggu</b><br>'
        'Mulai dari gambaran penjualan, lalu kenali pola setiap kelompok produk.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "SALES TRANSACTIONS\n\nHasil analisis tersimpan · 6 metode\n\n"
        "Periode mengikuti urutan W0–W51."
    )


def render_page_heading(number: str, title: str, subtitle: str) -> None:
    """Menampilkan heading dengan hierarki dan jarak seperti dashboard asli."""
    st.markdown(
        f'<div class="eyebrow">{number} / RUANG SALES</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.caption(subtitle)


def render_dashboard_footer() -> None:
    """Menampilkan footer tipis yang digunakan pada setiap halaman dashboard."""
    st.markdown(
        '<div class="footer">RUANG SALES &nbsp; · &nbsp; Sales Transactions · W0–W51 '
        '&nbsp; / &nbsp; Eksplorasi pola penjualan produk</div>',
        unsafe_allow_html=True,
    )


def render_plotly(fig: Any, key: str | None = None) -> None:
    """Menampilkan grafik dengan toolbar, lebar, dan tema seperti dashboard asli."""
    # Helper ini mempertahankan pengaturan plot() pada dashboard sumber:
    # tanpa tema otomatis Streamlit, tanpa logo Plotly, dan mode bar saat hover.
    fig.update_layout(title=None, margin=dict(l=8, r=25, t=20, b=15))
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        key=key,
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )


def render_kpi(label: str, value: str, detail: str = "") -> None:
    """Menampilkan satu indikator utama dalam kartu dashboard."""
    st.markdown(
        '<div class="kpi"><div class="kpi-label">' + label + '</div>'
        '<div class="kpi-value">' + value + '</div>'
        '<div class="kpi-detail">' + detail + '</div></div>',
        unsafe_allow_html=True,
    )


def render_warning_panel(warnings: list[str], status: str) -> None:
    """Menampilkan catatan kualitas model agar hasil eksploratif tidak disalahartikan."""
    if status == "FALLBACK_NO_SHAPE_GUARDRAIL":
        st.warning(
            "**Interpretasi penting — belum ada model segmentasi shape-only yang lolos guardrail.** "
            "Metode dengan skor tertinggi hanya ditampilkan sebagai pemimpin skor sementara, "
            "bukan rekomendasi final. Silhouette, stabilitas, dan asosiasi minggu nol perlu dibaca bersama."
        )
    for warning in warnings:
        st.caption(f"• {warning}")


def render_overview_page(bundle: dict[str, Any]) -> None:
    """Merender halaman ringkasan penjualan dan temuan analisis utama."""
    sales = bundle["sales"]
    st.markdown(
        '<div class="hero"><div class="eyebrow">SALES TRANSACTIONS / W0–W51</div>'
        '<h1>Penjualan, dalam satu pandangan.</h1>'
        '<p>Pahami pergerakan penjualan, temukan produk yang menonjol, dan kenali kelompok dengan pola yang serupa.</p>'
        f'<div class="hero-badge">{len(sales):,} PRODUK &nbsp; / &nbsp; {len(config.WEEK_COLUMNS)} MINGGU &nbsp; / &nbsp; {len(bundle["methods"])} METODE</div></div>',
        unsafe_allow_html=True,
    )
    metadata = bundle["metadata"]
    zero_share = 100 * (bundle["W"] == 0).mean()

    cols = st.columns(4)
    with cols[0]:
        render_kpi("JUMLAH PRODUK", f"{len(sales):,}", "Produk unik yang dianalisis")
    with cols[1]:
        render_kpi("TOTAL VOLUME", f"{bundle['W'].sum():,.0f}", "Akumulasi penjualan W0–W51")
    with cols[2]:
        render_kpi("PERIODE PENGAMATAN", f"{len(config.WEEK_COLUMNS)} minggu", "Urutan minggu sesuai data sumber")
    with cols[3]:
        render_kpi("PENGAMATAN BERNILAI NOL", f"{zero_share:.1f}%", "Dari seluruh pasangan produk × minggu")

    render_warning_panel(metadata.get("selection_warnings", []), metadata.get("selection_status", ""))

    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        st.subheader("Perjalanan penjualan mingguan")
        st.caption("Jumlah seluruh unit penjualan di setiap minggu. W0–W51 adalah urutan minggu, bukan tanggal kalender.")
        render_plotly(create_weekly_sales_figure(bundle["weekly_stats"], config), key="overview_weekly")
    with right, st.container(border=True):
        st.subheader("Sebaran minggu tanpa penjualan")
        st.caption(f"Sekitar {zero_share:.1f}% observasi produk-minggu bernilai nol.")
        render_plotly(create_zero_weeks_figure(bundle["product_stats"], config), key="overview_zero")

    with st.container(border=True):
        st.subheader("Produk dengan volume tertinggi")
        render_plotly(create_top_products_figure(bundle["product_stats"], 10, config), key="overview_top")
        top = bundle["product_stats"].nlargest(10, "Total volume")[ ["Produk", "Total volume", "Minggu aktif (%)", "Minggu puncak"] ]
        st.dataframe(top, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="callout"><b>Cara membaca:</b> tren total menunjukkan dinamika agregat, '
        'sedangkan distribusi minggu nol memperlihatkan tingkat kontinuitas penjualan antarproduk. '
        'Keduanya merupakan konteks sebelum pola tiap produk dibandingkan menggunakan FastDTW.</div>',
        unsafe_allow_html=True,
    )


def render_exploration_page(bundle: dict[str, Any]) -> None:
    """Merender halaman eksplorasi distribusi produk dan detail deret mingguan."""
    render_page_heading("02", "Kenali data sebelum mengelompokkan.", "Telusuri distribusi volume, tingkat aktivitas, dan perjalanan penjualan setiap produk.")
    stats = bundle["product_stats"].copy()
    product_codes = bundle["product_codes"].astype(str).tolist()
    selected_product = st.selectbox("Pilih produk untuk melihat deret mingguan", product_codes, index=0, key="explore_product")
    figures = create_exploration_figures(stats, bundle["W"], selected_product, config)

    a, b = st.columns(2)
    with a, st.container(border=True):
        st.subheader("Distribusi total volume")
        render_plotly(figures["sales_distribution"], key="explore_distribution")
    with b, st.container(border=True):
        st.subheader("Aktivitas dan volume")
        render_plotly(figures["activity_vs_volume"], key="explore_activity")
    with st.container(border=True):
        st.subheader(f"Deret mingguan · {selected_product}")
        render_plotly(figures["product_detail"], key="explore_series")

    product_row = stats.loc[stats["Produk"] == selected_product].iloc[0]
    cols = st.columns(4)
    with cols[0]:
        render_kpi("TOTAL VOLUME", f"{product_row['Total volume']:,.0f}", "Seluruh 52 minggu")
    with cols[1]:
        render_kpi("MINGGU NOL", f"{int(product_row['Minggu nol'])}", "Tidak ada penjualan")
    with cols[2]:
        render_kpi("MINGGU AKTIF", f"{product_row['Minggu aktif (%)']:.1f}%", "Penjualan lebih dari nol")
    with cols[3]:
        render_kpi("PUNCAK MINGGUAN", f"{product_row['Puncak mingguan']:,.0f}", str(product_row["Minggu puncak"]))

    st.subheader("Telusuri tabel produk")
    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        query = st.text_input("Cari kode produk", placeholder="Contoh: P43", key="explore_search").strip().upper()
    with f2:
        zero_range = st.slider("Jumlah minggu nol", min_value=0, max_value=52, value=(0, 52), key="explore_zero_range")
    with f3:
        min_volume = st.number_input("Volume minimum", min_value=0.0, value=0.0, step=10.0, key="explore_min_volume")
    filtered = stats.loc[
        stats["Minggu nol"].between(zero_range[0], zero_range[1])
        & (stats["Total volume"] >= min_volume)
        & stats["Produk"].str.upper().str.contains(query, regex=False)
    ].copy()
    st.caption(f"Menampilkan {len(filtered):,} dari {len(stats):,} produk. Filter ini hanya mengubah tabel, tidak mengubah label clustering.")
    st.dataframe(filtered, hide_index=True, use_container_width=True, height=360)
    st.download_button(
        "Unduh tabel hasil filter", filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="eksplorasi_produk_terfilter.csv", mime="text/csv", key="download_explore",
    )


def render_segmentation_page(bundle: dict[str, Any]) -> None:
    """Merender profil pola, kontribusi, dan anggota cluster untuk metode terpilih."""
    render_page_heading("03", "Setiap produk punya cerita pola.", "Pilih metode, skala visualisasi, dan cluster untuk membaca karakter kelompok.")
    methods = bundle["methods"]
    suggested = config.DEFAULT_METHOD
    default_index = methods.index(suggested) if suggested in methods else 0
    method = st.selectbox("Metode segmentasi", methods, index=default_index, format_func=format_method_name, key="segment_method")
    labels = get_method_labels(bundle, method)
    profiles = get_method_profiles(bundle, method)
    metrics = bundle["metrics"].loc[bundle["metrics"]["Metode"] == method].iloc[0]

    cols = st.columns(4)
    with cols[0]:
        render_kpi("JUMLAH CLUSTER", str(int(metrics["Jumlah_Cluster"])), format_method_name(method))
    with cols[1]:
        render_kpi("COVERAGE", f"{100*metrics['Coverage']:.1f}%", f"{int(metrics['Jumlah_Noise'])} produk noise")
    with cols[2]:
        render_kpi("SILHOUETTE", f"{metrics['Silhouette_FastDTW']:.3f}", "Lebih tinggi relatif lebih baik")
    with cols[3]:
        render_kpi("η² MINGGU NOL", f"{metrics['Eta2_Minggu_Nol']:.3f}", "Asosiasi label dengan frekuensi nol")

    scale_choice = st.radio("Skala profil pola", ["Penjualan asli", "Pola Min-Max"], horizontal=True, key="segment_scale")
    values = bundle["W"] if scale_choice == "Penjualan asli" else bundle["X_scaled"]
    y_title = "Median volume · penjualan asli" if scale_choice == "Penjualan asli" else "Median nilai Min-Max · per produk"
    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        st.subheader("Profil pola mingguan")
        st.caption("Garis menunjukkan median per minggu; pita transparan menunjukkan rentang persentil 10–90. Ini bukan centroid DBA.")
        render_plotly(create_cluster_pattern_figure(values, labels, y_title, config), key="segment_pattern")
    with right, st.container(border=True):
        st.subheader("Produk dan kontribusi volume")
        render_plotly(create_cluster_contribution_figure(profiles, config), key="segment_contribution")

    cluster_values = sorted(np.unique(labels).tolist())
    selected_cluster = st.selectbox("Kelompok yang ditelusuri", cluster_values, format_func=format_cluster_label, key="selected_cluster")
    member_mask = labels == int(selected_cluster)
    member_indices = np.flatnonzero(member_mask)
    profile = profiles.loc[profiles["Cluster"] == int(selected_cluster)]
    if not profile.empty:
        p = profile.iloc[0]
        st.markdown(f"**{format_cluster_label(selected_cluster)}** · {int(p['Jumlah_Produk'])} produk · {p['Persen_Produk']:.1f}% produk · kontribusi volume {p['Porsi_Volume']:.1f}%.")
        st.caption(
            f"Median total volume: {p['Median_Total_Sales']:,.1f} · Median minggu nol: {p['Median_Minggu_Nol']:.0f} "
            f"· Produk medoid: {p['Produk_Medoid']} · Perubahan rerata paruh kedua vs pertama: {p['Perubahan_Paruh_2_vs_1_Persen']:.1f}%"
        )

    label_lookup = pd.DataFrame({"Product_Code": bundle["product_codes"], "Cluster": labels})
    member_details = bundle["product_stats"].merge(label_lookup, left_on="Produk", right_on="Product_Code", validate="one_to_one")
    member_details = member_details.loc[member_details["Cluster"] == int(selected_cluster)].drop(columns="Product_Code")
    search = st.text_input("Cari anggota dalam cluster", placeholder="Kode produk", key="segment_member_search").strip().upper()
    if search:
        member_details = member_details.loc[member_details["Produk"].str.upper().str.contains(search, regex=False)]
    st.caption(f"Menampilkan {len(member_details):,} anggota dari {int(member_mask.sum()):,} produk pada kelompok ini.")

    membership_columns = get_membership_columns(bundle["labels"], method)
    if membership_columns and len(member_details):
        membership = bundle["labels"][[config.ID_COLUMN, *membership_columns, f"{method}_Confidence"]].copy()
        member_details = member_details.merge(membership, left_on="Produk", right_on=config.ID_COLUMN, how="left").drop(columns=[config.ID_COLUMN])
        st.caption("Derajat keanggotaan fuzzy menunjukkan tingkat kedekatan relatif pada pusat cluster, bukan probabilitas akurasi atau peluang produk benar.")

    st.dataframe(member_details, hide_index=True, use_container_width=True, height=360)
    st.download_button(
        "Unduh anggota cluster", member_details.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"anggota_{method}_cluster_{selected_cluster}.csv", mime="text/csv", key="download_members",
    )


def render_comparison_page(bundle: dict[str, Any]) -> None:
    """Merender evaluasi, stabilitas, kesepakatan, dan hasil tuning enam metode."""
    render_page_heading("04", "Enam metode, satu gambaran perbandingan.", "Bandingkan pemisahan, kekompakan, kestabilan, dan cakupan produk secara bersamaan.")
    metrics = bundle["metrics"].copy()
    metrics["Nama Metode"] = metrics["Metode"].map(format_method_name)
    default_method = bundle["metadata"].get("best_method", config.DEFAULT_METHOD)
    winner = metrics.loc[metrics["Metode"] == default_method].iloc[0]

    with st.container(border=True):
        st.subheader("Pemimpin skor multi-indeks sementara")
        st.markdown(
            f"**{format_method_name(default_method)}** · skor {winner['Skor_Multi_Indeks']:.3f} · "
            f"Silhouette {winner['Silhouette_FastDTW']:.3f} · coverage {100*winner['Coverage']:.1f}%"
        )
        st.caption("Skor gabungan adalah heuristik pemeringkatan. Status fallback berarti belum ada model yang memenuhi seluruh guardrail untuk klaim segmentasi shape-only.")
    render_warning_panel(bundle["metadata"].get("selection_warnings", []), bundle["metadata"].get("selection_status", ""))

    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        st.subheader("Indeks evaluasi utama")
        st.caption("Silhouette dan CHI: lebih tinggi lebih baik. DBI medoid: lebih rendah lebih baik. Stabilitas ARI: mendekati 1 berarti lebih konsisten.")
        render_plotly(create_metrics_comparison_figure(metrics, config), key="compare_metrics")
        st.caption("Warna emas menandai metode tampilan awal (Fuzzy C-Means), bukan pemenang skor multi-indeks.")
    with right, st.container(border=True):
        st.subheader("Coverage dan jumlah cluster")
        render_plotly(create_coverage_figure(metrics, config), key="compare_coverage")

    visible = metrics[[
        "Nama Metode", "Jumlah_Cluster", "Coverage", "Jumlah_Noise", "Silhouette_FastDTW",
        "DBI_Medoid_FastDTW", "DBI_MinMax_Euclidean", "CHI_MinMax_Euclidean",
        "Eta2_Minggu_Nol", "Stabilitas_ARI", "Stabilitas_NMI", "Skor_Multi_Indeks",
    ]].copy()
    visible = visible.rename(columns={
        "Jumlah_Cluster": "Jumlah cluster", "Coverage": "Coverage", "Jumlah_Noise": "Noise",
        "Silhouette_FastDTW": "Silhouette FastDTW", "DBI_Medoid_FastDTW": "DBI medoid FastDTW",
        "DBI_MinMax_Euclidean": "DBI Min-Max Euclidean", "CHI_MinMax_Euclidean": "CHI Min-Max Euclidean",
        "Eta2_Minggu_Nol": "η² minggu nol", "Stabilitas_ARI": "Stabilitas ARI",
        "Stabilitas_NMI": "Stabilitas NMI", "Skor_Multi_Indeks": "Skor multi-indeks",
    })
    visible["Coverage"] *= 100
    st.subheader("Tabel perbandingan lengkap")
    st.dataframe(visible, hide_index=True, use_container_width=True)
    st.download_button("Unduh metrik perbandingan", metrics.to_csv(index=False).encode("utf-8-sig"), "metrics_comparison.csv", "text/csv")

    tab1, tab2, tab3 = st.tabs(["Kesepakatan metode", "Tuning parameter", "Diagnostik Silhouette & PCA"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            render_plotly(create_similarity_heatmap(bundle["ari"], "ARI", config), key="compare_ari")
            st.caption("ARI = 1 berarti dua partisi sama setelah memperhitungkan pertukaran nomor cluster.")
        with c2:
            render_plotly(create_similarity_heatmap(bundle["nmi"], "NMI", config), key="compare_nmi")
            st.caption("NMI mengukur kesamaan informasi antarpartisi. ARI/NMI bukan akurasi karena tidak ada label ground truth.")
    with tab2:
        tuning_k = bundle["tuning"].loc[bundle["tuning"]["Metode"] != "DBSCAN"].copy()
        tuning_k["k_Aktual"] = tuning_k["Jumlah_Cluster"].astype(int)
        render_plotly(create_tuning_k_figure(tuning_k, config), key="compare_tuning_k")
        dbscan_tuning = get_method_tuning(bundle, "DBSCAN")
        if not dbscan_tuning.empty:
            render_plotly(create_dbscan_tuning_figure(dbscan_tuning, config), key="compare_dbscan_tuning")
        st.caption("Kandidat tuning ditampilkan bersama coverage dan jumlah anggota; kandidat DBSCAN dengan banyak noise tidak setara dengan partisi seluruh produk.")
    with tab3:
        method = st.selectbox("Metode untuk diagnostik", bundle["methods"], index=bundle["methods"].index(default_method), format_func=format_method_name, key="silhouette_method")
        labels = tuple(get_method_labels(bundle, method).tolist())
        detail = calculate_silhouette_for_method(method, labels)
        detail = attach_product_codes_to_silhouette(detail, bundle["product_codes"])
        if detail.empty:
            st.info("Diagnostik Silhouette individual tidak dapat dihitung untuk partisi ini.")
        else:
            render_plotly(create_silhouette_distribution_figure(detail, config), key="compare_silhouette_dist")
            render_plotly(create_silhouette_box_figure(detail, config), key="compare_silhouette_box")
            st.caption(f"Detail dihitung untuk {len(detail):,} produk non-noise menggunakan matriks jarak FastDTW.")
        pca = bundle["pca_data"].copy()
        pca["Cluster"] = get_method_labels(bundle, method)
        render_plotly(create_pca_scatter_figure(pca, config), key="compare_pca")
        st.caption("PCA hanya proyeksi dua dimensi untuk membantu eksplorasi visual. PCA tidak digunakan saat membentuk cluster.")


def render_methodology_page(bundle: dict[str, Any]) -> None:
    """Merender alur metodologi, definisi metrik, batas interpretasi, dan unduhan artefak."""
    render_page_heading("05", "Dari data mentah menjadi pola.", "Telusuri keputusan preprocessing, pengukuran jarak, metode clustering, dan batas interpretasi.")
    metadata = bundle["metadata"]
    steps = [
        ("01", "Validasi data", "Memeriksa Product_Code unik, 52 kolom W0–W51, nilai numerik, missing value, dan nilai negatif."),
        ("02", "Min-Max per produk", "Menggunakan nilai penjualan asli sebagai input. Setiap produk diskalakan secara independen ke [0, 1] agar bentuk pola mingguan dapat dibandingkan."),
        ("03", "FastDTW radius 2", "Mengukur kemiripan dua deret mingguan dengan dynamic time warping aproksimatif. Satu matriks jarak dipakai bersama oleh semua metode."),
        ("04", "Enam metode clustering", "K-Means + DBA, K-Medoids, Agglomerative, DBSCAN, Fuzzy C-Means + DBA, dan Fuzzy C-Medoids."),
        ("05", "Evaluasi & interpretasi", "Membandingkan Silhouette, DBI medoid, DBI/CHI Euclidean, coverage, η² minggu nol, stabilitas, dan kesepakatan antar-metode."),
    ]
    with st.container(border=True):
        st.subheader("Alur analisis")
        for number, title, description in steps:
            st.markdown(
                f'<div class="step"><div class="step-n">{number}</div><div><b>{title}</b><p>{description}</p></div></div>',
                unsafe_allow_html=True,
            )

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("Kamus metrik")
        definitions = [
            ("Silhouette FastDTW", "Menggambarkan kekompakan dan pemisahan berdasarkan matriks jarak FastDTW; rentang −1 sampai 1, lebih tinggi lebih baik."),
            ("DBI medoid FastDTW", "Indeks adaptasi berbasis sebaran ke medoid dan jarak antarmedoid FastDTW; lebih rendah lebih baik, bukan DBI standar."),
            ("DBI & CHI Euclidean", "Indeks internal pada fitur Min-Max dengan geometri Euclidean; hanya diagnostik tambahan."),
            ("Coverage", "Proporsi produk yang memperoleh cluster. Produk berlabel −1 pada DBSCAN dihitung sebagai noise."),
            ("ARI & NMI", "Kesamaan partisi antar-metode atau antar-seed, bukan akurasi terhadap label benar."),
            ("η² minggu nol", "Proporsi variasi jumlah minggu nol yang terkait dengan perbedaan cluster. Nilai tinggi menunjukkan segmentasi berkaitan kuat dengan tingkat aktivitas nol."),
        ]
        for term, description in definitions:
            st.markdown(f"**{term}**")
            st.caption(description)
    with c2, st.container(border=True):
        st.subheader("Konfigurasi eksperimen")
        st.write(f"**Preprocessing:** {metadata['preprocessing']}")
        st.write(f"**Jarak:** {metadata['distance']['name']} · radius {metadata['distance']['radius']}")
        st.write(f"**Metode:** {', '.join(format_method_name(method) for method in metadata['methods'])}")
        st.write(f"**Status seleksi:** `{metadata['selection_status']}`")
        st.write(f"**Waktu artefak (UTC):** {metadata['created_at_utc']}")
        st.caption("Parameter terpilih untuk setiap metode:")
        parameter_rows = [{"Metode": format_method_name(method), **params} for method, params in metadata.get("best_parameters", {}).items()]
        st.dataframe(pd.DataFrame(parameter_rows), hide_index=True, use_container_width=True)

    st.warning(
        "**Batas interpretasi:** Min-Max per produk menekankan bentuk pola dan menghilangkan perbedaan skala antarproduk pada ruang clustering. "
        "Oleh karena itu kontribusi volume dan ukuran bisnis dibaca kembali dari penjualan asli. Hasil clustering bersifat deskriptif/eksploratif, "
        "bukan prediksi, rekomendasi operasional otomatis, atau bukti hubungan sebab-akibat."
    )
    render_warning_panel(metadata.get("selection_warnings", []), metadata.get("selection_status", ""))

    st.subheader("Unduh data dan artefak")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Data penjualan asli", bundle["sales"].to_csv(index=False).encode("utf-8-sig"), "sales.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button("Label enam metode", bundle["labels"].to_csv(index=False).encode("utf-8-sig"), "product_labels.csv", "text/csv", use_container_width=True)
    with d3:
        st.download_button("Metrik perbandingan", bundle["metrics"].to_csv(index=False).encode("utf-8-sig"), "metrics_comparison.csv", "text/csv", use_container_width=True)

    d4, d5, d6 = st.columns(3)
    pipeline_path = config.MODEL_DIR / "clustering_pipelines.joblib"
    prep_path = config.MODEL_DIR / "preprocessing_pipeline.joblib"
    with d4:
        if pipeline_path.exists():
            st.download_button("Pipeline enam model", pipeline_path.read_bytes(), "clustering_pipelines.joblib", "application/octet-stream", use_container_width=True)
    with d5:
        if prep_path.exists():
            st.download_button("Pipeline preprocessing", prep_path.read_bytes(), "preprocessing_pipeline.joblib", "application/octet-stream", use_container_width=True)
    with d6:
        st.download_button("Paket hasil lengkap (.zip)", build_results_zip(), "ruang_sales_artifacts.zip", "application/zip", use_container_width=True)

    with st.expander("Metadata eksperimen (JSON)"):
        st.json(metadata)
