from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "food_consumption_clustered_for_dashboard.csv"

CLUSTER_SHORT = {
    1: "Menengah Stabil",
    2: "Rentan, Beban Pangan Tinggi",
    3: "Mapan, Beban Pangan Rendah",
}

CLUSTER_FULL = {
    1: "Konsumen Menengah Stabil",
    2: "Konsumen Ekonomi Rentan dengan Beban Pangan Tinggi",
    3: "Konsumen Mapan dengan Beban Pangan Rendah",
}

CLUSTER_COLORS = {
    "1 - Menengah Stabil": "#1d4ed8",
    "2 - Rentan, Beban Pangan Tinggi": "#dc2626",
    "3 - Mapan, Beban Pangan Rendah": "#0f766e",
}

FEATURES = [
    "income_million_idr",
    "expenditure_million_idr",
    "food_expenditure_pct",
    "social_class_score",
]

DISPLAY_NAMES = {
    "income_million_idr": "Pendapatan Rumah Tangga",
    "expenditure_million_idr": "Pengeluaran Rumah Tangga",
    "food_expenditure_pct": "Persentase Pengeluaran Makanan",
    "social_class_score": "Skor Kelas Sosial",
    "PC1": "PC1",
    "PC2": "PC2",
}


st.set_page_config(
    page_title="Dashboard Segmentasi Konsumen Makanan",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fb;
            --card: #ffffff;
            --text: #111827;
            --muted: #6b7280;
            --line: #e5e7eb;
        }

        html, body, [class*="css"] {
            background: var(--bg);
            color: var(--text);
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(255,255,255,0);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        #MainMenu, footer, [data-testid="stDeployButton"] {
            visibility: hidden;
        }

        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        h1 {
            letter-spacing: -0.03em;
            color: var(--text);
            font-size: 2.15rem;
            line-height: 1.08;
            margin-bottom: 0.25rem;
        }

        .subtle {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.5;
        }

        section[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.85rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.55rem;
        }

        [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        .chip {
            display: inline-block;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: #f8fafc;
            border: 1px solid var(--line);
            font-size: 0.82rem;
            color: var(--muted);
            margin: 0.18rem 0.35rem 0.18rem 0;
        }

        .panel-title {
            font-size: 1.1rem;
            font-weight: 650;
            color: var(--text);
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["cluster"] = data["cluster"].astype(int)
    data["cluster_short"] = data["cluster"].map(CLUSTER_SHORT)
    data["cluster_full"] = data["cluster"].map(CLUSTER_FULL)
    data["cluster_display"] = data["cluster"].astype(str) + " - " + data["cluster_short"]
    return data


def format_million(value: float) -> str:
    return f"{value:,.2f} juta"


def format_percent(value: float) -> str:
    return f"{value:,.1f}%"


def sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filter")
    st.sidebar.caption("Pilih irisan data yang ingin ditampilkan.")

    selected_clusters = st.sidebar.multiselect(
        "Cluster",
        options=sorted(data["cluster_display"].unique()),
        default=sorted(data["cluster_display"].unique()),
    )

    selected_cities = st.sidebar.multiselect(
        "Kota",
        options=sorted(data["(A1) CITY"].unique()),
        default=sorted(data["(A1) CITY"].unique()),
    )

    selected_provinces = st.sidebar.multiselect(
        "Provinsi",
        options=sorted(data["(A2) PROVINCE"].unique()),
        default=sorted(data["(A2) PROVINCE"].unique()),
    )

    selected_gender = st.sidebar.multiselect(
        "Gender",
        options=sorted(data["(B3) GENDER"].unique()),
        default=sorted(data["(B3) GENDER"].unique()),
    )

    return data[
        data["cluster_display"].isin(selected_clusters)
        & data["(A1) CITY"].isin(selected_cities)
        & data["(A2) PROVINCE"].isin(selected_provinces)
        & data["(B3) GENDER"].isin(selected_gender)
    ].copy()


def render_header(data: pd.DataFrame) -> None:
    st.title("Segmentasi Konsumen Makanan di Indonesia")
    st.markdown(
        """
        <div class="subtle">
        Dashboard ini menampilkan hasil K-Means clustering berdasarkan dimensi sosial ekonomi
        dengan reduksi PCA dua komponen. Struktur dibuat lebih ringkas supaya enak dipakai saat presentasi.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chips = [
        f"{len(data):,} responden",
        "K-Means k=3",
        "PCA 2 komponen",
        "Silhouette 0.697",
    ]
    st.markdown(
        "".join([f'<span class="chip">{chip}</span>' for chip in chips]),
        unsafe_allow_html=True,
    )


def render_metrics(data: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Jumlah Responden", f"{len(data):,}")
    with c2:
        st.metric("Rata-rata Pendapatan", format_million(data["income_million_idr"].mean()))
    with c3:
        st.metric("Rata-rata Pengeluaran", format_million(data["expenditure_million_idr"].mean()))
    with c4:
        st.metric("Rata-rata Beban Pangan", format_percent(data["food_expenditure_pct"].mean()))


def build_scatter(data: pd.DataFrame):
    fig = px.scatter(
        data,
        x="PC1",
        y="PC2",
        color="cluster_display",
        color_discrete_map=CLUSTER_COLORS,
        hover_data={
            "(A1) CITY": True,
            "(A2) PROVINCE": True,
            "(B3) GENDER": True,
            "income_million_idr": ":.2f",
            "expenditure_million_idr": ":.2f",
            "food_expenditure_pct": ":.1f",
            "PC1": ":.2f",
            "PC2": ":.2f",
        },
        labels={
            "cluster_display": "Cluster",
            "income_million_idr": "Pendapatan",
            "expenditure_million_idr": "Pengeluaran",
            "food_expenditure_pct": "% Pengeluaran Makanan",
        },
        height=520,
    )
    fig.update_traces(marker={"size": 8, "opacity": 0.82, "line": {"width": 0}})
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Cluster",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    return fig


def build_pie(data: pd.DataFrame):
    counts = (
        data.groupby(["cluster_display", "cluster_short"], as_index=False)
        .size()
        .rename(columns={"size": "jumlah_responden"})
    )
    fig = px.pie(
        counts,
        names="cluster_display",
        values="jumlah_responden",
        color="cluster_display",
        color_discrete_map=CLUSTER_COLORS,
        hole=3,
        height=800,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig, counts


def build_box(data: pd.DataFrame, selected_metric: str):
    fig = px.box(
        data,
        x="cluster_display",
        y=selected_metric,
        color="cluster_display",
        color_discrete_map=CLUSTER_COLORS,
        points="outliers",
        labels={
            "cluster_display": "Cluster",
            selected_metric: DISPLAY_NAMES.get(selected_metric, selected_metric),
        },
        height=420,
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf2f7")
    return fig


def build_bar(data: pd.DataFrame, dimension: str, title_label: str):
    counts = (
        data.groupby([dimension, "cluster_display"], as_index=False)
        .size()
        .rename(columns={"size": "jumlah_responden"})
    )
    fig = px.bar(
        counts,
        x=dimension,
        y="jumlah_responden",
        color="cluster_display",
        color_discrete_map=CLUSTER_COLORS,
        barmode="group",
        height=400,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="Cluster",
    )
    fig.update_xaxes(title=title_label, tickangle=-20, showgrid=False)
    fig.update_yaxes(title="Jumlah Responden", showgrid=True, gridcolor="#edf2f7")
    return fig


def render_profile_cards(data: pd.DataFrame) -> None:
    st.markdown("<div class='panel-title'>Profil Singkat Cluster</div>", unsafe_allow_html=True)
    summary = (
        data.groupby(["cluster", "cluster_short", "cluster_full"], as_index=False)
        .agg(
            jumlah_responden=("cluster", "size"),
            pendapatan_rata2=("income_million_idr", "mean"),
            pengeluaran_rata2=("expenditure_million_idr", "mean"),
            beban_pangan_rata2=("food_expenditure_pct", "mean"),
        )
        .round(2)
        .sort_values("cluster")
    )

    cols = st.columns(3)
    for _, row in summary.iterrows():
        idx = int(row["cluster"]) - 1
        with cols[idx]:
            st.metric(
                f"Cluster {int(row['cluster'])}: {row['cluster_short']}",
                f"{int(row['jumlah_responden']):,}",
                delta=f"Pendapatan {row['pendapatan_rata2']:.2f} | Beban pangan {row['beban_pangan_rata2']:.1f}%",
            )


def main() -> None:
    apply_styles()

    if not DATA_PATH.exists():
        st.error(f"File data tidak ditemukan: {DATA_PATH}")
        st.stop()

    data = load_data(DATA_PATH)
    filtered = sidebar_filters(data)

    if filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
        st.stop()

    render_header(filtered)
    render_metrics(filtered)

    st.markdown("---")
    render_profile_cards(filtered)

    overview_tab, profiling_tab, data_tab = st.tabs(["Overview", "Profil Cluster", "Data"])

    with overview_tab:
        left, right = st.columns([1.35, 1])

        with left:
            st.markdown("<div class='panel-title'>Peta Cluster PCA</div>", unsafe_allow_html=True)
            st.plotly_chart(build_scatter(filtered), use_container_width=True)

        with right:
            st.markdown("<div class='panel-title'>Komposisi Cluster</div>", unsafe_allow_html=True)
            pie_fig, cluster_counts = build_pie(filtered)
            st.plotly_chart(pie_fig, use_container_width=True)
            st.dataframe(
                cluster_counts.rename(
                    columns={
                        "cluster_display": "cluster",
                        "cluster_short": "label_singkat",
                    }
                )[["cluster", "label_singkat", "jumlah_responden"]],
                use_container_width=True,
                hide_index=True,
            )

    with profiling_tab:
        st.markdown("<div class='panel-title'>Perbandingan Indikator Sosial Ekonomi</div>", unsafe_allow_html=True)
        selected_metric = st.selectbox(
            "Pilih indikator",
            options=FEATURES,
            format_func=lambda value: DISPLAY_NAMES.get(value, value),
        )
        st.plotly_chart(build_box(filtered, selected_metric), use_container_width=True)

        st.markdown("<div class='panel-title'>Distribusi Demografi</div>", unsafe_allow_html=True)
        demo_left, demo_right = st.columns(2)

        with demo_left:
            st.plotly_chart(build_bar(filtered, "(A1) CITY", "Kota"), use_container_width=True)

        with demo_right:
            st.plotly_chart(build_bar(filtered, "(B3) GENDER", "Gender"), use_container_width=True)

        st.markdown("<div class='panel-title'>Profil Rata-rata Tiap Cluster</div>", unsafe_allow_html=True)
        profile = (
            filtered.groupby(["cluster", "cluster_short", "cluster_full"])[FEATURES]
            .mean()
            .round(2)
            .reset_index()
            .rename(columns=DISPLAY_NAMES)
        )
        st.dataframe(profile, use_container_width=True, hide_index=True)

    with data_tab:
        st.markdown("<div class='panel-title'>Data Detail</div>", unsafe_allow_html=True)
        detail_columns = [
            "(A1) CITY",
            "(A2) PROVINCE",
            "(B3) GENDER",
            "(B4) AGE",
            "(B10) SOCIAL CLAS",
            "income_million_idr",
            "expenditure_million_idr",
            "food_expenditure_pct",
            "PC1",
            "PC2",
            "cluster",
            "cluster_short",
        ]
        st.dataframe(filtered[detail_columns], use_container_width=True, hide_index=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Data Terfilter",
            data=csv,
            file_name="filtered_food_consumption_clusters.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
