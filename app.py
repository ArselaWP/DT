import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# KONFIGURASI
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "food_consumption_clustered_for_dashboard.csv"

CLUSTER_LABELS = {
    1: "Konsumen Menengah Stabil",
    2: "Konsumen Ekonomi Rentan dengan Beban Pangan Tinggi",
    3: "Konsumen Mapan dengan Beban Pangan Rendah",
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
)

# =====================================================
# FUNCTIONS
# =====================================================

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"File data tidak ditemukan:\n{path}")
        st.stop()

    data = pd.read_csv(path)

    data["cluster"] = data["cluster"].astype(int)

    if "cluster_label" not in data.columns:
        data["cluster_label"] = data["cluster"].map(CLUSTER_LABELS)

    data["cluster_display"] = (
        data["cluster"].astype(str) + " - " + data["cluster_label"]
    )

    return data


def format_million(value: float) -> str:
    return f"{value:,.2f} juta"


def format_percent(value: float) -> str:
    return f"{value:,.1f}%"


def metric_card(label: str, value: str):
    st.metric(label=label, value=value)


def sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Filter")

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

    filtered = data[
        data["cluster_display"].isin(selected_clusters)
        & data["(A1) CITY"].isin(selected_cities)
        & data["(A2) PROVINCE"].isin(selected_provinces)
        & data["(B3) GENDER"].isin(selected_gender)
    ]

    return filtered.copy()


# =====================================================
# MAIN APP
# =====================================================

def main():

    st.title("Analisis Segmentasi Konsumen Makanan di Indonesia")

    st.caption(
        "K-Means Clustering berdasarkan dimensi sosial ekonomi dengan PCA dua komponen."
    )

    data = load_data(DATA_PATH)

    filtered = sidebar_filters(data)

    if filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
        st.stop()

    # =====================================================
    # KPI
    # =====================================================

    total_respondents = len(filtered)
    avg_income = filtered["income_million_idr"].mean()
    avg_expenditure = filtered["expenditure_million_idr"].mean()
    avg_food_pct = filtered["food_expenditure_pct"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Jumlah Responden", f"{total_respondents:,}")

    with col2:
        metric_card(
            "Rata-rata Pendapatan",
            format_million(avg_income),
        )

    with col3:
        metric_card(
            "Rata-rata Pengeluaran",
            format_million(avg_expenditure),
        )

    with col4:
        metric_card(
            "Rata-rata Beban Pangan",
            format_percent(avg_food_pct),
        )

    st.divider()

    # =====================================================
    # PCA SCATTER & PIE CHART
    # =====================================================

    left, right = st.columns([1.5, 1])

    with left:

        st.subheader("Peta Cluster PCA")

        scatter = px.scatter(
            filtered,
            x="PC1",
            y="PC2",
            color="cluster_display",
            height=520,
            hover_data=[
                "(A1) CITY",
                "(A2) PROVINCE",
                "(B3) GENDER",
            ],
        )

        scatter.update_traces(
            marker=dict(
                size=8,
                opacity=0.8
            )
        )

        st.plotly_chart(
            scatter,
            use_container_width=True
        )

    with right:

        st.subheader("Komposisi Cluster")

        cluster_counts = (
            filtered.groupby(
                ["cluster", "cluster_label", "cluster_display"],
                as_index=False
            )
            .size()
            .rename(columns={"size": "jumlah_responden"})
        )

        pie = px.pie(
            cluster_counts,
            names="cluster_display",
            values="jumlah_responden",
            hole=0.45,
            height=320,
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

        st.dataframe(
            cluster_counts[
                [
                    "cluster",
                    "cluster_label",
                    "jumlah_responden",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # =====================================================
    # PROFIL CLUSTER
    # =====================================================

    st.subheader("Profil Rata-rata Tiap Cluster")

    profile = (
        filtered.groupby(
            ["cluster", "cluster_label"]
        )[FEATURES]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns=DISPLAY_NAMES)
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # BOXPLOT
    # =====================================================

    st.subheader("Perbandingan Indikator Sosial Ekonomi")

    selected_metric = st.selectbox(
        "Pilih indikator",
        FEATURES,
        format_func=lambda x: DISPLAY_NAMES[x],
    )

    box = px.box(
        filtered,
        x="cluster_display",
        y=selected_metric,
        color="cluster_display",
        points="outliers",
        height=430,
    )

    box.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        box,
        use_container_width=True
    )

    # =====================================================
    # DEMOGRAFI
    # =====================================================

    st.subheader("Distribusi Demografi")

    demo_left, demo_right = st.columns(2)

    with demo_left:

        city_counts = (
            filtered.groupby(
                ["(A1) CITY", "cluster_display"],
                as_index=False
            )
            .size()
            .rename(columns={"size": "jumlah_responden"})
        )

        city_chart = px.bar(
            city_counts,
            x="(A1) CITY",
            y="jumlah_responden",
            color="cluster_display",
            barmode="group",
            height=420,
        )

        st.plotly_chart(
            city_chart,
            use_container_width=True,
        )

    with demo_right:

        gender_counts = (
            filtered.groupby(
                ["(B3) GENDER", "cluster_display"],
                as_index=False
            )
            .size()
            .rename(columns={"size": "jumlah_responden"})
        )

        gender_chart = px.bar(
            gender_counts,
            x="(B3) GENDER",
            y="jumlah_responden",
            color="cluster_display",
            barmode="group",
            height=420,
        )

        st.plotly_chart(
            gender_chart,
            use_container_width=True,
        )

    # =====================================================
    # DETAIL DATA
    # =====================================================

    st.subheader("Data Detail")

    detail_columns = [
        "(A1) CITY",
        "(A2) PROVINCE",
        "(B3) GENDER",
        "(B4) AGE",
        "income_million_idr",
        "expenditure_million_idr",
        "food_expenditure_pct",
        "PC1",
        "PC2",
        "cluster",
        "cluster_label",
    ]

    existing_columns = [
        col
        for col in detail_columns
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[existing_columns],
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Data Terfilter",
        data=csv,
        file_name="filtered_food_consumption_clusters.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()