import streamlit as st
import pandas as pd
import os

# Dapatkan lokasi folder file app.py saat ini
base_path = os.path.dirname(__file__)
# Gabungkan dengan nama folder dan file
file_path = os.path.join(base_path, 'outputs', 'food_consumption_clustered_for_dashboard.csv')

try:
    df = pd.read_csv(file_path)
    # Lanjutkan kode dashboard kamu di sini...
    st.write("Data berhasil dimuat!")
except FileNotFoundError:
    st.error(f"File tidak ditemukan di path: {file_path}")
    
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
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
    page_icon="",
    layout="wide",
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["cluster"] = data["cluster"].astype(int)

    if "cluster_label" not in data.columns:
        data["cluster_label"] = data["cluster"].map(CLUSTER_LABELS)

    data["cluster_display"] = data["cluster"].astype(str) + " - " + data["cluster_label"]
    return data


def format_million(value: float) -> str:
    return f"{value:,.2f} juta"


def format_percent(value: float) -> str:
    return f"{value:,.1f}%"


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

    return data[
        data["cluster_display"].isin(selected_clusters)
        & data["(A1) CITY"].isin(selected_cities)
        & data["(A2) PROVINCE"].isin(selected_provinces)
        & data["(B3) GENDER"].isin(selected_gender)
    ].copy()


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def main() -> None:
    st.title("Analisis Segmentasi Konsumen Makanan di Indonesia")
    st.caption("K-Means Clustering berdasarkan dimensi sosial ekonomi dengan PCA dua komponen.")

    if not DATA_PATH.exists():
        st.error(f"File data tidak ditemukan: {DATA_PATH}")
        st.stop()

    data = load_data(DATA_PATH)
    filtered = sidebar_filters(data)

    if filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
        st.stop()

    total_respondents = len(filtered)
    avg_income = filtered["income_million_idr"].mean()
    avg_expenditure = filtered["expenditure_million_idr"].mean()
    avg_food_pct = filtered["food_expenditure_pct"].mean()

    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    with kpi_1:
        metric_card("Jumlah Responden", f"{total_respondents:,}")
    with kpi_2:
        metric_card("Rata-rata Pendapatan", format_million(avg_income))
    with kpi_3:
        metric_card("Rata-rata Pengeluaran", format_million(avg_expenditure))
    with kpi_4:
        metric_card("Rata-rata Beban Pangan", format_percent(avg_food_pct))

    st.divider()

    left, right = st.columns([1.45, 1])

    with left:
        st.subheader("Peta Cluster PCA")
        scatter = px.scatter(
            filtered,
            x="PC1",
            y="PC2",
            color="cluster_display",
            hover_data={
                "(A1) CITY": True,
                "(A2) PROVINCE": True,
                "(B3) GENDER": True,
                "income_million_idr": ":.2f",
                "expenditure_million_idr": ":.2f",
                "food_expenditure_pct": ":.1f",
                "cluster_display": True,
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
        scatter.update_traces(marker={"size": 8, "opacity": 0.78})
        scatter.update_layout(legend_title_text="Cluster", margin=dict(l=8, r=8, t=24, b=8))
        st.plotly_chart(scatter, use_container_width=True)

    with right:
        st.subheader("Komposisi Cluster")
        cluster_counts = (
            filtered.groupby(["cluster", "cluster_label"], as_index=False)
            .size()
            .rename(columns={"size": "jumlah_responden"})
        )
        cluster_counts["cluster_display"] = (
            cluster_counts["cluster"].astype(str) + " - " + cluster_counts["cluster_label"]
        )

        pie = px.pie(
            cluster_counts,
            names="cluster_display",
            values="jumlah_responden",
            hole=0.45,
            labels={"cluster_display": "Cluster", "jumlah_responden": "Jumlah Responden"},
            height=300,
        )
        pie.update_layout(showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
        st.plotly_chart(pie, use_container_width=True)

        st.dataframe(
            cluster_counts[["cluster", "cluster_label", "jumlah_responden"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Profil Rata-rata Tiap Cluster")
    profile = (
        filtered.groupby(["cluster", "cluster_label"])[FEATURES]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns=DISPLAY_NAMES)
    )
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("Perbandingan Indikator Sosial Ekonomi")
    selected_metric = st.selectbox(
        "Pilih indikator",
        options=FEATURES,
        format_func=lambda value: DISPLAY_NAMES.get(value, value),
    )

    box = px.box(
        filtered,
        x="cluster_display",
        y=selected_metric,
        color="cluster_display",
        points="outliers",
        labels={
            "cluster_display": "Cluster",
            selected_metric: DISPLAY_NAMES.get(selected_metric, selected_metric),
        },
        height=430,
    )
    box.update_layout(showlegend=False, margin=dict(l=8, r=8, t=24, b=8))
    st.plotly_chart(box, use_container_width=True)

    st.subheader("Distribusi Demografi")
    demo_left, demo_right = st.columns(2)

    with demo_left:
        city_counts = (
            filtered.groupby(["(A1) CITY", "cluster_display"], as_index=False)
            .size()
            .rename(columns={"size": "jumlah_responden"})
        )
        city_chart = px.bar(
            city_counts,
            x="(A1) CITY",
            y="jumlah_responden",
            color="cluster_display",
            barmode="group",
            labels={
                "(A1) CITY": "Kota",
                "jumlah_responden": "Jumlah Responden",
                "cluster_display": "Cluster",
            },
            height=420,
        )
        city_chart.update_layout(margin=dict(l=8, r=8, t=24, b=8))
        st.plotly_chart(city_chart, use_container_width=True)

    with demo_right:
        gender_counts = (
            filtered.groupby(["(B3) GENDER", "cluster_display"], as_index=False)
            .size()
            .rename(columns={"size": "jumlah_responden"})
        )
        gender_chart = px.bar(
            gender_counts,
            x="(B3) GENDER",
            y="jumlah_responden",
            color="cluster_display",
            barmode="group",
            labels={
                "(B3) GENDER": "Gender",
                "jumlah_responden": "Jumlah Responden",
                "cluster_display": "Cluster",
            },
            height=420,
        )
        gender_chart.update_layout(margin=dict(l=8, r=8, t=24, b=8))
        st.plotly_chart(gender_chart, use_container_width=True)

    st.subheader("Data Detail")
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
        "cluster_label",
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
