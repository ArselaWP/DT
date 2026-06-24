import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Segmentasi Konsumen Makanan", layout="wide")

# Path Data (Penting: Pastikan folder 'outputs' berada di level yang sama dengan 'app.py')
DATA_PATH = Path(__file__).parent / "outputs" / "food_consumption_clustered_for_dashboard.csv"

# Konfigurasi Awal
CLUSTER_LABELS = {
    1: "Konsumen Menengah Stabil",
    2: "Konsumen Ekonomi Rentan dengan Beban Pangan Tinggi",
    3: "Konsumen Mapan dengan Beban Pangan Rendah",
}

FEATURES = ["income_million_idr", "expenditure_million_idr", "food_expenditure_pct", "social_class_score"]
DISPLAY_NAMES = {
    "income_million_idr": "Pendapatan (Juta)",
    "expenditure_million_idr": "Pengeluaran (Juta)",
    "food_expenditure_pct": "Persentase Makanan",
    "social_class_score": "Skor Kelas Sosial",
}

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["cluster"] = data["cluster"].astype(int)
    data["cluster_label"] = data["cluster"].map(CLUSTER_LABELS)
    data["cluster_display"] = data["cluster"].astype(str) + " - " + data["cluster_label"]
    return data

def main():
    if not DATA_PATH.exists():
        st.error(f"File data tidak ditemukan di: {DATA_PATH}. Pastikan folder 'outputs' sudah di-upload ke GitHub.")
        st.stop()

    data = load_data(DATA_PATH)

    # --- Sidebar ---
    st.sidebar.title("Filter")
    selected_clusters = st.sidebar.multiselect("Cluster", options=sorted(data["cluster_display"].unique()), default=sorted(data["cluster_display"].unique()))
    selected_cities = st.sidebar.multiselect("Kota", options=sorted(data["(A1) CITY"].unique()), default=sorted(data["(A1) CITY"].unique()))
    
    filtered = data[data["cluster_display"].isin(selected_clusters) & data["(A1) CITY"].isin(selected_cities)]

    # --- Main Content ---
    st.title("Analisis Segmentasi Konsumen Makanan di Indonesia")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Responden", len(filtered))
    col2.metric("Avg Pendapatan", f"{filtered['income_million_idr'].mean():.2f} jt")
    col3.metric("Avg Pengeluaran", f"{filtered['expenditure_million_idr'].mean():.2f} jt")
    col4.metric("Avg Beban Pangan", f"{filtered['food_expenditure_pct'].mean():.1f}%")

    st.divider()

    # Layout Grafik
    left, right = st.columns([1.5, 1])
    
    with left:
        st.subheader("Peta Cluster PCA")
        fig_scatter = px.scatter(filtered, x="PC1", y="PC2", color="cluster_display")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with right:
        st.subheader("Komposisi Cluster")
        fig_pie = px.pie(filtered, names="cluster_display")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Data Detail")
    st.dataframe(filtered, use_container_width=True)

if __name__ == "__main__":
    main()