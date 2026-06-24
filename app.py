import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Segmentasi Konsumen Makanan",
    layout="wide",
)

# Path Data (menggunakan Path dari pathlib yang lebih aman)
DATA_PATH = Path(__file__).parent / "outputs" / "food_consumption_clustered_for_dashboard.csv"

# Label dan Konfigurasi
CLUSTER_LABELS = {
    1: "Konsumen Menengah Stabil",
    2: "Konsumen Ekonomi Rentan dengan Beban Pangan Tinggi",
    3: "Konsumen Mapan dengan Beban Pangan Rendah",
}

FEATURES = ["income_million_idr", "expenditure_million_idr", "food_expenditure_pct", "social_class_score"]
DISPLAY_NAMES = {
    "income_million_idr": "Pendapatan Rumah Tangga",
    "expenditure_million_idr": "Pengeluaran Rumah Tangga",
    "food_expenditure_pct": "Persentase Pengeluaran Makanan",
    "social_class_score": "Skor Kelas Sosial",
}

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["cluster"] = data["cluster"].astype(int)
    data["cluster_label"] = data["cluster"].map(CLUSTER_LABELS)
    data["cluster_display"] = data["cluster"].astype(str) + " - " + data["cluster_label"]
    return data

def main():
    if not DATA_PATH.exists():
        st.error(f"File data tidak ditemukan di: {DATA_PATH}")
        st.stop()

    data = load_data(DATA_PATH)
    
    # Sidebar Filters
    st.sidebar.title("Filter")
    selected_clusters = st.sidebar.multiselect("Cluster", options=sorted(data["cluster_display"].unique()), default=sorted(data["cluster_display"].unique()))
    selected_cities = st.sidebar.multiselect("Kota", options=sorted(data["(A1) CITY"].unique()), default=sorted(data["(A1) CITY"].unique()))
    
    filtered = data[data["cluster_display"].isin(selected_clusters) & data["(A1) CITY"].isin(selected_cities)]

    # Main Dashboard
    st.title("Analisis Segmentasi Konsumen Makanan di Indonesia")
    st.write("Data berhasil dimuat!")
    st.dataframe(filtered.head())
    # ... (tambahkan sisa komponen grafik kamu di sini)

if __name__ == "__main__":
    main()