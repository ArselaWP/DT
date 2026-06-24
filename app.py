import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Segmentasi Konsumen Makanan",
    page_icon="📊",
    layout="wide",
)

# Menentukan lokasi file data relatif terhadap app.py
# Folder 'outputs' harus berada di folder yang sama dengan app.py di GitHub
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "food_consumption_clustered_for_dashboard.csv"

# Konfigurasi Label dan Fitur
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

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan di: {path}")
    
    data = pd.read_csv(path)
    data["cluster"] = data["cluster"].astype(int)

    if "cluster_label" not in data.columns:
        data["cluster_label"] = data["cluster"].map(CLUSTER_LABELS)

    data["cluster_display"] = data["cluster"].astype(str) + " - " + data["cluster_label"]
    return data

def format_million(value: float) -> str: return f"{value:,.2f} juta"
def format_percent(value: float) -> str: return f"{value:,.1f}%"

def sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Filter")
    selected_clusters = st.sidebar.multiselect("Cluster", options=sorted(data["cluster_display"].unique()), default=sorted(data["cluster_display"].unique()))
    selected_cities = st.sidebar.multiselect("Kota", options=sorted(data["(A1) CITY"].unique()), default=sorted(data["(A1) CITY"].unique()))
    selected_provinces = st.sidebar.multiselect("Provinsi", options=sorted(data["(A2) PROVINCE"].unique()), default=sorted(data["(A2) PROVINCE"].unique()))
    selected_gender = st.sidebar.multiselect("Gender", options=sorted(data["(B3) GENDER"].unique()), default=sorted(data["(B3) GENDER"].unique()))

    return data[
        data["cluster_display"].isin(selected_clusters)
        & data["(A1) CITY"].isin(selected_cities)
        & data["(A2) PROVINCE"].isin(selected_provinces)
        & data["(B3) GENDER"].isin(selected_gender)
    ].copy()

def main() -> None:
    st.title("Analisis Segmentasi Konsumen Makanan di Indonesia")
    st.caption("K-Means Clustering berdasarkan dimensi sosial ekonomi dengan PCA dua komponen.")

    try:
        data = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        st.stop()

    filtered = sidebar_filters(data)

    if filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
        st.stop()

    # Metrics
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Jumlah Responden", f"{len(filtered):,}")
    kpi_2.metric("Rata-rata Pendapatan", format_million(filtered["income_million_idr"].mean()))
    kpi_3.metric("Rata-rata Pengeluaran", format_million(filtered["expenditure_million_idr"].mean()))
    kpi_4.metric("Rata-rata Beban Pangan", format_percent(filtered["food_expenditure_pct"].mean()))

    st.divider()

    # Grafik dan Data Detail (Lanjutkan sesuai kode lengkapmu di bawah sini...)
    # [Tambahkan sisa kode grafik scatter, pie, box, dan dataframe yang kamu inginkan di sini]
    
if __name__ == "__main__":
    main()