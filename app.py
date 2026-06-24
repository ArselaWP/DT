import pandas as pd
import os

# Gunakan path relatif
data_path = os.path.join(os.path.dirname(__file__), 'outputs', 'food_consumption_clustered_for_dashboard.csv')

# Cek apakah file ada sebelum dibaca
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.error(f"File tidak ditemukan di: {data_path}")
    # Opsi lain: tampilkan pesan error yang lebih informatif