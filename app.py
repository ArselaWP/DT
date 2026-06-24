import pandas as pd
import os

# Gunakan cara ini agar path-nya selalu benar di mana pun dijalankan
file_path = 'outputs/food_consumption_clustered_for_dashboard.csv'

try:
    df = pd.read_csv(file_path)
    # Tampilkan sesuatu di layar agar kita tahu data sudah terbaca
    st.write("Data berhasil dimuat!") 
except FileNotFoundError:
    st.error(f"File tidak ditemukan di path: {os.path.abspath(file_path)}")