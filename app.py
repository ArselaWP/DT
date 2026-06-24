import streamlit as st
import pandas as pd
import os

st.title("Analisis Segmentasi Konsumen")

# Gunakan cara ini agar path-nya selalu benar di mana pun dijalankan
file_path = 'outputs/food_consumption_clustered_for_dashboard.csv'

try:
    df = pd.read_csv(file_path)
    st.success("Data berhasil dimuat!") 
    
    # Tambahkan ini untuk memastikan data memang terbaca dengan benar
    st.write("Pratinjau Data:")
    st.dataframe(df.head()) 
    
except FileNotFoundError:
    st.error(f"File tidak ditemukan di path: {os.path.abspath(file_path)}")
    st.write("Pastikan folder 'outputs' sudah ada di GitHub dan berisi file tersebut.")