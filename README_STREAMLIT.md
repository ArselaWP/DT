# Dashboard Segmentasi Konsumen Makanan

Dashboard ini membaca file:

`outputs/food_consumption_clustered_for_dashboard.csv`

## Cara Menjalankan di VS Code

1. Buka folder project ini di VS Code.
2. Buka terminal di VS Code.
3. Install package:

```bash
pip install -r requirements.txt
```

4. Jalankan dashboard:

```bash
streamlit run app.py
```

5. Buka URL yang muncul di terminal, biasanya:

```text
http://localhost:8501
```

## Isi Dashboard

- Ringkasan jumlah responden, pendapatan, pengeluaran, dan beban pangan.
- Visualisasi cluster berdasarkan PC1 dan PC2.
- Komposisi responden tiap cluster.
- Profil rata-rata tiap cluster.
- Perbandingan indikator sosial ekonomi.
- Distribusi kota dan gender.
- Tabel detail dan tombol download data terfilter.
