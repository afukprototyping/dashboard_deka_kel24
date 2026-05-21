# Always Healthy Hospital - Customer Experience Dashboard

Dashboard interaktif untuk menganalisis performa Customer Experience menggunakan Streamlit dengan integrasi API dari Google Gemini.

## Fitur Utama
* **Pelacakan KPI**: Memantau Net Promoter Score (NPS), Customer Satisfaction Index (CSI), Customer Loyalty Index (CLI), dan Customer Effort Score (CES).
* **Analisis Performa**: Peringkat 15 cabang dan evaluasi 10 titik interaksi/touchpoints utama pasien.
* **Integrasi AI**: Chatbot analitik menggunakan model Gemini 2.5 Flash untuk wawasan operasional secara real-time.

## Panduan Instalasi (Lokal)
1. Kloning repositori:
   ```bash
   git clone [https://github.com/username-kamu/nama-repo.git](https://github.com/username-kamu/nama-repo.git)
   cd nama-repo
   ```
   
2. Instal dependensi:
```Bash
pip install -r requirements.txt
```

3. Konfigurasi API Key (buat direktori dan file `.streamlit/secrets.toml`):
   ```toml
   GEMINI_API_KEY = "masukkan-api-key-gemini-di-sini"
   ```
   
4. Jalankan aplikasi:
```Bash
streamlit run app.py
```

## Deployment (Streamlit Cloud)
Saat men-deploy di Streamlit Community Cloud, tambahkan variabel berikut pada menu **Advanced Settings > Secrets**:
```toml
GEMINI_API_KEY = "masukkan-api-key-gemini-di-sini"
```
Pastikan file requirements.txt sudah disertakan di dalam repositori agar Streamlit dapat menginstal dependensi yang dibutuhkan.

## Struktur File
```
├── app.py                 # Skrip utama aplikasi Streamlit
├── data.csv               # Dataset Customer Experience (3.600 responden)
├── requirements.txt       # Daftar dependensi Python
└── README.md              # Dokumentasi proyek
```
