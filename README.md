# Statistics Story AI - Backend Service

Layanan API Backend untuk **Statistics Story AI** — platform edukasi dan analisis statistik interaktif bertenaga AI dengan narasi mendalam, visualisasi data, dan studi kasus praktis.

---

## 🛠️ Tech Stack & Ekosistem

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+)
- **Server**: [Uvicorn](https://www.uvicorn.org/) (ASGI Server dengan hot-reload)
- **Database & ORM**: [PostgreSQL](https://www.postgresql.org/) & [SQLAlchemy 2.0](https://www.sqlalchemy.org/) dengan driver [psycopg v3](https://www.psycopg.org/)
- **Mesin Komputasi Statistik**: [Pandas](https://pandas.pydata.org/) & [SciPy Stats](https://scipy.org/)
- **Dual Engine AI Narrative**: 
  - [Google Gemini API](https://ai.google.dev/) (`gemini-2.5-flash`)
  - [Groq API](https://groq.com/) (`openai/gpt-oss-120b`) dengan sistem *automatic failover & cooldown 30 menit* saat kuota harian habis (HTTP 429).
- **Keamanan & Autentikasi**: [PyJWT](https://pyjwt.readthedocs.io/) & [Bcrypt](https://pypi.org/project/bcrypt/) (mendukung HTTP-Only Cookie & Bearer Token)

---

## 🚀 Fitur Utama

1. **Autentikasi Pengguna**:
   - Registrasi akun, login dengan enkripsi password Bcrypt.
   - Manajemen sesi aman berbasis JWT token (disimpan dalam HttpOnly cookie dan header Authorization).
   - Validasi email dan proteksi endpoint terautentikasi.

2. **Manajemen Dataset Fleksibel**:
   - Mendukung upload file **CSV**, **Excel (.xlsx, .xls)**, dan **JSON**.
   - Mendukung input manual baris/tabel langsung dari antarmuka frontend.
   - Sanitasi nilai non-numerik, nilai tak terhingga (`inf`), dan nilai hilang (`null/NaN`).

3. **Modul Komputasi Statistik**:
   - **Statistik Deskriptif**: Mean, Median, Modus (termasuk deteksi multimodal), Min, Max, Range, Varians, Standar Deviasi, Kuartil (Q1, Q2, Q3, IQR), Skewness, dan Kurtosis.
   - **Uji Beda (T-Test)**: Independent Two-Sample T-Test, Paired T-Test, dan One-Sample T-Test beserta degree of freedom ($df$) dan $p$-value.
   - **Analisis Korelasi**: Koefisien Korelasi Pearson dan Spearman rank correlation dengan matriks signifikansi.
   - **Regresi Linier**: Model regresi sederhana & berganda, intercept, koefisien variabel, $R^2$ (R-squared), Adjusted $R^2$, F-statistik, serta tabel ANOVA.

4. **Mesin Narasi AI (Storytelling)**:
   - Menghasilkan narasi edukatif 4 bagian yang mengalir:
     1. *Ringkasan Eksekutif & Karakteristik Data*
     2. *Dinamika Sebaran & Variabilitas*
     3. *Pola Kunci, Anomali & Insight Mendalam*
     4. *Kesimpulan Strategis & Rekomendasi Aksi*
   - Dilengkapi generator studi kasus kontekstual (*real-world business/educational use cases*).
   - Pembersihan otomatis artefak markdown table dan format ASCII agar tampilan narasi bersih dan enak dibaca.
   - Dukungan multi-bahasa: Bahasa Indonesia (`id`) & Bahasa Inggris (`en`).

---

## 📂 Struktur Direktori

```text
statistics_story_ai_backend/
├── .env.example          # Template konfigurasi environment variable
├── .gitignore            # Aturan pengabaian git (mengabaikan venv, cache, .env)
├── main.py               # Kode utama API: model, routes, kalkulasi, & AI prompt
├── requirements.txt      # Daftar pustaka & dependensi Python
└── README.md             # Dokumentasi teknis repositori backend
```

---

## ⚡ Panduan Instalasi & Menjalankan Lokal

### 1. Prasyarat
- Python 3.11 atau lebih baru
- PostgreSQL Server aktif (lokal atau cloud/Supabase/Neon)
- Google Gemini API Key dan/atau Groq API Key

### 2. Buat & Aktifkan Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment (`.env`)
Salin file `.env.example` menjadi `.env`:
```bash
cp .env.example .env
```
Buka file `.env` dan sesuaikan nilainya:
```env
DATABASE_URL=postgresql+psycopg://postgres:PASSWORD_POSTGRES_ANDA@localhost:5432/statistics_story_ai
SECRET_KEY=kunci_rahasia_jwt_acak_dan_panjang
FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
ENVIRONMENT=development

# Konfigurasi AI Storyteller
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
```

### 5. Jalankan Server API
Jalankan server menggunakan uvicorn:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
atau langsung:
```bash
python main.py
```

Server API akan aktif di `http://127.0.0.1:8000`.

---

## 📖 Dokumentasi API Interaktif

FastAPI menyediakan dokumentasi interaktif bawaan yang dapat diakses langsung melalui browser saat server berjalan:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📌 Ringkasan Endpoint Utama

| Metode | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | Mendaftarkan pengguna baru |
| `POST` | `/api/v1/auth/login` | Login dan mendapatkan token / cookie |
| `GET` | `/api/v1/auth/me` | Memeriksa informasi profil pengguna aktif |
| `POST` | `/api/v1/auth/logout` | Menghapus sesi autentikasi |
| `GET` | `/api/v1/analyses` | Mengambil seluruh daftar analisis pengguna |
| `POST` | `/api/v1/analyses` | Membuat sesi analisis baru |
| `GET` | `/api/v1/analyses/{id}` | Mengambil detail lengkap analisis tertentu |
| `POST` | `/api/v1/analyses/{id}/dataset` | Mengunggah dataset (CSV/XLSX/JSON/Manual) |
| `PUT` | `/api/v1/analyses/{id}/config` | Memperbarui konfigurasi metode statistik |
| `POST` | `/api/v1/analyses/{id}/run` | Menjalankan perhitungan statistik numerik |
| `POST` | `/api/v1/analyses/{id}/narrative` | Menghasilkan interpretasi & narasi cerita AI |
| `POST` | `/api/v1/analyses/{id}/save` | Menandai sesi analisis sebagai tersimpan/favorit |
| `DELETE` | `/api/v1/analyses/{id}` | Menghapus sesi analisis |
