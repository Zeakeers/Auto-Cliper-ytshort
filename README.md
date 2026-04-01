# Auto-Clipper AI (Linux Mint / RTX 4060 Optimized)

Scraper dan Editor Video Otomatis untuk platform Shorts, didukung akselerasi GPU lokal (CUDA) secara penuh. Dirancang untuk efisiensi di NVIDIA RTX 4060.

## ✨ Fitur Utama
1. **Pemetik Klip Ramai**: Ekstrak bagian video dengan *replay* terbanyak via `yt-dlp` heatmap.
2. **Mode A (Podcast)**: Pemotongan rasio 16:9 jadi 9:16 vertikal, melacak wajah pembicara menggunakan `mediapipe` dan `pyannote`.
3. **Mode B (Gaming)**: Layar dibagi rata (Gameplay di atas, Webcam di bawah).
4. **Auto-Caption (Whisper AI)**: Subtitle presisi menggunakan model Whisper `CUDA` (Hormozi style).
5. **YouTube Scheduler**: Sistem upload otomatis via YouTube Data API v3.

## 🛠️ Persyaratan Lingkungan (Linux Mint)
Pastikan Anda sudah menginstal **Driver NVIDIA Proprietary** (> 535) dan komponen dasar *developer*:
```bash
sudo apt update
sudo apt install build-essential python3-venv ffmpeg autoconf libtool
```

## 🚀 Panduan Setup Lokal

### 1. Kredensial & API (Wajib)
Sebelum menjalankan program, Anda membutuhkan beberapa API key dan kredensial:

**A. Hugging Face Access Token (Untuk Pyannote Diarization & Model AI):**
1. Buat akun di [Hugging Face](https://huggingface.co/).
2. Pergi ke **Settings > Access Tokens** dan buat token baru (tipe Read access atau Fine-grained).
3. Kunjungi halaman model [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) dan [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) secara manual di browser, lalu klik ikon/tombol untuk menyetujui syarat penggunaannya (Accept user conditions).
4. Buat file `.env` di dalam direktori `config/` (yaitu: `config/.env`) dan tambahkan token Anda:
   ```env
   HF_TOKEN=1234xxxx_token_hugging_face_lo_disini_xxxx
   YOUTUBE_API_KEY=AIzaSy_opsional_bisa_kosong_XXXXX
   ```

**B. YouTube OAuth Client (Untuk Auto-Upload ke Channel Anda):**
1. Buka [Google Cloud Console](https://console.cloud.google.com/), login dan buat project baru.
2. Buka menu **APIs & Services > Library**, cari lalu **Enable** atau aktifkan **YouTube Data API v3**.
3. Masuk ke **Credentials**, klik tombol **Create Credentials > OAuth client ID**.
4. Jika diminta mengatur *OAuth consent screen*, pilih **External** dan isi field wajib saja, tambahkan email Anda sebagai *Test users*.
5. Lanjut ke pembuatan OAuth client ID, pada *Application type*, pilih **Desktop App**, lalu klik **Create**.
6. Download JSON file-nya, ganti namanya (rename) menjadi `client_secret.json`. Taruh file tersebut ke dalam folder `config/` di project ini (path wajib: `config/client_secret.json`).

### 2. Buat Virtual Environment
Dari terminal Linux di dalam direktori proyek ini, jalankan:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi (Pip)
Pastikan environment virtual (`venv`) Anda sudah **aktif**. Lalu jalankan perintah pip berikut untuk mengunggah framework deep learning sesuai *requirements* yang sudah diefisiensikan:
```bash
# Otomatis menginstal PyTorch (CUDA), Tensor, Whisper, Pyannote, dll.
pip install -r requirements.txt
```

### 4. Cek Kelancaran Hardware
Pastikan library berhasil di-bind ke GPU RTX 4060 Anda:
```bash
python utils/gpu_check.py
```
> _Jika konsol memunculkan pesan centang hijau "CUDA Tersedia!" dan menyebut FFmpeg NVENC, maka instalasi sukses dan pipeline AI siap digunakan!_

## 💻 Penggunaan Command Line Interface (CLI)
Gunakan `main.py` menggunakan argumen command line berikut:
```bash
# Bantuan / Dokumentasi Argumen
python main.py --help

# Menjalankan Mode Podcast dengan 3 Klip, otomatis unggah jam 20:00
python main.py --url "https://youtube.com/watch?v=XXXX" --mode A --clips 3 --time "20:00"

# Menjalankan Mode Gaming Split Screen
python main.py --url "https://youtube.com/watch?v=XXXX" --mode B --clips 1
```

## 🗂 Struktur Direktori
- `config/`: Konfigurasi environment (buat file `.env` untuk API Key YouTube)
- `core/`: Modul utama (Fetcher, Editor, Caption, Audio, Uploader)
- `data/temp/`: File temporary (sedang diproses)
- `data/output/`: Hasil rendering akhir (siap upload)
- `data/sfx/`: Koleksi file-file suara *sound effect*
