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
1. **Buat Virtual Environment:**
   Dari terminal di direktori proyek ini, jalankan:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install Dependensi:**
   Dengan VRAM 8 GB dan environment CUDA 12, instal file _requirements_ yang telah diefisiensikan:
   ```bash
   pip install -r requirements.txt
   ```
3. **Cek Sistem Hardware:**
   Pastikan GPU RTX 4060 dan NVENC terbaca dengan sukses oleh library _deep-learning_ PyTorch:
   ```bash
   python utils/gpu_check.py
   ```
   > _Jika konsol memunculkan pesan centang hijau "CUDA Tersedia!" dan menyebut FFmpeg NVENC, maka instalasi sukses!_

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
