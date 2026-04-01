import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
# Direktori utama
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "data" / "temp"
OUTPUT_DIR = BASE_DIR / "data" / "output"
SFX_DIR = BASE_DIR / "data" / "sfx"

# Buat folder jika belum ada (berguna saat pertama kali setup)
for d in [TEMP_DIR, OUTPUT_DIR, SFX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Konfigurasi AI
WHISPER_MODEL = "small" # Stabil di RTX 4060 (karena ~2GB sisa VRAM sering terpakai oleh OS Linux & Chrome)
DEVICE = "cuda"

# API & Secrets
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") # Untuk Pyannote Diarization (Speaker Detection)
