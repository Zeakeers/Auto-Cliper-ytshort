import os
import json
import datetime
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Mendefinisikan Izin yang akan kita minta ke YouTube (Upload Only)
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Lokasi File Config
BASE_DIR = Path(__file__).resolve().parent.parent
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, "config", "client_secret.json")
TOKEN_FILE = os.path.join(BASE_DIR, "config", "token.json")

def get_authenticated_service():
    """
    Mengontrol Login Akun Google: Jika Token kadaluarsa atau tidak ada,
    buka peramban untuk otentikasi.
    """
    creds = None
    # Token.json menyimpan akses otomatis (refresh token)
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
    # Jika tidak ada token (Pertama Kali) atau Kadaluarsa
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("   -> Memperbarui token akses YouTube...")
            creds.refresh(Request())
        else:
            print("   -> Membuka peramban untuk Login Akun Google perdana...")
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise FileNotFoundError(f"File Konfigurasi '{CLIENT_SECRETS_FILE}' tidak ditemukan! Anda harus meletakkan surat kuasa client_secret.json dari Google Cloud Console di folder config/.")
                
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Simpan kredensial permanen di token.json
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    return build('youtube', 'v3', credentials=creds)

def parse_schedule_time(time_str: str) -> str:
    """
    Mengonversi input jam 'HH:MM' pengguna ke format rilis kalender Google (ISO 8601 di UTC).
    Sistem diasumsikan berjalan pada host berzona waktu lokal (Misal: WIB / UTC+7).
    """
    now = datetime.datetime.now()
    try:
        hour, minute = map(int, time_str.split(":"))
        # Set jadwal rilis pada hari yang sama dengan jam input
        dt_local = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Jika jadwal ini sudah terlewat di hari ini, jadwalkan untuk BESOK
        if dt_local < now:
            dt_local += datetime.timedelta(days=1)
            
        # Hitung offset Timezone (Local -> UTC) karena YouTube maunya string berformat 'T'
        utc_dt = dt_local.astimezone(datetime.timezone.utc)
        return utc_dt.isoformat()
    except Exception as e:
        print(f"   ❌ Format waktu '{time_str}' tidak valid, menggunakan mode Upload Biasa.")
        return None

import requests
from config.settings import HF_TOKEN

def generate_youtube_title(transcript: str) -> str:
    """Menggunakan AI (Hugging Face API) untuk merangkum transcript menjadi judul viral"""
    default_title = f'AI Auto-Clipper Highlight #{datetime.datetime.now().strftime("%d%m%H")}'
    
    if not HF_TOKEN or len(transcript.strip()) < 10:
        return default_title
        
    print("   -> 🧠 Membaca Transkrip & Menciptakan Judul Viral AI...")
    # Menggunakan model Phi-3 yang saat ini tersedia secara gratis 24/7 di HuggingFace
    API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Memotong transcript jika terlalu panjang
    safe_transcript = " ".join(transcript.split()[:400])
    
    # Phi-3 menggunakan gaya prompt chat (user/assistant)
    prompt = f"<|user|>\nBuatkan HANYA 1 baris Judul YouTube Shorts clickbait (maksimal 60 karakter, 2 emoji, bahasa Indonesia) berdasarkan cerita video ini:\n\"{safe_transcript}\"<|end|>\n<|assistant|>\n"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 30,
            "temperature": 0.8,
            "return_full_text": False
        }
    }
    
    try:
        # Tingkatkan timeout karena model mungkin sedang "bangun tidur" (Cold Start)
        response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        response.raise_for_status()
        data = response.json()
        raw_title = data[0]['generated_text'].strip()
        
        # Bersihkan format nakal dari AI
        clean_title = raw_title.replace('"', '').replace("'", "").split("\n")[0].strip()
        if not clean_title:
            return default_title
            
        print(f"   ✨ Rekomendasi Judul AI: {clean_title}")
        return clean_title[:95]
        
    except Exception as e:
        print(f"   ⚠️ Gagal memuat AI Judul: {e}. Memakai Default.")
        return default_title

def upload_to_youtube(video_path: str, schedule: str = None, title: str = None):
    """
    Memublikasikan file MP4 menjadi Shorts dengan judul Default / AI
    dan secara opsional menetapkan waktu rilis publik (Upload Scheduling).
    """
    print(f"\n📤 [Uploader] Menginisiasi YouTube API untuk mengunggah: {video_path}")
    
    youtube = get_authenticated_service()
    
    if not title:
         title = f'AI Auto-Clipper Highlight #{datetime.datetime.now().strftime("%d%m%H")}'
         
    # Menyiapkan Struktur Data (Metadata YouTube)
    # Ini bisa Anda edit manual ke depannya jika butuh!
    body = {
        'snippet': {
            'title': title, # Menggunakan judul AI jika ada
            'description': 'Klip canggih hasil ekstraksi otomatis menggunakan Teknologi Pyannote dan NVENC Pipeline.\n#shorts #podcast #aiedit',
            'tags': ['shorts', 'podcast', 'highlights', 'auto-clipper', 'coding']
        },
        'status': {
            'privacyStatus': 'private',     # Selalu jadikan privat terlebih dahulu sebelum tayang (aturan wajib API Publikasi)
            'selfDeclaredMadeForKids': False
        }
    }
    
    if schedule:
        schedule_iso = parse_schedule_time(schedule)
        if schedule_iso:
            body['status']['publishAt'] = schedule_iso
            print(f"   -> Mengatur Jam Rilis Publik: {schedule_iso} UTC (Sesuai dengan: {schedule} Local)")

    # Menyatukan Metadata dengan Media File asli
    print("   -> Mengunggah File Video. (Mohon tunggu beberapa detik)...")
    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=MediaFileUpload(video_path, chunksize=-1, resumable=True)
    )
    
    response = insert_request.execute()
    
    print(f"   ✅ [SUKSES!] Video Terunggah ke YouTube Studio.")
    print(f"   🔗 Cek Video Anda di tautan ini: https://www.youtube.com/watch?v={response['id']}")
