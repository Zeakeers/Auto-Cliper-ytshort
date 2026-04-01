import os
import subprocess
import whisper
import torch
import pysubs2
from config.settings import WHISPER_MODEL, TEMP_DIR

def add_auto_captions(video_path: str):
    """
    Ekstrak audio dari video untuk kemudian diproses model Whisper AI Large-v3.
    Hasil word-timestamp ditransfer ke file `.ass` format ala Hormozi.
    File final dirender dengan *Hardware Acceleration* FFmpeg NVENC.
    """
    print(f"\n📝 [Captioner] Memulai proses Auto-Caption AI (Model: {WHISPER_MODEL}) pada GPU...")
    
    # 1. Ekstrak Audio Sementara
    audio_path = video_path.replace(".mp4", "_temp_audio.wav")
    print("   -> Mengekstrak gelombang audio...")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Transkripsi AI
    print(f"   -> Mentranskrip dengan Whisper (Perataan Kata / Word-Level)...")
    try:
        model = whisper.load_model(WHISPER_MODEL, device="cuda")
    except Exception as e:
        print(f"   ❌ Model CUDA Gagal Dimuat: {str(e)}\n   -> Fallback CPU...")
        model = whisper.load_model(WHISPER_MODEL, device="cpu")
        
    # Panggil fungsi inferensi
    result = model.transcribe(audio_path, word_timestamps=True, verbose=False)
    
    # Sangat penting: Hapus model whisper dari Memori RTX 4060 Anda agar tidak Crash!
    del model
    torch.cuda.empty_cache()
    
    # 3. Formulasi Subtitle (Hormozi / Alex-style)
    print("   -> Membuat Animasi Kata Tebal (Styling pysubs2)...")
    subs = pysubs2.SSAFile()
    
    # Membuat parameter visual teks kuning tebal
    style = pysubs2.SSAStyle(
        fontname="Arial Black",     # Pastikan OS memiliki font Arial Black / Helvetica
        fontsize=24,                
        primarycolor=pysubs2.Color(255, 255, 255, 0),  # Warna Teks: Putih Solid (Alpha 0 = Tidak Transparan)
        outlinecolor=pysubs2.Color(0, 0, 0, 0),        # Warna Border: Hitam Solid
        outline=5,                                     # Border ditebalkan menjadi 5
        backcolor=pysubs2.Color(0, 0, 0, 128),         # Bayangan
        shadow=3,
        bold=-1,                                       # Di-Bold Penuh
        alignment=2,                
        marginv=120                 
    )
    subs.styles["Hormozi"] = style
    
    # Pasang per-event berdasarkan hasil Whisper AI
    for segment in result.get('segments', []):
        for word_info in segment.get('words', []):
            start_ms = int(word_info['start'] * 1000)
            end_ms = int(word_info['end'] * 1000)
            text = word_info['word'].strip().upper() # Gaya bold besar kapital
            
            # Buat segmen baru untuk *setiap* kata agar munculnya lompat-lompat 1 layar
            event = pysubs2.SSAEvent(
                start=start_ms, 
                end=end_ms, 
                text=text, 
                style="Hormozi"
            )
            subs.append(event)
            
    ass_path = os.path.join(TEMP_DIR, "dynamic_subs.ass")
    subs.save(ass_path)
    
    # 4. Burn FFmpeg NVENC (Direct-Burn Hardware Accelerated)
    print("   -> Merender subtitle animasi ke dalam video menggunakan NVENC (Extra Fast)...")
    final_output_path = video_path.replace("_edited.mp4", "_final.mp4").replace("temp", "output")
    
    # Hapus file output _final jika sudah ada tertimpa sebelumnnya
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
        
    # Eksekusi FFmpeg burning NVENC secara manual 
    # Karena parameter VF (Video Filter) subtitles cukup sensitif apabila di handle wrapper python
    # Mengonversi path khusus Ass Subtitles untuk ffmpeg CLI (escaping colon jika diperlukan di OS tertentu)
    ass_filter_path = ass_path.replace("\\", "/").replace(":", "\\:")
    
    ffmpeg_burn = [
        "ffmpeg", "-y", "-hwaccel", "cuda",
        "-i", video_path,
        "-vf", f"subtitles='{ass_filter_path}'",
        "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-b:v", "5M", # Render NVENC Kualitas Tinggi
        "-c:a", "copy", # Audio tidak perlu di render lagi, irit resos CPU
        final_output_path
    ]
    
    result_render = subprocess.run(ffmpeg_burn, capture_output=True, text=True)
    
    if result_render.returncode != 0:
         print("   ❌ Error merender subtitle:", result_render.stderr)
    
    # Cleanup memory hard-disk sementara
    os.remove(audio_path)
    os.remove(ass_path)
    
    print(f"   ✅ [Selesai!] Video Siap Upload: {final_output_path}")
    return final_output_path, result.get('text', '')
