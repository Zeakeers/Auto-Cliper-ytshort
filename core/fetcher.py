import os
import random
import yt_dlp
from config.settings import TEMP_DIR

def download_best_segments(url: str, num_clips: int = 3, clip_duration: int = 60):
    """
    Menggunakan yt-dlp untuk ekstrak data heatmap dan mengunduh segmen paling viral secara parsial.
    """
    print(f"🎬 [Fetcher] Menarik metadata untuk: {url}")
    
    ydl_opts_meta = {
        'skip_download': True,
        'quiet': True,
        'extract_flat': False
    }

    with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
        info = ydl.extract_info(url, download=False)
    
    heatmap = info.get('heatmap', [])
    video_duration = info.get('duration', 0)
    
    timestamps = []
    
    if heatmap:
        print("🔥 [Fetcher] Grafik Heatmap (Replay) ditemukan! Mencari puncak ketenaran...")
        # Sort heatmap berdasarkan nilai "value" (seberapa sering di-replay)
        heatmap.sort(key=lambda x: x['value'], reverse=True)
        
        # Ekstrak N peaks tanpa tumpang tindih
        for point in heatmap:
            start_t = point['start_time']
            end_t = point['start_time'] + clip_duration
            
            # Pastikan tidak tumpang tindih dengan clip sebelumnya (minimal geser 30 detik)
            overlap = any(
                (start_t >= t[0]-30 and start_t <= t[1]+30)
                for t in timestamps
            )
            
            if not overlap and end_t <= video_duration:
                timestamps.append((start_t, end_t))
            
            if len(timestamps) == num_clips:
                break
    
    if len(timestamps) < num_clips:
        print("⚠️ [Fetcher] Heatmap tidak cukup atau tidak aktif. Menggunakan strategi fallback acak...")
        while len(timestamps) < num_clips:
            if video_duration <= clip_duration:
                start_t = 0
                end_t = video_duration
                timestamps.append((start_t, end_t))
                break
            else:
                start_t = random.randint(0, int(video_duration - clip_duration))
                end_t = start_t + clip_duration
                
                # Cek overlap
                overlap = any(
                    (start_t >= t[0]-10 and start_t <= t[1]+10) 
                    for t in timestamps
                )
                if not overlap:
                    timestamps.append((start_t, end_t))
    
    # Urutkan secara kronologis (dari awal video ke akhir)
    timestamps.sort(key=lambda x: x[0])
    
    downloaded_files = []
    
    print("📥 [Fetcher] Mulai mengunduh segmen parsial (Bebas Kuota, Lebih Cepat!)...")
    for idx, (start, end) in enumerate(timestamps):
        output_name = os.path.join(TEMP_DIR, f"clip_{idx+1}.mp4")
        
        # Hapus file jika sudah ada sebelumnya di cache temp
        if os.path.exists(output_name):
            os.remove(output_name)
            
        print(f"   -> Mengunduh Klip #{idx+1} ({int(start)}s s/d {int(end)}s)")
        
        # Config yt_dlp untuk parsial download
        # Membutuhkan ffmpeg untuk pemotongan langsung tanpa mendownload stream utuh
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': output_name,
            'download_ranges': yt_dlp.utils.download_range_func(None, [(start, end)]),
            'force_keyframes_at_cuts': True,
            'quiet': True,
            'no_warnings': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            downloaded_files.append(output_name)
            print(f"   ✅ File Klip tersimpan: {output_name}")
        except Exception as e:
            print(f"   ❌ Gagal mengunduh klip #{idx+1}: {e}")
            
    return downloaded_files
