import os
import cv2
import math
import subprocess
import ffmpeg
import numpy as np
import torch

# Workaround resmi untuk PyTorch 2.6+ (WeightsUnpickler error)
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        import pyannote.audio.core.task
        torch.serialization.add_safe_globals([
            torch.torch_version.TorchVersion,
            pyannote.audio.core.task.Specifications,
            pyannote.audio.core.task.Problem,
            pyannote.audio.core.task.Resolution,
        ])
except Exception as e:
    pass

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import mediapipe as mp

from pyannote.audio import Pipeline
from config.settings import HF_TOKEN, TEMP_DIR

def extract_frame_at_time(video_path, time_sec):
    """
    Mengekstrak 1 frame spesifik (numpy array) menggunakan cv2 untuk face detection.
    Langkah tak taktis agar VRAM tidak habis karena melacak seluruh 60 detik.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps):
        fps = 30
    frame_number = int(time_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def find_face_x_center(frame):
    """
    Menemukan wajah menggunakan MediaPipe Face Detection CPU/GPU
    Mengembalikan persentase posisi X (0.0 sampai 1.0)
    """
    try:
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(frame)
            if not results.detections:
                return None # Jangan kembalikan 0.5, biarkan fungsi pemanggil memakai fallback terakhirnya
            
            # Ambil wajah yang paling yakin / besar
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x_center = bbox.xmin + (bbox.width / 2)
            # Bounding box bisa diluar batas karena blur
            return max(0.0, min(1.0, x_center))
    except Exception as e:
        # Bypass darurat jika instalasi mediapipe bermasalah
        return None

def crop_and_concat(input_path, output_path, timeline_crop_data, vid_width, vid_height):
    """
    Menggunakan FFmpeg-python Filtergraph untuk Split + Crop + Concat secara Mulus dalam Sekali Jalan.
    Bebas dari Bug Audio Demuxer FFmpeg dan Mendukung Otomasi Hook di awal.
    """
    print(f"   -> [Editor] Filtergraph memproses {len(timeline_crop_data)} segmen asli...")
    
    # Resolusi target 9:16
    target_h = vid_height
    target_w = int(vid_height * 9 / 16)
    if target_w % 2 != 0: target_w += 1
    
    # --- SISTEM HOOK (Cuplikan Puncak) ---
    if len(timeline_crop_data) > 0:
        longest_seg = max(timeline_crop_data, key=lambda s: s['end'] - s['start'])
        hook_seg = {
            'start': longest_seg['start'],
            'end': min(longest_seg['end'], longest_seg['start'] + 4.0), # 4 detik
            'x': longest_seg['x'],
            'speaker': longest_seg['speaker']
        }
        # Rakit Timeline dengan Hook di depan
        final_timeline = [hook_seg] + timeline_crop_data
        print(f"   🔥 Hook (4 Detik) Ditambahkan Dari Menit {int(hook_seg['start']//60)}:{int(hook_seg['start']%60):02d}!")
    else:
        final_timeline = timeline_crop_data
        
    streams = []
    
    for idx, seg in enumerate(final_timeline):
        pixel_x_center = seg['x'] * vid_width
        crop_x = int(pixel_x_center - (target_w / 2))
        crop_x = max(0, min(crop_x, vid_width - target_w))
        crop_y = 0
        
        duration = max(0.1, seg['end'] - seg['start'])
        
        inp = ffmpeg.input(input_path, ss=seg['start'], t=duration)
        v = inp.video.filter('setpts', 'PTS-STARTPTS').filter('crop', target_w, target_h, crop_x, crop_y)
        a = inp.audio.filter('asetpts', 'PTS-STARTPTS')
        
        streams.extend([v, a])
        
    if not streams:
        print("   ❌ Kosong: Tidak ada segmen yang valid!")
        return
        
    print("   -> [Editor] Menyatukan seluruh segmen tanpa henti (Single-Pass Encoding)...")
    try:
        joined = ffmpeg.concat(*streams, v=1, a=1)
        out = ffmpeg.output(joined, output_path, vcodec='h264_nvenc', preset='fast', acodec='aac', audio_bitrate='128k')
        out.overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        print("   ❌ FFMPEG Filtergraph Error:", e.stderr.decode('utf8'))

def edit_video(input_path: str, mode: str):
    """
    Eksekutor Utama Editor.
    """
    output_path = input_path.replace(".mp4", "_edited.mp4")
    
    # Ambil resolusi video
    probe = ffmpeg.probe(input_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    vid_width = int(video_stream['width'])
    vid_height = int(video_stream['height'])
    vid_duration = float(video_stream['duration'])
    
    if mode.upper() == 'A':
        print(f"\n🎧 [Editor] Mode Podcast (Auto-Crop) dimulai pada klip: {input_path}")
        
        if not HF_TOKEN:
            print("   ⚠️ Peringatan: HF_TOKEN belum terisi di .env! Diarization mungkin gagal.")
            
        print("   -> 1. Memuat model Speaker Diarization AI ke VRAM...")
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
            # Pasang ke GPU jika tersedia
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception as e:
            print("   ❌ Gagal memuat model Pyannote:", e)
            print("   ⚠️ FALLBACK: Tetap memotong video murni di tengah (Center Crop 9:16)...")
            # Kita biarkan MediaPipe CPU mencari wajah paling dominan di tengah video
            sample_time = vid_duration / 2
            frame = extract_frame_at_time(input_path, sample_time)
            x_pos = 0.5
            if frame is not None:
                x_pos = find_face_x_center(frame)
                
            timeline_crop_data = [{'start': 0.0, 'end': vid_duration, 'x': x_pos}]
            crop_and_concat(input_path, output_path, timeline_crop_data, vid_width, vid_height)
            return output_path
            
        print("   -> 2. Menganalisa suara pembicara (Timeline Extraction)...")
        # Ini akan mengekstrak wav temporal lalu masuk ke pipeline
        audio_temp = input_path.replace(".mp4", ".wav")
        ffmpeg.input(input_path).output(audio_temp, acodec="pcm_s16le", ac=1, ar="16k").overwrite_output().run(quiet=True)
        
        diarization = pipeline(audio_temp)
        os.remove(audio_temp) # Hapus audio temp
        
        # Kosongkan VRAM (Free Memory Pyannote model)
        del pipeline
        torch.cuda.empty_cache()
        
        # Bangun Timeline Data
        timeline_crop_data = []
        last_end = 0.0
        
        # State tracker cerdas untuk Face Detection
        last_x_pos = 0.5
        speaker_x_memory = {}
        
        print("   -> 3. Melacak posisi wajah per-transisi Speaker (3-Point Sampling & Memory)...")
        # Iterasi dari hasil Diarization: (Turn, _, Speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            
            # Tes Sapuan 10-Point Waktu untuk mencegah Meleset
            x_pos = None
            for pct in [0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9]:
                sample_time = start + ((end - start) * pct)
                frame = extract_frame_at_time(input_path, sample_time)
                if frame is not None:
                    detected_x = find_face_x_center(frame)
                    if detected_x is not None:
                        x_pos = detected_x
                        break
                        
            # Caching System: Jangan langsung lompat ke tengah (0.5) jika wajah hilang
            if x_pos is not None:
                last_x_pos = x_pos
                speaker_x_memory[speaker] = x_pos
            else:
                if speaker in speaker_x_memory:
                    x_pos = speaker_x_memory[speaker]
                else:
                    x_pos = last_x_pos
                    
            # Kita rapikan segment agar seamless dan membatasi overlap ekstrem
            if len(timeline_crop_data) > 0:
                if start <= timeline_crop_data[-1]['start']:
                    start = timeline_crop_data[-1]['end'] + 0.1
                timeline_crop_data[-1]['end'] = max(timeline_crop_data[-1]['start'] + 0.1, start)
                
            timeline_crop_data.append({
                'start': start,
                'end': end,
                'x': x_pos,
                'speaker': speaker
            })
            last_end = end
            
        # Kadang awal video ada gap diem
        if timeline_crop_data and timeline_crop_data[0]['start'] > 0.0:
            timeline_crop_data[0]['start'] = 0.0
            
        # Kadang akhir video ada gap diem
        if timeline_crop_data and timeline_crop_data[-1]['end'] < vid_duration:
            timeline_crop_data[-1]['end'] = vid_duration
            
        # Apabila Pyannote sama sekali gagal dapatkan suara
        if len(timeline_crop_data) == 0:
            print("   ⚠️ Tidak ditemukan percakapan! Memotong layar tengah sebagai _fallback_.")
            timeline_crop_data.append({
                'start': 0.0,
                'end': vid_duration,
                'x': 0.5
            })
            
        # Render NVENC
        crop_and_concat(input_path, output_path, timeline_crop_data, vid_width, vid_height)
        
    elif mode.upper() == 'B':
        print(f"   -> [Editor] (Mode B) Menerapkan Split-Screen Gameplay pada {input_path}")
        # MOCK IMPLEMENTATION FOR MODE B
        output_path = input_path # Skip sementara jika B
        
    return output_path
