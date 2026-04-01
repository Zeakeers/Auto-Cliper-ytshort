import os
# Paksa PyTorch 2.6+ untuk menerima pre-trained models dari Pyannote secara diam-diam
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import click
from core.fetcher import download_best_segments
from core.editor import edit_video
from core.caption import add_auto_captions
from core.uploader import upload_to_youtube, generate_youtube_title

@click.command()
@click.option('--url', required=True, prompt='YouTube Video URL', help='Link video YouTube asli (16:9)')
@click.option('--mode', type=click.Choice(['A', 'B'], case_sensitive=False), default='A', 
              help='Mode A: Podcast (Speaker-Centric), Mode B: Gaming (Split-Screen)')
@click.option('--clips', default=3, type=int, help='Jumlah klip Shorts yang ingin diekstrak')
@click.option('--time', default=None, help='Jadwal upload YouTube, cth: "20:00"')
def main(url, mode, clips, time):
    """
    AUTO-CLIPPER: AI Shorts Generator (Optimized for RTX 4060 / CUDA)
    """
    click.secho(f"🚀 Memulai Auto-Clipper untuk URL: {url} | Mode: {mode.upper()}", fg="green", bold=True)
    
    # 1. Fetcher (Analisis Heatmap & Download)
    click.secho("\n[1/4] Mengunduh bagian video paling banyak diputar...", fg="cyan")
    video_paths = download_best_segments(url, num_clips=clips)
    
    for idx, video_path in enumerate(video_paths):
        click.secho(f"\n🎥 Memproses Klip #{idx+1} ({video_path})", fg="yellow", bold=True)
        
        # 2. Editor (AI Crop & Scene arrangement)
        click.secho(f"[2/4] Eksekusi Editing (Mode {mode.upper()})...", fg="cyan")
        edited_path = edit_video(video_path, mode=mode)
        
        # 3. Captioning & VFX (Whisper NVENC)
        click.secho("[3/4] Menambahkan Animasi Subtitle & Efek (Whisper AI)...", fg="cyan")
        final_path, transcript = add_auto_captions(edited_path)
        
        # 4. Uploader
        click.secho(f"[4/4] Mempersiapkan Pengunggahan YouTube...", fg="cyan")
        ai_title = generate_youtube_title(transcript)
        
        msg = f"🚀 Video Siap! Rekomendasi Judul:\n✨ \"{ai_title}\"\nLanjutkan Upload ke YouTube Studio otomatis?"
        if time:
            msg = f"🕒 (Jadwal Tayang: {time}) " + msg
            
        if click.confirm(msg, default=True):
            upload_to_youtube(final_path, schedule=time, title=ai_title)
        else:
            click.secho(f"✅ [DISIMPAN LOKAL] Batal unggah. File Anda aman di: {final_path}", fg="yellow")
            
    click.secho("\n🎉 Semua proses selesai!", fg="green", bold=True)

if __name__ == '__main__':
    main()
