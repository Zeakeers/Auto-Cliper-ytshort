import sys
import subprocess
try:
    import torch
except ImportError:
    print("❌ PyTorch belum terinstall. Jalankan: pip install -r requirements.txt")
    sys.exit(1)

def check_cuda():
    print("🔍 Memeriksa status CUDA untuk NVIDIA RTX 4060...")
    if torch.cuda.is_available():
        print(f"✅ CUDA Tersedia!")
        print(f"🚀 Device: {torch.cuda.get_device_name(0)}")
        print(f"📦 Versi CUDA PyTorch: {torch.version.cuda}")
    else:
        print("❌ CUDA tidak terdeteksi. Pastikan Driver dan PyTorch versi CUDA terpasang.")

def check_ffmpeg():
    print("\n🔍 Memeriksa FFmpeg & integrasi NVENC (Hardware Acceleration)...")
    try:
        result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True)
        if 'cuda' in result.stdout or 'nvenc' in result.stdout:
            print("✅ FFmpeg mendukung NVENC/CUDA!")
        else:
            print("⚠️ FFmpeg ditemukan, namun akselerasi NVENC mungkin tidak aktif.")
    except FileNotFoundError:
        print("❌ FFmpeg tidak ditemukan di sistem. Install dengan: sudo apt install ffmpeg")

if __name__ == "__main__":
    check_cuda()
    check_ffmpeg()
