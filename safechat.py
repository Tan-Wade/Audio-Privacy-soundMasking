# SafeChat-style masking & authorized recovery (minimal reproducible demo)
# - If you have a WAV file (mono, 16 kHz or 16–48 kHz), put its path in `INPUT_WAV`.
# - If not, this script will synthesize a voice-like signal for the demo.
#
# What this code shows:
# 1) Generate masking sound (white or voice-like) and mix at target SNR.
# 2) Save clean, mask, and mixed files.
# 3) "Authorized" side removes the mask using LMS adaptive filtering (requires known mask signal).
# 4) Reports SNRs and improvements.
#
# Files are written to /mnt/data and links will be shown after execution.

import os
import numpy as np
from pathlib import Path

# Try to import soundfile; if unavailable, fall back to scipy.io.wavfile
try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    from scipy.io import wavfile
    HAVE_SF = False

SR = 16000  # sample rate
DURATION = 6.0  # seconds for synthetic demo
OUTPUT_DIR = Path("/mnt/data/safechat_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======== Audio I/O helpers ========
def save_wav(path, data, sr=SR):
    data = np.asarray(data, dtype=np.float32)
    data = np.clip(data, -1.0, 1.0)
    if HAVE_SF:
        sf.write(str(path), data, sr, subtype="PCM_16")
    else:
        # scipy expects int16
        from scipy.io.wavfile import write as wav_write
        wav_write(str(path), sr, (data * 32767).astype(np.int16))

def load_wav(path, force_mono=True, target_sr=SR):
    if HAVE_SF:
        data, sr = sf.read(str(path), always_2d=False)
    else:
        sr, data = wavfile.read(str(path))
        # normalize to float32
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32) / np.max(1e-9 + np.abs(data))
    # mono
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    # resample if needed (simple linear to avoid extra deps)
    if sr != target_sr:
        # naive resample
        import math
        ratio = target_sr / sr
        new_len = int(math.ceil(len(data) * ratio))
        x_old = np.linspace(0, 1, num=len(data), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = target_sr
    return data.astype(np.float32), sr

# ======== Signal utilities ========
def rms(x):
    return np.sqrt(np.mean(np.square(x)) + 1e-12)

def snr_db(clean, noise):
    return 20.0 * np.log10((rms(clean) + 1e-12) / (rms(noise) + 1e-12))

def mix_at_snr(clean, noise, target_snr_db):
    # scale noise to reach target SNR wrt clean
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms < 1e-12:
        return clean.copy(), noise.copy(), clean.copy()
    desired_noise_rms = clean_rms / (10.0 ** (target_snr_db / 20.0))
    scale = desired_noise_rms / noise_rms
    noise_scaled = noise * scale
    mixed = clean + noise_scaled
    return mixed, noise_scaled, mixed

# ======== Mask generators ========
def gen_white_noise(n):
    return np.random.randn(n).astype(np.float32)

def butter_bandpass(low_hz, high_hz, sr, order=4):
    # bilinear transform design (manual to avoid scipy.signal dependency)
    # To keep it lightweight, we approximate with simple IIR by cascading first-order filters.
    # For better quality, replace with scipy.signal.butter + filtfilt if available.
    # Here, we’ll use a simple FIR windowed-sinc bandpass to keep dependencies minimal.
    numtaps = 513  # odd, moderately narrow
    nyq = sr / 2.0
    f1 = low_hz / nyq
    f2 = high_hz / nyq
    # windowed-sinc bandpass kernel
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    # avoid div by zero at center
    def sinc(x):
        return np.sinc(x / np.pi)
    h = (2 * f2 * sinc(2 * np.pi * f2 * n) - 2 * f1 * sinc(2 * np.pi * f1 * n))
    # Hann window
    w = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(numtaps)) / (numtaps - 1)))
    h = h * w
    # normalize
    h = h / np.sum(h)
    return h.astype(np.float32)

def apply_fir(x, h):
    # linear convolution, 'same' length
    y = np.convolve(x, h, mode='same')
    return y.astype(np.float32)

def gen_voice_like_noise(n, sr=SR):
    # Start with white noise -> band-limit to speech band -> random amplitude modulation
    w = gen_white_noise(n)
    h = butter_bandpass(200.0, 3800.0, sr, order=4)
    banded = apply_fir(w, h)
    # Random smooth envelope (simulate syllabic energy)
    env_len = max(1, n // 400)  # coarse envelope resolution
    env = np.abs(np.random.randn(env_len)).astype(np.float32)
    # low-pass the envelope with FIR
    h_env = butter_bandpass(0.1, 3.0, sr=sr, order=4)  # very low band -> acts like LP for envelope up to ~3Hz
    # Resample env to signal length
    env_up = np.interp(np.linspace(0, env_len - 1, num=n), np.arange(env_len), env)
    # Smooth with short FIR
    h_short = np.ones(51, dtype=np.float32) / 51.0
    env_smooth = apply_fir(env_up, h_short)
    env_smooth = env_smooth / (np.max(np.abs(env_smooth)) + 1e-9)
    voice_like = banded * (0.3 + 0.7 * env_smooth)  # keep floor > 0
    return voice_like.astype(np.float32)

# ======== Adaptive filter (LMS) for authorized removal ========
def lms_cancel(mixed, mask_ref, mu=0.01, order=64):
    """
    Cancel mask_ref from mixed using LMS adaptive filtering.
    mixed: observed signal = clean + mask*H
    mask_ref: reference input (known mask at source)
    Returns: recovered (estimate of clean), error history, filter taps
    """
    n = len(mixed)
    w = np.zeros(order, dtype=np.float32)
    xbuf = np.zeros(order, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    e = np.zeros(n, dtype=np.float32)

    # Simple time-alignment: assume small delay; we can brute-force a small lag search
    # For MVP, assume zero-lag; in real use, do GCC-PHAT or chirp sync.
    for i in range(n):
        xbuf[1:] = xbuf[:-1]
        xbuf[0] = mask_ref[i] if i < len(mask_ref) else 0.0
        y[i] = np.dot(w, xbuf)  # estimated mask after unknown channel
        e[i] = mixed[i] - y[i]  # error tries to be clean speech
        # LMS update
        w += 2 * mu * e[i] * xbuf
    recovered = e
    return recovered, e, w

# ======== Demo pipeline ========
INPUT_WAV = ""  # put your file path here if you have one, e.g., "/mnt/data/your_clean.wav"
TARGET_SNR_DB = 0.0  # SNR of clean vs. mask in the MIX (lower -> stronger mask)

# Load or synthesize clean speech
if INPUT_WAV and os.path.exists(INPUT_WAV):
    clean, _ = load_wav(INPUT_WAV, target_sr=SR)
else:
    # synthesize a "voice-like" clean signal: sum of AM vowels-like tones + short pauses
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False).astype(np.float32)
    f0 = 140.0  # base pitch
    # three "formants" (rough vowel-ish tones), with slight time-varying AM
    f1, f2, f3 = 700.0, 1100.0, 2500.0
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * t)
    clean = (0.6*np.sin(2*np.pi*(f0)*t) +
             0.3*np.sin(2*np.pi*(f1)*t) * am +
             0.2*np.sin(2*np.pi*(f2)*t) * (1-am) +
             0.1*np.sin(2*np.pi*(f3)*t)).astype(np.float32)
    # simulate pauses by multiplying with a slow gate
    gate = (np.sin(2*np.pi*0.25*t) > -0.2).astype(np.float32) * 0.8
    clean *= gate
    clean = clean / (np.max(np.abs(clean)) + 1e-9)

# Generate a masking signal (choose one)
np.random.seed(7)
n = len(clean)
mask_white = gen_white_noise(n)
mask_voice_like = gen_voice_like_noise(n, sr=SR)
# pick which mask to use for demo
mask = mask_voice_like

# Mix at target SNR
mixed, mask_scaled, _ = mix_at_snr(clean, mask, TARGET_SNR_DB)

# Authorized-side removal (knows the mask waveform before channel)
recovered, _, taps = lms_cancel(mixed, mask_scaled, mu=0.01, order=128)

# Compute SNRs
snr_in = snr_db(clean, mask_scaled)
snr_after = snr_db(clean, recovered - clean)  # error between recovered and clean is residual noise

# Save audio files
clean_path = OUTPUT_DIR / "01_clean.wav"
mask_path = OUTPUT_DIR / "02_mask.wav"
mixed_path = OUTPUT_DIR / "03_mixed.wav"
recovered_path = OUTPUT_DIR / "04_recovered.wav"

save_wav(clean_path, clean, SR)
save_wav(mask_path, mask_scaled / (np.max(np.abs(mask_scaled)) + 1e-9) * 0.8, SR)
save_wav(mixed_path, mixed / (np.max(np.abs(mixed)) + 1e-9) * 0.8, SR)
# normalize recovered for listening
rec_norm = recovered / (np.max(np.abs(recovered)) + 1e-9) * 0.95
save_wav(recovered_path, rec_norm, SR)

print("=== SafeChat-style Masking Demo ===")
print(f"Sample Rate: {SR} Hz, Length: {len(clean)/SR:.2f} s")
print(f"Target SNR in mix: {TARGET_SNR_DB:.1f} dB")
print(f"Measured SNR(clean vs. mask) in MIX: {snr_in:.2f} dB")
print(f"SNR after authorized removal (clean vs. residual): {snr_after:.2f} dB")
print("\nOutput files:")
print(f"- Clean: {clean_path}")
print(f"- Mask (scaled): {mask_path}")
print(f"- Mixed: {mixed_path}")
print(f"- Recovered (authorized): {recovered_path}")