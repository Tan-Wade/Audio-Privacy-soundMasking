#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Privacy Protection System using Sound Masking Techniques

Core Functions:
1. Apply masking noise to clean speech (similar to "encryption")
2. Generate mixed signal (simulate what eavesdroppers would record)
3. Authorized parties use known parameters for reverse recovery
4. Unauthorized parties can only hear mixed signals

To run this project, open the terminal and run the following command:
python audio_privacy_system.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import argparse
import warnings
import json
import secrets
import time
import uuid
warnings.filterwarnings('ignore')

# Import audio quality metrics module
try:
    from audio_metrics import AudioMetrics
    HAVE_METRICS = True
except ImportError:
    HAVE_METRICS = False

# Import hybrid encryption module
try:
    from encryption_module import HybridEncryption
    HAVE_ENCRYPTION = True
except ImportError:
    HAVE_ENCRYPTION = False
    print("Warning: Encryption module not available. Install cryptography: pip install cryptography")

# Audio processing dependencies
try:
    import soundfile as sf
    HAVE_SF = True
except ImportError:
    try:
        from scipy.io import wavfile
        HAVE_SF = False
    except ImportError:
        print("Warning: Need to install soundfile or scipy for audio processing")
        HAVE_SF = None

class AudioPrivacySystem:
    """Audio Privacy Protection System Main Class"""
    
    def __init__(self, sample_rate: int = 16000, target_snr_db: float = 0.0, production_mode: bool = False, 
                 enable_encryption: bool = False):
        """
        Initialize Audio Privacy Protection System
        
        Args:
            sample_rate: Sample rate, default 16kHz (suitable for speech)
            target_snr_db: Target SNR, default 0dB (strong masking effect)
            production_mode: Production mode, if True, will not save mask audio files
            enable_encryption: Enable hybrid encryption for mask parameters
        """
        self.sr = sample_rate
        self.target_snr_db = target_snr_db
        self.production_mode = production_mode
        self.enable_encryption = enable_encryption
        
        # Setup input/output directories
        self.dataset_dir = Path("./dataset")
        self.input_dir = self.dataset_dir / "input"
        self.output_dir = self.dataset_dir / "output"
        
        # Create organized output subdirectories
        self.audio_dir = self.output_dir / "audio"
        self.clean_audio_dir = self.audio_dir / "clean"
        self.mixed_audio_dir = self.audio_dir / "mixed"
        self.recovered_audio_dir = self.audio_dir / "recovered"
        self.mask_audio_dir = self.audio_dir / "masks"
        
        self.encryption_dir = self.output_dir / "encryption"
        self.keys_dir = self.encryption_dir / "keys"
        self.params_dir = self.encryption_dir / "params"
        
        # Create all directories
        self.dataset_dir.mkdir(exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create audio subdirectories
        self.audio_dir.mkdir(exist_ok=True)
        self.clean_audio_dir.mkdir(exist_ok=True)
        self.mixed_audio_dir.mkdir(exist_ok=True)
        self.recovered_audio_dir.mkdir(exist_ok=True)
        self.mask_audio_dir.mkdir(exist_ok=True)
        
        # Create encryption subdirectories
        self.encryption_dir.mkdir(exist_ok=True)
        self.keys_dir.mkdir(exist_ok=True)
        self.params_dir.mkdir(exist_ok=True)
        
        # Initialize audio quality evaluator
        self.metrics_calc = AudioMetrics(sample_rate) if HAVE_METRICS else None
        
        # Initialize encryption module
        self.crypto = HybridEncryption() if (HAVE_ENCRYPTION and enable_encryption) else None
        
        # Voice feature parameters
        self.voice_params = {
            'speech_band': (200, 4000), # Speech frequency band
            'syllable_rate': (2, 5),    # Syllable rate (Hz)
        }
        
    def load_audio(self, file_path: str, force_mono: bool = True) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        # Check if soundfile is available
        try:
            import soundfile as sf
            have_soundfile = True
        except ImportError:
            have_soundfile = False
        
        # For non-WAV formats, must use soundfile
        if file_ext in ['.m4a', '.mp3', '.flac', '.ogg']:
            if not have_soundfile:
                raise RuntimeError(f"File format {file_ext} requires soundfile library, install: pip install soundfile")
            try:
                data, sr = sf.read(file_path, always_2d=False)
            except Exception as e:
                raise RuntimeError(f"Cannot read audio file {file_path}: {e}")
        elif have_soundfile:
            # Try soundfile
            try:
                data, sr = sf.read(file_path, always_2d=False)
            except Exception:
                # If soundfile fails, try scipy (WAV only)
                if file_ext == '.wav':
                    sr, data = wavfile.read(file_path)
                    # Normalize to float32
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    else:
                        data = data.astype(np.float32) / np.max(1e-9 + np.abs(data))
                else:
                    raise RuntimeError(f"Cannot read audio file {file_path}, ensure soundfile library is installed")
        else:
            # Only scipy available, WAV format only
            if file_ext != '.wav':
                raise RuntimeError(f"File format {file_ext} requires soundfile library, install: pip install soundfile")
            
            sr, data = wavfile.read(file_path)
            # Normalize to float32
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            else:
                data = data.astype(np.float32) / np.max(1e-9 + np.abs(data))
        
        # Convert to mono
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        
        # Resample if needed
        if sr != self.sr:
            data = self._resample(data, sr, self.sr)
            sr = self.sr
            
        return data.astype(np.float32), sr
    
    def save_audio(self, file_path: str, data: np.ndarray, sr: int = None):
        """Save audio file"""
        if sr is None:
            sr = self.sr
            
        data = np.asarray(data, dtype=np.float32)
        data = np.clip(data, -1.0, 1.0)
        
        # Try soundfile first
        try:
            import soundfile as sf
            sf.write(file_path, data, sr, subtype="PCM_16")
        except ImportError:
            # Fallback to scipy
            from scipy.io.wavfile import write as wav_write
            wav_write(file_path, sr, (data * 32767).astype(np.int16))
    
    def _resample(self, data: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
        """Simple resampling (linear interpolation)"""
        if old_sr == new_sr:
            return data
            
        ratio = new_sr / old_sr
        new_len = int(len(data) * ratio)
        x_old = np.linspace(0, 1, num=len(data), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        return np.interp(x_new, x_old, data).astype(np.float32)
    
    def generate_voice_like_mask(self, length: int, sr: int = None, mask_type: str = "multi_tone", seed: int = None) -> np.ndarray:
        """
        Generate masking noise
        
        Args:
            length: Signal length
            sr: Sample rate
            mask_type: Type of masking noise ("voice_like", "multi_tone")
            seed: Random seed for reproducibility (if None, use random seed)
        """
        if sr is None:
            sr = self.sr
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        if mask_type == "voice_like":
            return self._generate_voice_like_noise(length, sr)
        elif mask_type == "multi_tone":
            return self._generate_multi_tone_mask(length, sr)
        else:
            return self._generate_multi_tone_mask(length, sr)
    
    def _generate_voice_like_noise(self, length: int, sr: int) -> np.ndarray:
        """Original voice-like masking noise"""
        # 1. Generate white noise
        white_noise = np.random.randn(length).astype(np.float32)
        
        # 2. Bandpass filter to speech frequency band
        filtered_noise = self._bandpass_filter(white_noise, sr)
        
        # 3. Add syllable modulation (simulate speech energy changes)
        syllable_modulation = self._generate_syllable_modulation(length, sr)
        
        # 4. Combine to generate voice-like noise
        voice_like = filtered_noise * syllable_modulation
        
        # 5. Normalize and boost volume
        voice_like = voice_like / (np.max(np.abs(voice_like)) + 1e-9)
        voice_like *= 2.0  # STRONG boost volume for aggressive masking effect
        
        return voice_like.astype(np.float32)
    
    
    def _generate_multi_tone_mask(self, length: int, sr: int) -> np.ndarray:
        """Multi-tone masking with speech-like characteristics"""
        t = np.linspace(0, length / sr, length, endpoint=False)
        mask = np.zeros(length, dtype=np.float32)
        
        # Generate multiple tones in speech frequency range
        speech_tones = [300, 500, 800, 1200, 1800, 2500, 3200]  # Common speech frequencies
        
        for i, freq in enumerate(speech_tones):
            # Add slight frequency modulation
            fm_freq = 0.5 + i * 0.2
            freq_mod = freq * (1 + 0.1 * np.sin(2 * np.pi * fm_freq * t))
            
            # Generate tone with amplitude modulation
            amplitude = 0.8 + 0.4 * np.sin(2 * np.pi * (0.3 + i * 0.1) * t)
            phase = 2 * np.pi * freq_mod * t + np.random.uniform(0, 2*np.pi)
            
            tone = amplitude * np.sin(phase)
            mask += tone * 0.15
        
        # Add some filtered noise for texture
        noise = np.random.randn(length).astype(np.float32)
        filtered_noise = self._bandpass_filter(noise, sr)
        mask += filtered_noise * 0.3
        
        # Normalize and boost volume
        mask = mask / (np.max(np.abs(mask)) + 1e-9)
        mask *= 2.0  # STRONG boost volume for aggressive masking effect
        
        return mask.astype(np.float32)
    
    
    def _bandpass_filter_custom(self, signal: np.ndarray, sr: int, low_freq: float, high_freq: float) -> np.ndarray:
        """Custom bandpass filter with specific frequency range"""
        numtaps = 257
        nyq = sr / 2.0
        f1 = low_freq / nyq
        f2 = high_freq / nyq
        
        # Windowed sinc bandpass filter
        n = np.arange(numtaps) - (numtaps - 1) / 2.0
        
        def sinc(x):
            return np.sinc(x / np.pi)
            
        h = (2 * f2 * sinc(2 * np.pi * f2 * n) - 2 * f1 * sinc(2 * np.pi * f1 * n))
        
        # Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(numtaps) / (numtaps - 1)))
        h = h * window
        
        # Normalize
        h = h / np.sum(h)
        
        # Apply filter
        filtered = np.convolve(signal, h, mode='same')
        return filtered.astype(np.float32)
    
    def _bandpass_filter(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Simple bandpass filter (windowed sinc)"""
        low_freq, high_freq = self.voice_params['speech_band']
        
        # Design FIR filter
        numtaps = 513
        nyq = sr / 2.0
        f1 = low_freq / nyq
        f2 = high_freq / nyq
        
        # Windowed sinc bandpass filter
        n = np.arange(numtaps) - (numtaps - 1) / 2.0
        
        def sinc(x):
            return np.sinc(x / np.pi)
            
        h = (2 * f2 * sinc(2 * np.pi * f2 * n) - 2 * f1 * sinc(2 * np.pi * f1 * n))
        
        # Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(numtaps) / (numtaps - 1)))
        h = h * window
        
        # Normalize
        h = h / np.sum(h)
        
        # Apply filter
        filtered = np.convolve(signal, h, mode='same')
        return filtered.astype(np.float32)
    
    def _generate_syllable_modulation(self, length: int, sr: int) -> np.ndarray:
        """Generate syllable modulation signal"""
        # Random syllable rate
        syllable_rate = np.random.uniform(*self.voice_params['syllable_rate'])
        
        # Generate random envelope
        env_length = max(1, length // 400)  # Coarse envelope
        envelope = np.abs(np.random.randn(env_length)).astype(np.float32)
        
        # Upsample to signal length
        t_env = np.linspace(0, env_length - 1, num=length)
        envelope_up = np.interp(t_env, np.arange(env_length), envelope)
        
        # Smooth envelope (simulate syllable boundaries)
        smooth_kernel = np.ones(51, dtype=np.float32) / 51.0
        envelope_smooth = np.convolve(envelope_up, smooth_kernel, mode='same')
        envelope_smooth = envelope_smooth / (np.max(np.abs(envelope_smooth)) + 1e-9)
        
        # Add syllable modulation
        t = np.linspace(0, length / sr, length, endpoint=False)
        syllable_mod = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t))
        
        return envelope_smooth * syllable_mod
    
    def mix_signals(self, clean: np.ndarray, mask: np.ndarray, target_snr_db: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix clean signal and masking signal at specified SNR
        
        Args:
            clean: Clean speech signal
            mask: Masking noise signal
            target_snr_db: Target SNR (dB)
            
        Returns:
            mixed: Mixed signal
            scaled_mask: Scaled masking signal
        """
        if target_snr_db is None:
            target_snr_db = self.target_snr_db
            
        # Calculate RMS
        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-12)
        mask_rms = np.sqrt(np.mean(mask ** 2) + 1e-12)
        
        if mask_rms < 1e-12:
            return clean.copy(), mask.copy()
        
        # Calculate required masking signal amplitude
        # Apply VERY strong masking by reducing effective SNR significantly
        effective_snr = target_snr_db - 8.0  # Reduce SNR by 8dB for VERY strong masking
        desired_mask_rms = clean_rms / (10.0 ** (effective_snr / 20.0))
        scale_factor = desired_mask_rms / mask_rms
        
        # Scale masking signal
        scaled_mask = mask * scale_factor
        
        # Mix signals
        mixed = clean + scaled_mask
        
        return mixed, scaled_mask
    
    def lms_recovery(self, mixed: np.ndarray, mask_ref: np.ndarray, 
                     mu: float = 0.01, filter_order: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        LMS adaptive filter for authorized recovery
        
        Args:
            mixed: Observed signal (clean + mask)
            mask_ref: Reference masking signal (known to authorized party)
            mu: Learning rate
            filter_order: Filter order
            
        Returns:
            recovered: Recovered clean signal
            filter_taps: Filter coefficients
        """
        n = len(mixed)
        w = np.zeros(filter_order, dtype=np.float32)
        x_buffer = np.zeros(filter_order, dtype=np.float32)
        recovered = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            # Update input buffer
            x_buffer[1:] = x_buffer[:-1]
            x_buffer[0] = mask_ref[i] if i < len(mask_ref) else 0.0
            
            # Calculate filter output
            y = np.dot(w, x_buffer)
            
            # Calculate error signal (should be clean speech estimate)
            error = mixed[i] - y
            recovered[i] = error
            
            # LMS update
            w += 2 * mu * error * x_buffer
        
        return recovered, w
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio (dB)"""
        signal_power = np.mean(signal ** 2) + 1e-12
        noise_power = np.mean(noise ** 2) + 1e-12
        
        if noise_power < 1e-12:
            return float('inf')  # No noise case
        
        snr = 10.0 * np.log10(signal_power / noise_power)
        
        # Handle NaN and infinity
        if np.isnan(snr) or np.isinf(snr):
            return 0.0
            
        return snr
    
    def generate_mask_params(self, length: int, scale_factor: float, mask_type: str = "multi_tone", 
                            identifier: str = None) -> Dict:
        """
        Generate masking parameters for transmission to authorized party
        
        Args:
            length: Audio length (number of samples)
            scale_factor: Scale factor for mask signal
            mask_type: Type of masking noise
            identifier: Optional unique identifier for this communication
            
        Returns:
            Dictionary containing all parameters needed to regenerate mask
        """
        # Generate cryptographically secure random seed
        seed = secrets.randbits(32)
        
        # Generate unique identifier if not provided
        if identifier is None:
            identifier = str(uuid.uuid4())
        
        # Current timestamp
        timestamp = int(time.time())
        
        mask_params = {
            # Core generation parameters
            'seed': seed,
            'length': length,
            'sample_rate': self.sr,
            'mask_type': mask_type,
            'scale_factor': float(scale_factor),
            
            # Security and tracking fields
            'timestamp': timestamp,
            'identifier': identifier,
            
            # Additional metadata
            'version': '1.0',
            'target_snr_db': self.target_snr_db
        }
        
        return mask_params
    
    def regenerate_mask_from_params(self, mask_params: Dict) -> np.ndarray:
        """
        Regenerate masking noise from parameters (used by authorized party)
        
        Args:
            mask_params: Dictionary containing mask generation parameters
            
        Returns:
            Regenerated mask signal
        """
        # Extract parameters
        seed = mask_params['seed']
        length = mask_params['length']
        sr = mask_params['sample_rate']
        mask_type = mask_params['mask_type']
        scale_factor = mask_params['scale_factor']
        
        # Regenerate mask with same seed
        mask = self.generate_voice_like_mask(length, sr, mask_type, seed=seed)
        
        # Apply scale factor
        scaled_mask = mask * scale_factor
        
        return scaled_mask
    
    def save_mask_params(self, mask_params: Dict, output_path: str, receiver_public_key: str = None):
        """
        Save masking parameters to JSON file (with optional encryption)
        
        Args:
            mask_params: Masking parameters dictionary
            output_path: Output file path
            receiver_public_key: Receiver's public key path (for encryption)
        """
        # Convert Path to string if needed
        output_path_str = str(output_path)
        
        # Always save plain JSON first
        with open(output_path_str, 'w', encoding='utf-8') as f:
            json.dump(mask_params, f, indent=2, ensure_ascii=False)
        
        # If encryption is enabled and public key is provided, also save encrypted version
        if self.enable_encryption and self.crypto and receiver_public_key:
            # Create encrypted version with different filename
            encrypted_path = output_path_str.replace('.json', '_encrypted.json')
            encrypted_package = self.crypto.hybrid_encrypt(mask_params, receiver_public_key)
            with open(encrypted_path, 'w', encoding='utf-8') as f:
                json.dump(encrypted_package, f, indent=2, ensure_ascii=False)
            print(f"🔒 Parameters encrypted and saved (Hybrid encryption: RSA+AES)")
            print(f"   Plain version: {output_path_str}")
            print(f"   Encrypted version: {encrypted_path}")
        else:
            print(f"📄 Parameters saved (plain JSON)")
    
    def load_mask_params(self, params_path: str, receiver_private_key: str = None) -> Dict:
        """
        Load masking parameters from JSON file (with optional decryption)
        
        Args:
            params_path: Parameters file path
            receiver_private_key: Receiver's private key path (for decryption)
            
        Returns:
            Masking parameters dictionary
        """
        with open(params_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data is encrypted
        if 'encryption_method' in data and 'encrypted_session_key' in data:
            # Data is encrypted, need to decrypt
            if not self.crypto:
                raise RuntimeError("Encryption module not loaded, cannot decrypt. Please install: pip install cryptography")
            if not receiver_private_key:
                raise ValueError("Data is encrypted, need to provide receiver private key path for decryption")
            
            print("🔓 Detected encrypted data, decrypting...")
            mask_params = self.crypto.hybrid_decrypt(data, receiver_private_key)
            print("✓ Decryption successful")
            return mask_params
        else:
            # Data is plain JSON
            return data
    
    def process_audio_pair(self, clean_path: str, output_prefix: str = "", mask_type: str = "voice_like",
                           receiver_public_key: str = None) -> dict:
        """
        Process audio pair: clean speech -> masking -> mixing -> recovery
        
        Args:
            clean_path: Clean speech file path
            output_prefix: Output file prefix
            
        Returns:
            Processing results dictionary
        """
        # 1. Load clean speech
        clean, _ = self.load_audio(clean_path)
        print(f"Loading clean speech: {clean_path}, length: {len(clean)/self.sr:.2f}s")
        
        # 2. Generate random seed for this session
        seed = secrets.randbits(32)
        
        # 3. Generate masking noise with seed
        mask = self.generate_voice_like_mask(len(clean), mask_type=mask_type, seed=seed)
        print(f"Generating {mask_type} masking noise (seed: {seed})")
        
        # 4. Mix signals
        mixed, scaled_mask = self.mix_signals(clean, mask)
        print(f"Mixing signals, target SNR: {self.target_snr_db:.1f}dB")
        
        # 5. Calculate scale factor
        mask_rms = np.sqrt(np.mean(mask ** 2) + 1e-12)
        scaled_mask_rms = np.sqrt(np.mean(scaled_mask ** 2) + 1e-12)
        scale_factor = scaled_mask_rms / mask_rms if mask_rms > 1e-12 else 1.0
        
        # 6. Generate mask parameters (for transmission to authorized party)
        mask_params = self.generate_mask_params(
            length=len(clean),
            scale_factor=scale_factor,
            mask_type=mask_type
        )
        print(f"Generated mask parameters (identifier: {mask_params['identifier']})")
        
        # 7. Authorized recovery
        recovered, filter_taps = self.lms_recovery(mixed, scaled_mask)
        print("Executing LMS authorized recovery")
        
        # 8. Calculate performance metrics
        snr_input = self.calculate_snr(clean, scaled_mask)
        snr_after = self.calculate_snr(clean, recovered - clean)
        improvement = snr_after - snr_input
        
        print(f"Input SNR: {snr_input:.2f}dB")
        print(f"Recovery SNR: {snr_after:.2f}dB")
        print(f"SNR improvement: {improvement:.2f}dB")
        
        # 9. Detailed quality assessment (if available)
        if self.metrics_calc:
            print("\n📊 Detailed Quality Assessment:")
            mixed_metrics = self.metrics_calc.calculate_all_metrics(clean, mixed)
            recovery_metrics = self.metrics_calc.evaluate_recovery_quality(clean, recovered)
            
            print(f"Mixed signal quality:")
            print(f"  - SNR: {mixed_metrics['snr_db']:.2f} dB")
            print(f"  - STOI: {mixed_metrics['stoi']:.3f}")
            print(f"  - Cosine similarity: {mixed_metrics['cosine_similarity']:.3f}")
            print(f"  - SI-SNR: {mixed_metrics['si_snr_db']:.2f} dB")
            
            print(f"Recovery signal quality:")
            print(f"  - SNR: {recovery_metrics['snr_db']:.2f} dB")
            print(f"  - STOI: {recovery_metrics['stoi']:.3f}")
            print(f"  - Signal preservation: {recovery_metrics['signal_preservation']:.3f}")
            print(f"  - Cosine similarity: {recovery_metrics['cosine_similarity']:.3f}")
            print(f"  - SI-SNR: {recovery_metrics['si_snr_db']:.2f} dB")
        
        # 10. Save files
        if not output_prefix:
            output_prefix = Path(clean_path).stem
            
        # Save to organized output subdirectories
        clean_out = self.clean_audio_dir / f"{output_prefix}_clean.wav"
        mask_out = self.mask_audio_dir / f"{output_prefix}_mask_{mask_type}.wav"
        mixed_out = self.mixed_audio_dir / f"{output_prefix}_mixed_{mask_type}.wav"
        recovered_out = self.recovered_audio_dir / f"{output_prefix}_recovered_{mask_type}.wav"
        params_out = self.params_dir / f"{output_prefix}_mask_params_{mask_type}.json"
        
        self.save_audio(clean_out, clean)
        self.save_audio(mixed_out, mixed)
        self.save_audio(recovered_out, recovered)
        
        # Save mask audio only in non-production mode (for demo/debugging)
        if not self.production_mode:
            self.save_audio(mask_out, scaled_mask)
            print(f"Saved mask audio (demo mode): {mask_out}")
        else:
            print("Production mode: mask audio not saved (use mask_params instead)")
        
        # Always save mask parameters (for transmission to authorized party)
        self.save_mask_params(mask_params, params_out, receiver_public_key)
        print(f"Saved mask parameters: {params_out}")
        
        # 11. Return results
        output_files = {
            'clean': str(clean_out),
            'mixed': str(mixed_out),
            'recovered': str(recovered_out),
            'mask_params': str(params_out)
        }
        
        # Only include mask audio file path in non-production mode
        if not self.production_mode:
            output_files['mask'] = str(mask_out)
        
        results = {
            'input_file': clean_path,
            'output_files': output_files,
            'metrics': {
                'input_snr_db': snr_input,
                'output_snr_db': snr_after,
                'improvement_db': improvement,
                'signal_length_sec': len(clean) / self.sr
            },
            'parameters': {
                'sample_rate': self.sr,
                'target_snr_db': self.target_snr_db,
                'filter_order': 128,
                'learning_rate': 0.01
            },
            'mask_params': mask_params,
            'production_mode': self.production_mode
        }
        
        return results
    
    def batch_process(self, clean_files: List[str], output_prefixes: List[str] = None) -> List[dict]:
        """Batch process multiple audio files"""
        if output_prefixes is None:
            output_prefixes = [Path(f).stem for f in clean_files]
            
        results = []
        for clean_file, prefix in zip(clean_files, output_prefixes):
            try:
                result = self.process_audio_pair(clean_file, prefix)
                results.append(result)
                print(f"✓ Processing completed: {prefix}")
            except Exception as e:
                print(f"✗ Processing failed: {prefix}, error: {e}")
                results.append(None)
                
        return results
    
    def authorized_recovery(self, mixed_audio_path: str, params_path: str, output_path: str,
                           receiver_private_key: str = None) -> dict:
        """
        Authorized party recovery: Use mask parameters to recover clean audio from mixed audio
        
        Args:
            mixed_audio_path: Mixed audio file path
            params_path: Mask parameters file path (may be encrypted)
            output_path: Output recovered audio file path
            receiver_private_key: Receiver's private key path (for decryption if needed)
            
        Returns:
            Recovery results dictionary
        """
        print("=== Authorized Recovery Process ===")
        
        # 1. Load mixed audio
        print(f"1. Loading mixed audio: {mixed_audio_path}")
        mixed, _ = self.load_audio(mixed_audio_path)
        
        # 2. Load and decrypt (if needed) mask parameters
        print(f"2. Loading mask parameters: {params_path}")
        mask_params = self.load_mask_params(params_path, receiver_private_key)
        
        # 3. Regenerate mask from parameters
        print("3. Regenerating mask from parameters...")
        scaled_mask = self.regenerate_mask_from_params(mask_params)
        
        # 4. LMS recovery
        print("4. Executing LMS adaptive recovery...")
        recovered, filter_taps = self.lms_recovery(mixed, scaled_mask)
        
        # 5. Save recovered audio to organized directory
        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute():
            # If relative path, save to recovered audio directory
            filename = output_path_obj.name
            output_path = self.recovered_audio_dir / filename
        
        print(f"5. Saving recovered audio: {output_path}")
        self.save_audio(output_path, recovered)
        
        print("✓ Authorized recovery completed!")
        
        # 6. Return results
        results = {
            'mixed_audio': mixed_audio_path,
            'params_file': params_path,
            'recovered_audio': output_path,
            'mask_params': mask_params,
            'encrypted': 'encryption_method' in open(params_path, 'r').read()
        }
        
        return results
    
    def generate_keypair_for_receiver(self, receiver_name: str = "receiver") -> dict:
        """
        Generate RSA keypair for receiver
        
        Args:
            receiver_name: Receiver identifier name
            
        Returns:
            Keypair information dictionary
        """
        if not self.crypto:
            raise RuntimeError("Encryption module not loaded. Please install: pip install cryptography")
        
        print(f"=== Generating Receiver Keypair: {receiver_name} ===")
        
        # Generate keypair
        private_pem, public_pem = self.crypto.generate_rsa_keypair(2048)
        
        # Save to keys directory
        private_path = self.keys_dir / f"{receiver_name}_private.pem"
        public_path = self.keys_dir / f"{receiver_name}_public.pem"
        
        self.crypto.save_keypair(private_pem, public_pem, str(private_path), str(public_path))
        
        print(f"✓ Private key saved: {private_path}")
        print(f"✓ Public key saved: {public_path}")
        print(f"⚠️  Warning: Please keep private key file secure!")
        
        return {
            'private_key': str(private_path),
            'public_key': str(public_path),
            'receiver_name': receiver_name
        }
    


def main():
    """Main function - Audio Privacy Protection System"""
    
    # ========== ENVIRONMENT CONFIGURATION ==========
    # You can manually change this value: "dev" or "prod"
    ENVIRONMENT = "dev"  # Options: "dev" or "prod"
    # ========================================================
    
    # Validate environment
    if ENVIRONMENT not in ["dev", "prod"]:
        raise ValueError(f"Invalid ENVIRONMENT value: {ENVIRONMENT}. Must be 'dev' or 'prod'.")
    
    # Set production mode based on environment
    production_mode = (ENVIRONMENT == "prod")
    
    parser = argparse.ArgumentParser(description='Audio Privacy Protection System')
    parser.add_argument('--input', '-i', type=str, help='Input audio file path')
    parser.add_argument('--batch', '-b', type=str, help='Batch processing directory path')
    parser.add_argument('--snr', type=float, default=0.0, help='Target SNR (dB)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate (Hz)')
    parser.add_argument('--mask-type', type=str, default='multi_tone', 
                       choices=['voice_like', 'multi_tone'],
                       help='Type of masking noise')
    
    # Encryption related arguments
    parser.add_argument('--enable-encryption', action='store_true', 
                       help='Enable hybrid encryption for mask parameters')
    parser.add_argument('--generate-keypair', type=str, 
                       help='Generate RSA keypair for specified receiver name')
    parser.add_argument('--public-key', type=str, 
                       help='Receiver public key path (for encryption)')
    parser.add_argument('--private-key', type=str, 
                       help='Receiver private key path (for decryption)')
    
    # Recovery mode arguments
    parser.add_argument('--recover', action='store_true',
                       help='Recovery mode: recover clean audio from mixed audio')
    parser.add_argument('--mixed-audio', type=str,
                       help='Mixed audio file path (for recovery mode)')
    parser.add_argument('--params-file', type=str,
                       help='Mask parameters file path (for recovery mode)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (for recovery mode)')
    
    args = parser.parse_args()
    
    print("=== Audio Privacy Protection System ===")
    print("Based on sound masking techniques for smartphone audio privacy")
    print(f"Environment: {ENVIRONMENT.upper()}")
    if production_mode:
        print("🔒 Production Mode: Mask audio will not be saved (only parameters)")
    else:
        print("🔧 Dev Mode: All files including mask audio will be saved")
    if args.enable_encryption:
        print("🔐 Encryption: Enabled (Hybrid RSA+AES)")
    print()
    
    # Initialize system
    system = AudioPrivacySystem(sample_rate=args.sample_rate, target_snr_db=args.snr, 
                               production_mode=production_mode, enable_encryption=args.enable_encryption)
    
    # Handle keypair generation
    if args.generate_keypair:
        keypair_info = system.generate_keypair_for_receiver(args.generate_keypair)
        print("\nInstructions:")
        print("- Sender uses public key to encrypt parameters")
        print("- Receiver uses private key to decrypt parameters")
        return
    
    # Auto-generate keypair if encryption is enabled but no keypair exists
    if args.enable_encryption and not list(system.keys_dir.glob("*_public.pem")):
        print("🔑 No existing keypair found. Generating default keypair...")
        keypair_info = system.generate_keypair_for_receiver("default_receiver")
        print("✓ Default keypair generated for encryption")
        # Use the generated keypair for subsequent operations
        args.public_key = keypair_info['public_key']
    
    # Handle recovery mode
    if args.recover:
        if not args.mixed_audio or not args.params_file or not args.output:
            print("Error: Recovery mode requires --mixed-audio, --params-file and --output")
            return
        
        result = system.authorized_recovery(
            args.mixed_audio,
            args.params_file,
            args.output,
            args.private_key
        )
        
        print("\nRecovery Results:")
        print(f"- Mixed audio: {result['mixed_audio']}")
        print(f"- Parameters file: {result['params_file']}")
        print(f"- Recovered audio: {result['recovered_audio']}")
        print(f"- Encrypted: {'Yes' if result['encrypted'] else 'No'}")
        return
    
    if args.input:
        # Process single file
        print(f"Processing single file: {args.input}")
        # Auto-find public key if encryption is enabled but no key specified
        public_key_to_use = args.public_key
        if args.enable_encryption and not public_key_to_use:
            public_key_files = list(system.keys_dir.glob("*_public.pem"))
            if public_key_files:
                public_key_to_use = str(public_key_files[0])
                print(f"🔑 Using existing public key: {public_key_to_use}")
        
        result = system.process_audio_pair(args.input, mask_type=args.mask_type, 
                                          receiver_public_key=public_key_to_use)
        print(f"\nProcessing results:")
        print(f"- Input SNR: {result['metrics']['input_snr_db']:.2f}dB")
        print(f"- Recovery SNR: {result['metrics']['output_snr_db']:.2f}dB")
        print(f"- SNR improvement: {result['metrics']['improvement_db']:.2f}dB")
        print(f"\nOutput files saved to: {system.output_dir}")
        
    elif args.batch:
        # Batch processing
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"Error: Directory does not exist: {batch_dir}")
            return
            
        audio_files = []
        for ext in ['*.wav', '*.m4a', '*.mp3', '*.flac']:
            audio_files.extend(batch_dir.glob(ext))
        
        if not audio_files:
            print(f"Error: No audio files found in directory: {batch_dir}")
            return
            
        print(f"Batch processing {len(audio_files)} files")
        results = system.batch_process([str(f) for f in audio_files])
        
        valid_results = [r for r in results if r is not None]
        if valid_results:
            avg_improvement = np.mean([r['metrics']['improvement_db'] for r in valid_results])
            print(f"\nBatch processing results:")
            print(f"- Successfully processed: {len(valid_results)}/{len(results)} files")
            print(f"- Average SNR improvement: {avg_improvement:.2f}dB")
        
    else:
        # Default: Process all files in dataset/input
        input_files = []
        for ext in ['*.wav', '*.m4a', '*.mp3', '*.flac']:
            input_files.extend(system.input_dir.glob(ext))
        
        if input_files:
            print(f"Found {len(input_files)} input audio files")
            print("Processing all files...")
            
            results = []
            for i, input_file in enumerate(input_files, 1):
                print(f"\n[{i}/{len(input_files)}] Processing: {input_file.name}")
                try:
                    result = system.process_audio_pair(str(input_file), mask_type=args.mask_type, 
                                                    receiver_public_key=args.public_key)
                    results.append((input_file.name, result))
                    print(f"✓ Completed: SNR improvement = {result['metrics']['improvement_db']:.2f}dB")
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    results.append((input_file.name, None))
            
            # Summary
            successful = [r for r in results if r[1] is not None]
            if successful:
                avg_improvement = np.mean([r[1]['metrics']['improvement_db'] for r in successful])
                print(f"\n📊 Processing Summary:")
                print(f"- Total files: {len(input_files)}")
                print(f"- Successful: {len(successful)}")
                print(f"- Failed: {len(results) - len(successful)}")
                print(f"- Average SNR improvement: {avg_improvement:.2f}dB")
                print(f"\nOutput files saved to: {system.output_dir}")
        else:
            print("No input audio files found.")
            print("Please place audio files in dataset/input/ directory, or use:")
            print("  python audio_privacy_system.py --input <file_path>")
            print("  python audio_privacy_system.py --batch <directory_path>")
    
    print("\n=== System Description ===")
    print("1. Clean speech: Original speech signal")
    print("2. Masking noise: Voice-like noise signal")
    print("3. Mixed signal: What eavesdroppers would record (unclear)")
    print("4. Recovered signal: Clear speech recovered by authorized parties")
    print("\nCore principle: Only authorized parties know the exact masking parameters to reverse recovery")


if __name__ == "__main__":
    main()
