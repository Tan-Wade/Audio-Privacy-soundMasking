#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Privacy Protection System using Sound Masking Techniques
音频隐私保护系统 - 基于声音掩蔽技术的智能手机音频隐私保护

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
    """Audio Privacy Protection System Main Class 音频隐私保护系统主类"""
    
    def __init__(self, sample_rate: int = 16000, target_snr_db: float = 0.0, production_mode: bool = False, 
                 enable_encryption: bool = False):
        """
        Initialize Audio Privacy Protection System
        初始化音频隐私保护系统
        
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
        
        # Setup input/output directories 设置输入输出目录
        self.dataset_dir = Path("./dataset")
        self.input_dir = self.dataset_dir / "input"
        self.output_dir = self.dataset_dir / "output"
        self.keys_dir = self.dataset_dir / "keys"
        
        # Create directories 创建目录
        self.dataset_dir.mkdir(exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.keys_dir.mkdir(exist_ok=True)
        
        # Initialize audio quality evaluator 初始化音频质量评估器
        self.metrics_calc = AudioMetrics(sample_rate) if HAVE_METRICS else None
        
        # Initialize encryption module 初始化加密模块
        self.crypto = HybridEncryption() if (HAVE_ENCRYPTION and enable_encryption) else None
        
        # Voice feature parameters 语音特征参数
        self.voice_params = {
            'speech_band': (200, 4000), # Speech frequency band 语音频带
            'syllable_rate': (2, 5),    # Syllable rate (Hz) 音节速率
        }
        
    def load_audio(self, file_path: str, force_mono: bool = True) -> Tuple[np.ndarray, int]:
        """Load audio file 加载音频文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        # Check if soundfile is available 检查soundfile是否可用
        try:
            import soundfile as sf
            have_soundfile = True
        except ImportError:
            have_soundfile = False
        
        # For non-WAV formats, must use soundfile 对于非WAV格式，必须使用soundfile
        if file_ext in ['.m4a', '.mp3', '.flac', '.ogg']:
            if not have_soundfile:
                raise RuntimeError(f"File format {file_ext} requires soundfile library, install: pip install soundfile")
            try:
                data, sr = sf.read(file_path, always_2d=False)
            except Exception as e:
                raise RuntimeError(f"Cannot read audio file {file_path}: {e}")
        elif have_soundfile:
            # Try soundfile 尝试soundfile
            try:
                data, sr = sf.read(file_path, always_2d=False)
            except Exception:
                # If soundfile fails, try scipy (WAV only) 如果soundfile失败，尝试scipy（仅WAV格式）
                if file_ext == '.wav':
                    sr, data = wavfile.read(file_path)
                    # Normalize to float32 归一化到float32
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    else:
                        data = data.astype(np.float32) / np.max(1e-9 + np.abs(data))
                else:
                    raise RuntimeError(f"Cannot read audio file {file_path}, ensure soundfile library is installed")
        else:
            # Only scipy available, WAV format only 只有scipy可用，仅支持WAV格式
            if file_ext != '.wav':
                raise RuntimeError(f"File format {file_ext} requires soundfile library, install: pip install soundfile")
            
            sr, data = wavfile.read(file_path)
            # Normalize to float32 归一化到float32
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            else:
                data = data.astype(np.float32) / np.max(1e-9 + np.abs(data))
        
        # Convert to mono 转换为单声道
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        
        # Resample if needed 重采样（如果需要）
        if sr != self.sr:
            data = self._resample(data, sr, self.sr)
            sr = self.sr
            
        return data.astype(np.float32), sr
    
    def save_audio(self, file_path: str, data: np.ndarray, sr: int = None):
        """Save audio file 保存音频文件"""
        if sr is None:
            sr = self.sr
            
        data = np.asarray(data, dtype=np.float32)
        data = np.clip(data, -1.0, 1.0)
        
        # Try soundfile first 优先尝试soundfile
        try:
            import soundfile as sf
            sf.write(file_path, data, sr, subtype="PCM_16")
        except ImportError:
            # Fallback to scipy 回退到scipy
            from scipy.io.wavfile import write as wav_write
            wav_write(file_path, sr, (data * 32767).astype(np.int16))
    
    def _resample(self, data: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
        """Simple resampling (linear interpolation) 简单重采样（线性插值）"""
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
        生成掩蔽噪声
        
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
        """Original voice-like masking noise 原始的类语音掩蔽噪声"""
        # 1. Generate white noise 生成白噪声
        white_noise = np.random.randn(length).astype(np.float32)
        
        # 2. Bandpass filter to speech frequency band 带通滤波到语音频带
        filtered_noise = self._bandpass_filter(white_noise, sr)
        
        # 3. Add syllable modulation (simulate speech energy changes) 添加音节式调制
        syllable_modulation = self._generate_syllable_modulation(length, sr)
        
        # 4. Combine to generate voice-like noise 组合生成类语音噪声
        voice_like = filtered_noise * syllable_modulation
        
        # 5. Normalize and boost volume 归一化并大幅提升音量
        voice_like = voice_like / (np.max(np.abs(voice_like)) + 1e-9)
        voice_like *= 2.0  # STRONG boost volume for aggressive masking effect 大幅提升音量以获得激进掩蔽效果
        
        return voice_like.astype(np.float32)
    
    
    def _generate_multi_tone_mask(self, length: int, sr: int) -> np.ndarray:
        """Multi-tone masking with speech-like characteristics 多音调掩蔽"""
        t = np.linspace(0, length / sr, length, endpoint=False)
        mask = np.zeros(length, dtype=np.float32)
        
        # Generate multiple tones in speech frequency range 在语音频段生成多个音调
        speech_tones = [300, 500, 800, 1200, 1800, 2500, 3200]  # Common speech frequencies
        
        for i, freq in enumerate(speech_tones):
            # Add slight frequency modulation 添加轻微频率调制
            fm_freq = 0.5 + i * 0.2
            freq_mod = freq * (1 + 0.1 * np.sin(2 * np.pi * fm_freq * t))
            
            # Generate tone with amplitude modulation 生成带幅度调制的音调
            amplitude = 0.8 + 0.4 * np.sin(2 * np.pi * (0.3 + i * 0.1) * t)
            phase = 2 * np.pi * freq_mod * t + np.random.uniform(0, 2*np.pi)
            
            tone = amplitude * np.sin(phase)
            mask += tone * 0.15
        
        # Add some filtered noise for texture 添加一些滤波噪声增加纹理
        noise = np.random.randn(length).astype(np.float32)
        filtered_noise = self._bandpass_filter(noise, sr)
        mask += filtered_noise * 0.3
        
        # Normalize and boost volume 归一化并大幅提升音量
        mask = mask / (np.max(np.abs(mask)) + 1e-9)
        mask *= 2.0  # STRONG boost volume for aggressive masking effect 大幅提升音量以获得激进掩蔽效果
        
        return mask.astype(np.float32)
    
    
    def _bandpass_filter_custom(self, signal: np.ndarray, sr: int, low_freq: float, high_freq: float) -> np.ndarray:
        """Custom bandpass filter with specific frequency range 自定义带通滤波器"""
        numtaps = 257
        nyq = sr / 2.0
        f1 = low_freq / nyq
        f2 = high_freq / nyq
        
        # Windowed sinc bandpass filter 窗口化sinc带通滤波器
        n = np.arange(numtaps) - (numtaps - 1) / 2.0
        
        def sinc(x):
            return np.sinc(x / np.pi)
            
        h = (2 * f2 * sinc(2 * np.pi * f2 * n) - 2 * f1 * sinc(2 * np.pi * f1 * n))
        
        # Hann window Hann窗
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(numtaps) / (numtaps - 1)))
        h = h * window
        
        # Normalize 归一化
        h = h / np.sum(h)
        
        # Apply filter 应用滤波器
        filtered = np.convolve(signal, h, mode='same')
        return filtered.astype(np.float32)
    
    def _bandpass_filter(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Simple bandpass filter (windowed sinc) 简单的带通滤波器（窗口化sinc）"""
        low_freq, high_freq = self.voice_params['speech_band']
        
        # Design FIR filter 设计FIR滤波器
        numtaps = 513
        nyq = sr / 2.0
        f1 = low_freq / nyq
        f2 = high_freq / nyq
        
        # Windowed sinc bandpass filter 窗口化sinc带通滤波器
        n = np.arange(numtaps) - (numtaps - 1) / 2.0
        
        def sinc(x):
            return np.sinc(x / np.pi)
            
        h = (2 * f2 * sinc(2 * np.pi * f2 * n) - 2 * f1 * sinc(2 * np.pi * f1 * n))
        
        # Hann window Hann窗
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(numtaps) / (numtaps - 1)))
        h = h * window
        
        # Normalize 归一化
        h = h / np.sum(h)
        
        # Apply filter 应用滤波器
        filtered = np.convolve(signal, h, mode='same')
        return filtered.astype(np.float32)
    
    def _generate_syllable_modulation(self, length: int, sr: int) -> np.ndarray:
        """Generate syllable modulation signal 生成音节式调制信号"""
        # Random syllable rate 随机音节速率
        syllable_rate = np.random.uniform(*self.voice_params['syllable_rate'])
        
        # Generate random envelope 生成随机包络
        env_length = max(1, length // 400)  # Coarse envelope 粗粒度包络
        envelope = np.abs(np.random.randn(env_length)).astype(np.float32)
        
        # Upsample to signal length 上采样到信号长度
        t_env = np.linspace(0, env_length - 1, num=length)
        envelope_up = np.interp(t_env, np.arange(env_length), envelope)
        
        # Smooth envelope (simulate syllable boundaries) 平滑包络（模拟音节边界）
        smooth_kernel = np.ones(51, dtype=np.float32) / 51.0
        envelope_smooth = np.convolve(envelope_up, smooth_kernel, mode='same')
        envelope_smooth = envelope_smooth / (np.max(np.abs(envelope_smooth)) + 1e-9)
        
        # Add syllable modulation 添加音节式调制
        t = np.linspace(0, length / sr, length, endpoint=False)
        syllable_mod = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t))
        
        return envelope_smooth * syllable_mod
    
    def mix_signals(self, clean: np.ndarray, mask: np.ndarray, target_snr_db: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix clean signal and masking signal at specified SNR
        将干净信号和掩蔽信号按指定信噪比混合
        
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
            
        # Calculate RMS 计算RMS
        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-12)
        mask_rms = np.sqrt(np.mean(mask ** 2) + 1e-12)
        
        if mask_rms < 1e-12:
            return clean.copy(), mask.copy()
        
        # Calculate required masking signal amplitude 计算所需的掩蔽信号幅度
        # Apply VERY strong masking by reducing effective SNR significantly 通过大幅降低有效SNR应用极强掩蔽
        effective_snr = target_snr_db - 8.0  # Reduce SNR by 8dB for VERY strong masking
        desired_mask_rms = clean_rms / (10.0 ** (effective_snr / 20.0))
        scale_factor = desired_mask_rms / mask_rms
        
        # Scale masking signal 缩放掩蔽信号
        scaled_mask = mask * scale_factor
        
        # Mix signals 混合信号
        mixed = clean + scaled_mask
        
        return mixed, scaled_mask
    
    def lms_recovery(self, mixed: np.ndarray, mask_ref: np.ndarray, 
                     mu: float = 0.01, filter_order: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        LMS adaptive filter for authorized recovery
        LMS自适应滤波器进行授权恢复
        
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
            # Update input buffer 更新输入缓冲区
            x_buffer[1:] = x_buffer[:-1]
            x_buffer[0] = mask_ref[i] if i < len(mask_ref) else 0.0
            
            # Calculate filter output 计算滤波器输出
            y = np.dot(w, x_buffer)
            
            # Calculate error signal (should be clean speech estimate) 计算误差信号
            error = mixed[i] - y
            recovered[i] = error
            
            # LMS update LMS更新
            w += 2 * mu * error * x_buffer
        
        return recovered, w
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio (dB) 计算信噪比（dB）"""
        signal_power = np.mean(signal ** 2) + 1e-12
        noise_power = np.mean(noise ** 2) + 1e-12
        
        if noise_power < 1e-12:
            return float('inf')  # No noise case 无噪声情况
        
        snr = 10.0 * np.log10(signal_power / noise_power)
        
        # Handle NaN and infinity 处理NaN和无穷大值
        if np.isnan(snr) or np.isinf(snr):
            return 0.0
            
        return snr
    
    def generate_mask_params(self, length: int, scale_factor: float, mask_type: str = "multi_tone", 
                            identifier: str = None) -> Dict:
        """
        Generate masking parameters for transmission to authorized party
        生成掩蔽参数以传输给授权方
        
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
        根据参数重新生成掩蔽噪声（授权方使用）
        
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
        保存掩蔽参数到JSON文件（可选加密）
        
        Args:
            mask_params: Masking parameters dictionary
            output_path: Output file path
            receiver_public_key: Receiver's public key path (for encryption)
        """
        # If encryption is enabled and public key is provided
        if self.enable_encryption and self.crypto and receiver_public_key:
            # Use hybrid encryption
            encrypted_package = self.crypto.hybrid_encrypt(mask_params, receiver_public_key)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(encrypted_package, f, indent=2, ensure_ascii=False)
            print(f"🔒 参数已加密保存（混合加密：RSA+AES）")
        else:
            # Save as plain JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mask_params, f, indent=2, ensure_ascii=False)
    
    def load_mask_params(self, params_path: str, receiver_private_key: str = None) -> Dict:
        """
        Load masking parameters from JSON file (with optional decryption)
        从JSON文件加载掩蔽参数（可选解密）
        
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
                raise RuntimeError("加密模块未加载，无法解密。请安装：pip install cryptography")
            if not receiver_private_key:
                raise ValueError("数据已加密，需要提供接收方私钥路径进行解密")
            
            print("🔓 检测到加密数据，正在解密...")
            mask_params = self.crypto.hybrid_decrypt(data, receiver_private_key)
            print("✓ 解密成功")
            return mask_params
        else:
            # Data is plain JSON
            return data
    
    def process_audio_pair(self, clean_path: str, output_prefix: str = "", mask_type: str = "voice_like",
                           receiver_public_key: str = None) -> dict:
        """
        Process audio pair: clean speech -> masking -> mixing -> recovery
        处理音频对：干净语音 -> 掩蔽 -> 混合 -> 恢复
        
        Args:
            clean_path: Clean speech file path
            output_prefix: Output file prefix
            
        Returns:
            Processing results dictionary
        """
        # 1. Load clean speech 加载干净语音
        clean, _ = self.load_audio(clean_path)
        print(f"Loading clean speech: {clean_path}, length: {len(clean)/self.sr:.2f}s")
        
        # 2. Generate random seed for this session
        seed = secrets.randbits(32)
        
        # 3. Generate masking noise with seed 生成掩蔽噪声
        mask = self.generate_voice_like_mask(len(clean), mask_type=mask_type, seed=seed)
        print(f"Generating {mask_type} masking noise (seed: {seed})")
        
        # 4. Mix signals 混合信号
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
        
        # 7. Authorized recovery 授权恢复
        recovered, filter_taps = self.lms_recovery(mixed, scaled_mask)
        print("Executing LMS authorized recovery")
        
        # 8. Calculate performance metrics 计算性能指标
        snr_input = self.calculate_snr(clean, scaled_mask)
        snr_after = self.calculate_snr(clean, recovered - clean)
        improvement = snr_after - snr_input
        
        print(f"Input SNR: {snr_input:.2f}dB")
        print(f"Recovery SNR: {snr_after:.2f}dB")
        print(f"SNR improvement: {improvement:.2f}dB")
        
        # 9. Detailed quality assessment (if available) 详细质量评估
        if self.metrics_calc:
            print("\n📊 Detailed Quality Assessment:")
            mixed_metrics = self.metrics_calc.calculate_all_metrics(clean, mixed)
            recovery_metrics = self.metrics_calc.evaluate_recovery_quality(clean, recovered)
            
            print(f"Mixed signal quality:")
            print(f"  - SNR: {mixed_metrics['snr_db']:.2f} dB")
            print(f"  - STOI: {mixed_metrics['stoi']:.3f}")
            
            print(f"Recovery signal quality:")
            print(f"  - SNR: {recovery_metrics['snr_db']:.2f} dB")
            print(f"  - STOI: {recovery_metrics['stoi']:.3f}")
            print(f"  - Signal preservation: {recovery_metrics['signal_preservation']:.3f}")
        
        # 10. Save files 保存文件
        if not output_prefix:
            output_prefix = Path(clean_path).stem
            
        # Save to new output directory 保存到新的输出目录
        clean_out = self.output_dir / f"{output_prefix}_clean.wav"
        mask_out = self.output_dir / f"{output_prefix}_mask_{mask_type}.wav"
        mixed_out = self.output_dir / f"{output_prefix}_mixed_{mask_type}.wav"
        recovered_out = self.output_dir / f"{output_prefix}_recovered_{mask_type}.wav"
        params_out = self.output_dir / f"{output_prefix}_mask_params_{mask_type}.json"
        
        self.save_audio(clean_out, clean)
        self.save_audio(mixed_out, mixed)
        self.save_audio(recovered_out, recovered)
        
        # Save mask audio only in non-production mode (for demo/debugging)
        if not self.production_mode:
            self.save_audio(mask_out, scaled_mask)
            print(f"Saved mask audio (demo mode): {mask_out.name}")
        else:
            print("Production mode: mask audio not saved (use mask_params instead)")
        
        # Always save mask parameters (for transmission to authorized party)
        self.save_mask_params(mask_params, params_out, receiver_public_key)
        print(f"Saved mask parameters: {params_out.name}")
        
        # 11. Return results 返回结果
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
        """Batch process multiple audio files 批量处理多个音频文件"""
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
        授权方恢复：使用掩蔽参数从混合音频中恢复干净音频
        
        Args:
            mixed_audio_path: Mixed audio file path
            params_path: Mask parameters file path (may be encrypted)
            output_path: Output recovered audio file path
            receiver_private_key: Receiver's private key path (for decryption if needed)
            
        Returns:
            Recovery results dictionary
        """
        print("=== 授权方恢复流程 ===")
        
        # 1. Load mixed audio 加载混合音频
        print(f"1. 加载混合音频: {mixed_audio_path}")
        mixed, _ = self.load_audio(mixed_audio_path)
        
        # 2. Load and decrypt (if needed) mask parameters 加载并解密（如需要）掩蔽参数
        print(f"2. 加载掩蔽参数: {params_path}")
        mask_params = self.load_mask_params(params_path, receiver_private_key)
        
        # 3. Regenerate mask from parameters 根据参数重新生成掩蔽信号
        print("3. 根据参数重新生成掩蔽信号...")
        scaled_mask = self.regenerate_mask_from_params(mask_params)
        
        # 4. LMS recovery LMS恢复
        print("4. 执行LMS自适应恢复...")
        recovered, filter_taps = self.lms_recovery(mixed, scaled_mask)
        
        # 5. Save recovered audio 保存恢复的音频
        print(f"5. 保存恢复的音频: {output_path}")
        self.save_audio(output_path, recovered)
        
        print("✓ 授权恢复完成！")
        
        # 6. Return results 返回结果
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
        为接收方生成RSA密钥对
        
        Args:
            receiver_name: Receiver identifier name
            
        Returns:
            Keypair information dictionary
        """
        if not self.crypto:
            raise RuntimeError("加密模块未加载。请安装：pip install cryptography")
        
        print(f"=== 生成接收方密钥对: {receiver_name} ===")
        
        # Generate keypair
        private_pem, public_pem = self.crypto.generate_rsa_keypair(2048)
        
        # Save to keys directory
        private_path = self.keys_dir / f"{receiver_name}_private.pem"
        public_path = self.keys_dir / f"{receiver_name}_public.pem"
        
        self.crypto.save_keypair(private_pem, public_pem, str(private_path), str(public_path))
        
        print(f"✓ 私钥已保存: {private_path}")
        print(f"✓ 公钥已保存: {public_path}")
        print(f"⚠️  警告: 请妥善保管私钥文件！")
        
        return {
            'private_key': str(private_path),
            'public_key': str(public_path),
            'receiver_name': receiver_name
        }
    


def main():
    """Main function - Audio Privacy Protection System 主函数 - 音频隐私保护系统"""
    
    # ========== ENVIRONMENT CONFIGURATION 环境配置 ==========
    # You can manually change this value: "dev" or "prod"
    # 您可以手动修改此值："dev" 或 "prod"
    ENVIRONMENT = "dev"  # Options: "dev" or "prod"
    # ========================================================
    
    # Validate environment 验证环境变量
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
    
    # Encryption related arguments 加密相关参数
    parser.add_argument('--enable-encryption', action='store_true', 
                       help='Enable hybrid encryption for mask parameters')
    parser.add_argument('--generate-keypair', type=str, 
                       help='Generate RSA keypair for specified receiver name')
    parser.add_argument('--public-key', type=str, 
                       help='Receiver public key path (for encryption)')
    parser.add_argument('--private-key', type=str, 
                       help='Receiver private key path (for decryption)')
    
    # Recovery mode arguments 恢复模式参数
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
    
    # Initialize system 初始化系统
    system = AudioPrivacySystem(sample_rate=args.sample_rate, target_snr_db=args.snr, 
                               production_mode=production_mode, enable_encryption=args.enable_encryption)
    
    # Handle keypair generation 处理密钥对生成
    if args.generate_keypair:
        keypair_info = system.generate_keypair_for_receiver(args.generate_keypair)
        print("\n提示：")
        print("- 发送方使用公钥加密参数")
        print("- 接收方使用私钥解密参数")
        return
    
    # Handle recovery mode 处理恢复模式
    if args.recover:
        if not args.mixed_audio or not args.params_file or not args.output:
            print("错误：恢复模式需要指定 --mixed-audio, --params-file 和 --output")
            return
        
        result = system.authorized_recovery(
            args.mixed_audio,
            args.params_file,
            args.output,
            args.private_key
        )
        
        print("\n恢复结果:")
        print(f"- 混合音频: {result['mixed_audio']}")
        print(f"- 参数文件: {result['params_file']}")
        print(f"- 恢复音频: {result['recovered_audio']}")
        print(f"- 是否加密: {'是' if result['encrypted'] else '否'}")
        return
    
    if args.input:
        # Process single file 处理单个文件
        print(f"Processing single file: {args.input}")
        result = system.process_audio_pair(args.input, mask_type=args.mask_type, 
                                          receiver_public_key=args.public_key)
        print(f"\nProcessing results:")
        print(f"- Input SNR: {result['metrics']['input_snr_db']:.2f}dB")
        print(f"- Recovery SNR: {result['metrics']['output_snr_db']:.2f}dB")
        print(f"- SNR improvement: {result['metrics']['improvement_db']:.2f}dB")
        print(f"\nOutput files saved to: {system.output_dir}")
        
    elif args.batch:
        # Batch processing 批量处理
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
        # Default: Process all files in dataset/input 默认：处理dataset/input中的所有文件
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
                    result = system.process_audio_pair(str(input_file), mask_type=args.mask_type)
                    results.append((input_file.name, result))
                    print(f"✓ Completed: SNR improvement = {result['metrics']['improvement_db']:.2f}dB")
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    results.append((input_file.name, None))
            
            # Summary 总结
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
