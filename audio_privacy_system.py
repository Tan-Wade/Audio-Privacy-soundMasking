#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频隐私保护系统 - 基于声音掩蔽技术的智能手机音频隐私保护
Audio Privacy Protection System using Sound Masking Techniques

核心功能：
1. 对干净语音施加掩蔽噪声（类似"加密"）
2. 生成混合信号（模拟被监听方录到的声音）
3. 授权方使用已知参数进行反向恢复
4. 非授权方只能听到含混的混合信号

作者：基于论文 "Exploiting Sound Masking for Audio Privacy in Smartphones"
"""

import os
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入音频质量评估模块
try:
    from audio_metrics import AudioMetrics
    HAVE_METRICS = True
except ImportError:
    HAVE_METRICS = False

# 音频处理依赖
try:
    import soundfile as sf
    HAVE_SF = True
except ImportError:
    try:
        from scipy.io import wavfile
        HAVE_SF = False
    except ImportError:
        print("警告：需要安装 soundfile 或 scipy 来处理音频文件")
        HAVE_SF = None

class AudioPrivacySystem:
    """音频隐私保护系统主类"""
    
    def __init__(self, sample_rate: int = 16000, target_snr_db: float = 0.0):
        """
        初始化音频隐私保护系统
        
        Args:
            sample_rate: 采样率，默认16kHz（适合语音）
            target_snr_db: 目标信噪比，默认0dB（掩蔽效果较强）
        """
        self.sr = sample_rate
        self.target_snr_db = target_snr_db
        self.output_dir = Path("./audio_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化音频质量评估器
        self.metrics_calc = AudioMetrics(sample_rate) if HAVE_METRICS else None
        
        # 语音特征参数（针对中文语音优化）
        self.voice_params = {
            'f0_range': (80, 300),      # 基频范围
            'formants': [800, 1200, 2500, 3500],  # 共振峰频率
            'speech_band': (200, 4000), # 语音频带
            'syllable_rate': (2, 5),    # 音节速率 (Hz)
        }
        
    def load_audio(self, file_path: str, force_mono: bool = True) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
            
        if HAVE_SF:
            data, sr = sf.read(file_path, always_2d=False)
        else:
            sr, data = wavfile.read(file_path)
            # 归一化到float32
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            else:
                data = data.astype(np.float32) / np.max(1e-9 + np.abs(data))
        
        # 转换为单声道
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        
        # 重采样（如果需要）
        if sr != self.sr:
            data = self._resample(data, sr, self.sr)
            sr = self.sr
            
        return data.astype(np.float32), sr
    
    def save_audio(self, file_path: str, data: np.ndarray, sr: int = None):
        """保存音频文件"""
        if sr is None:
            sr = self.sr
            
        data = np.asarray(data, dtype=np.float32)
        data = np.clip(data, -1.0, 1.0)
        
        if HAVE_SF:
            sf.write(file_path, data, sr, subtype="PCM_16")
        else:
            from scipy.io.wavfile import write as wav_write
            wav_write(file_path, sr, (data * 32767).astype(np.int16))
    
    def _resample(self, data: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
        """简单重采样（线性插值）"""
        if old_sr == new_sr:
            return data
            
        ratio = new_sr / old_sr
        new_len = int(len(data) * ratio)
        x_old = np.linspace(0, 1, num=len(data), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        return np.interp(x_new, x_old, data).astype(np.float32)
    
    def generate_voice_like_mask(self, length: int, sr: int = None) -> np.ndarray:
        """
        生成类语音掩蔽噪声
        基于论文建议，使用语音样式的噪声比白噪声更有效
        """
        if sr is None:
            sr = self.sr
            
        # 1. 生成白噪声
        white_noise = np.random.randn(length).astype(np.float32)
        
        # 2. 带通滤波到语音频带
        filtered_noise = self._bandpass_filter(white_noise, sr)
        
        # 3. 添加音节式调制（模拟语音的能量变化）
        syllable_modulation = self._generate_syllable_modulation(length, sr)
        
        # 4. 组合生成类语音噪声
        voice_like = filtered_noise * syllable_modulation
        
        # 5. 归一化
        voice_like = voice_like / (np.max(np.abs(voice_like)) + 1e-9)
        
        return voice_like.astype(np.float32)
    
    def _bandpass_filter(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """简单的带通滤波器（窗口化sinc）"""
        low_freq, high_freq = self.voice_params['speech_band']
        
        # 设计FIR滤波器
        numtaps = 513
        nyq = sr / 2.0
        f1 = low_freq / nyq
        f2 = high_freq / nyq
        
        # 窗口化sinc带通滤波器
        n = np.arange(numtaps) - (numtaps - 1) / 2.0
        
        def sinc(x):
            return np.sinc(x / np.pi)
            
        h = (2 * f2 * sinc(2 * np.pi * f2 * n) - 2 * f1 * sinc(2 * np.pi * f1 * n))
        
        # Hann窗
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(numtaps) / (numtaps - 1)))
        h = h * window
        
        # 归一化
        h = h / np.sum(h)
        
        # 应用滤波器
        filtered = np.convolve(signal, h, mode='same')
        return filtered.astype(np.float32)
    
    def _generate_syllable_modulation(self, length: int, sr: int) -> np.ndarray:
        """生成音节式调制信号"""
        # 随机音节速率
        syllable_rate = np.random.uniform(*self.voice_params['syllable_rate'])
        
        # 生成随机包络
        env_length = max(1, length // 400)  # 粗粒度包络
        envelope = np.abs(np.random.randn(env_length)).astype(np.float32)
        
        # 上采样到信号长度
        t_env = np.linspace(0, env_length - 1, num=length)
        envelope_up = np.interp(t_env, np.arange(env_length), envelope)
        
        # 平滑包络（模拟音节边界）
        smooth_kernel = np.ones(51, dtype=np.float32) / 51.0
        envelope_smooth = np.convolve(envelope_up, smooth_kernel, mode='same')
        envelope_smooth = envelope_smooth / (np.max(np.abs(envelope_smooth)) + 1e-9)
        
        # 添加音节式调制
        t = np.linspace(0, length / sr, length, endpoint=False)
        syllable_mod = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t))
        
        return envelope_smooth * syllable_mod
    
    def mix_signals(self, clean: np.ndarray, mask: np.ndarray, target_snr_db: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        将干净信号和掩蔽信号按指定信噪比混合
        
        Args:
            clean: 干净语音信号
            mask: 掩蔽噪声信号
            target_snr_db: 目标信噪比（dB）
            
        Returns:
            mixed: 混合后的信号
            scaled_mask: 缩放后的掩蔽信号
        """
        if target_snr_db is None:
            target_snr_db = self.target_snr_db
            
        # 计算RMS
        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-12)
        mask_rms = np.sqrt(np.mean(mask ** 2) + 1e-12)
        
        if mask_rms < 1e-12:
            return clean.copy(), mask.copy()
        
        # 计算所需的掩蔽信号幅度
        desired_mask_rms = clean_rms / (10.0 ** (target_snr_db / 20.0))
        scale_factor = desired_mask_rms / mask_rms
        
        # 缩放掩蔽信号
        scaled_mask = mask * scale_factor
        
        # 混合信号
        mixed = clean + scaled_mask
        
        return mixed, scaled_mask
    
    def lms_recovery(self, mixed: np.ndarray, mask_ref: np.ndarray, 
                     mu: float = 0.01, filter_order: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        LMS自适应滤波器进行授权恢复
        
        Args:
            mixed: 观测信号 (clean + mask)
            mask_ref: 参考掩蔽信号（授权方已知）
            mu: 学习率
            filter_order: 滤波器阶数
            
        Returns:
            recovered: 恢复的干净信号
            filter_taps: 滤波器系数
        """
        n = len(mixed)
        w = np.zeros(filter_order, dtype=np.float32)
        x_buffer = np.zeros(filter_order, dtype=np.float32)
        recovered = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            # 更新输入缓冲区
            x_buffer[1:] = x_buffer[:-1]
            x_buffer[0] = mask_ref[i] if i < len(mask_ref) else 0.0
            
            # 计算滤波器输出
            y = np.dot(w, x_buffer)
            
            # 计算误差信号（这应该是干净语音的估计）
            error = mixed[i] - y
            recovered[i] = error
            
            # LMS更新
            w += 2 * mu * error * x_buffer
        
        return recovered, w
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """计算信噪比（dB）"""
        signal_power = np.mean(signal ** 2) + 1e-12
        noise_power = np.mean(noise ** 2) + 1e-12
        
        if noise_power < 1e-12:
            return float('inf')  # 无噪声情况
        
        snr = 10.0 * np.log10(signal_power / noise_power)
        
        # 处理NaN和无穷大值
        if np.isnan(snr) or np.isinf(snr):
            return 0.0
            
        return snr
    
    def process_audio_pair(self, clean_path: str, output_prefix: str = "") -> dict:
        """
        处理音频对：干净语音 -> 掩蔽 -> 混合 -> 恢复
        
        Args:
            clean_path: 干净语音文件路径
            output_prefix: 输出文件前缀
            
        Returns:
            处理结果字典
        """
        # 1. 加载干净语音
        clean, _ = self.load_audio(clean_path)
        print(f"加载干净语音: {clean_path}, 长度: {len(clean)/self.sr:.2f}秒")
        
        # 2. 生成掩蔽噪声
        mask = self.generate_voice_like_mask(len(clean))
        print("生成类语音掩蔽噪声")
        
        # 3. 混合信号
        mixed, scaled_mask = self.mix_signals(clean, mask)
        print(f"混合信号，目标SNR: {self.target_snr_db:.1f}dB")
        
        # 4. 授权恢复
        recovered, filter_taps = self.lms_recovery(mixed, scaled_mask)
        print("执行LMS授权恢复")
        
        # 5. 计算性能指标
        snr_input = self.calculate_snr(clean, scaled_mask)
        snr_after = self.calculate_snr(clean, recovered - clean)
        improvement = snr_after - snr_input
        
        print(f"输入SNR: {snr_input:.2f}dB")
        print(f"恢复后SNR: {snr_after:.2f}dB")
        print(f"SNR改善: {improvement:.2f}dB")
        
        # 6. 详细质量评估（如果可用）
        if self.metrics_calc:
            print("\n📊 详细质量评估:")
            mixed_metrics = self.metrics_calc.calculate_all_metrics(clean, mixed)
            recovery_metrics = self.metrics_calc.evaluate_recovery_quality(clean, recovered)
            
            print(f"混合信号质量:")
            print(f"  - SNR: {mixed_metrics['snr_db']:.2f} dB")
            print(f"  - STOI: {mixed_metrics['stoi']:.3f}")
            
            print(f"恢复信号质量:")
            print(f"  - SNR: {recovery_metrics['snr_db']:.2f} dB")
            print(f"  - STOI: {recovery_metrics['stoi']:.3f}")
            print(f"  - 信号保持度: {recovery_metrics['signal_preservation']:.3f}")
        
        # 7. 保存文件
        if not output_prefix:
            output_prefix = Path(clean_path).stem
            
        clean_out = self.output_dir / f"{output_prefix}_clean.wav"
        mask_out = self.output_dir / f"{output_prefix}_mask.wav"
        mixed_out = self.output_dir / f"{output_prefix}_mixed.wav"
        recovered_out = self.output_dir / f"{output_prefix}_recovered.wav"
        
        self.save_audio(clean_out, clean)
        self.save_audio(mask_out, scaled_mask)
        self.save_audio(mixed_out, mixed)
        self.save_audio(recovered_out, recovered)
        
        # 8. 返回结果
        results = {
            'input_file': clean_path,
            'output_files': {
                'clean': str(clean_out),
                'mask': str(mask_out),
                'mixed': str(mixed_out),
                'recovered': str(recovered_out)
            },
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
            }
        }
        
        return results
    
    def batch_process(self, clean_files: List[str], output_prefixes: List[str] = None) -> List[dict]:
        """批量处理多个音频文件"""
        if output_prefixes is None:
            output_prefixes = [Path(f).stem for f in clean_files]
            
        results = []
        for clean_file, prefix in zip(clean_files, output_prefixes):
            try:
                result = self.process_audio_pair(clean_file, prefix)
                results.append(result)
                print(f"✓ 完成处理: {prefix}")
            except Exception as e:
                print(f"✗ 处理失败: {prefix}, 错误: {e}")
                results.append(None)
                
        return results
    


def main():
    """主函数 - 演示音频隐私保护系统"""
    print("=== 音频隐私保护系统演示 ===")
    print("基于声音掩蔽技术的智能手机音频隐私保护")
    print()
    
    # 初始化系统
    system = AudioPrivacySystem(sample_rate=16000, target_snr_db=0.0)
    
    # 检查是否有现有音频文件
    existing_files = []
    for filename in ['01_clean.wav', '02_mask.wav', '03_mixed.wav', '04_recovered.wav']:
        if os.path.exists(filename):
            existing_files.append(filename)
    
    if existing_files:
        print(f"发现现有音频文件: {existing_files}")
        print("使用现有文件进行演示...")
        
        # 使用现有的clean文件
        if '01_clean.wav' in existing_files:
            result = system.process_audio_pair('01_clean.wav', 'demo')
            print(f"\n处理结果:")
            print(f"- 输入SNR: {result['metrics']['input_snr_db']:.2f}dB")
            print(f"- 恢复后SNR: {result['metrics']['output_snr_db']:.2f}dB")
            print(f"- SNR改善: {result['metrics']['improvement_db']:.2f}dB")
            print(f"\n输出文件保存在: {system.output_dir}")
    else:
        print("未发现现有音频文件。")
        print("请将你的8位数字录音文件命名为 '01_clean.wav' 并放在项目根目录，然后重新运行。")
        print("或者运行 'python demo.py' 进行快速演示。")
    
    print("\n=== 系统说明 ===")
    print("1. 干净语音: 原始语音信号")
    print("2. 掩蔽噪声: 类语音样式的噪声信号")
    print("3. 混合信号: 模拟被监听方录到的声音（含混不清）")
    print("4. 恢复信号: 授权方使用已知参数恢复的清晰语音")
    print("\n核心原理：只有授权方知道掩蔽噪声的精确参数，可以反向恢复原始语音")


if __name__ == "__main__":
    main()
