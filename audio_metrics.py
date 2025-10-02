#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Quality Metrics Module
音频质量评估模块

Provides various audio quality assessment metrics including SNR, PESQ, STOI, etc.
提供多种音频质量评估指标，包括SNR、PESQ、STOI等
"""

import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class AudioMetrics:
    """Audio Quality Assessment Class 音频质量评估类"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        
    def calculate_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        """
        计算信噪比 (Signal-to-Noise Ratio)
        
        Args:
            clean: 干净信号
            noisy: 含噪信号
            
        Returns:
            SNR in dB
        """
        noise = noisy - clean
        signal_power = np.mean(clean ** 2) + 1e-12
        noise_power = np.mean(noise ** 2) + 1e-12
        
        if noise_power < 1e-12:
            return float('inf')
        
        snr = 10.0 * np.log10(signal_power / noise_power)
        
        # 处理NaN和无穷大值
        if np.isnan(snr) or np.isinf(snr):
            return 0.0
            
        return snr
    
    def calculate_si_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        """
        计算尺度不变信噪比 (Scale-Invariant SNR)
        
        Args:
            clean: 干净信号
            noisy: 含噪信号
            
        Returns:
            SI-SNR in dB
        """
        # 计算最佳尺度因子
        s_target = clean
        e_noise = noisy - clean
        
        # 计算尺度因子
        alpha = np.dot(s_target, noisy) / (np.dot(s_target, s_target) + 1e-12)
        s_hat = alpha * s_target
        e_residual = noisy - s_hat
        
        # 计算SI-SNR
        si_snr = 10.0 * np.log10(
            (np.dot(s_target, s_target) + 1e-12) / 
            (np.dot(e_residual, e_residual) + 1e-12)
        )
        return si_snr
    
    def calculate_stoi(self, clean: np.ndarray, noisy: np.ndarray, 
                      fs: int = 16000, extended: bool = False) -> float:
        """
        计算短时客观可懂度 (Short-Time Objective Intelligibility)
        
        Args:
            clean: 干净信号
            noisy: 含噪信号
            fs: 采样率
            extended: 是否使用扩展版本
            
        Returns:
            STOI score (0-1, higher is better)
        """
        # 简化的STOI实现
        # 实际应用中建议使用pystoi库
        
        # 分帧参数
        frame_length = int(0.025 * fs)  # 25ms
        hop_length = int(0.01 * fs)     # 10ms
        
        # 确保信号长度一致
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        # 分帧
        frames_clean = self._frame_signal(clean, frame_length, hop_length)
        frames_noisy = self._frame_signal(noisy, frame_length, hop_length)
        
        # 计算每帧的相关系数
        correlations = []
        for clean_frame, noisy_frame in zip(frames_clean, frames_noisy):
            if np.std(clean_frame) > 1e-6 and np.std(noisy_frame) > 1e-6:
                corr = np.corrcoef(clean_frame, noisy_frame)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        # 返回平均相关系数作为STOI估计
        stoi = np.mean(correlations) if correlations else 0.0
        return max(0.0, min(1.0, stoi))
    
    def _frame_signal(self, signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """信号分帧"""
        frames = []
        for i in range(0, len(signal) - frame_length + 1, hop_length):
            frame = signal[i:i + frame_length]
            frames.append(frame)
        return np.array(frames)
    
    def calculate_spectral_distance(self, clean: np.ndarray, noisy: np.ndarray) -> Dict[str, float]:
        """
        计算频谱距离指标
        
        Args:
            clean: 干净信号
            noisy: 含噪信号
            
        Returns:
            包含多种频谱距离指标的字典
        """
        # 计算功率谱密度
        clean_psd = np.abs(np.fft.fft(clean)) ** 2
        noisy_psd = np.abs(np.fft.fft(noisy)) ** 2
        
        # 归一化
        clean_psd = clean_psd / (np.sum(clean_psd) + 1e-12)
        noisy_psd = noisy_psd / (np.sum(noisy_psd) + 1e-12)
        
        # 计算KL散度
        kl_div = np.sum(clean_psd * np.log((clean_psd + 1e-12) / (noisy_psd + 1e-12)))
        
        # 计算JS散度
        m = 0.5 * (clean_psd + noisy_psd)
        js_div = 0.5 * kl_div + 0.5 * np.sum(noisy_psd * np.log((noisy_psd + 1e-12) / (m + 1e-12)))
        
        # 计算欧几里得距离
        euclidean_dist = np.sqrt(np.sum((clean_psd - noisy_psd) ** 2))
        
        # 计算余弦相似度
        cosine_sim = np.dot(clean_psd, noisy_psd) / (
            np.sqrt(np.dot(clean_psd, clean_psd)) * np.sqrt(np.dot(noisy_psd, noisy_psd)) + 1e-12
        )
        
        return {
            'kl_divergence': kl_div,
            'js_divergence': js_div,
            'euclidean_distance': euclidean_dist,
            'cosine_similarity': cosine_sim
        }
    
    def calculate_all_metrics(self, clean: np.ndarray, noisy: np.ndarray) -> Dict[str, float]:
        """
        计算所有音频质量指标
        
        Args:
            clean: 干净信号
            noisy: 含噪信号
            
        Returns:
            包含所有指标的字典
        """
        # 确保信号长度一致
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        metrics = {}
        
        # 基本SNR指标
        metrics['snr_db'] = self.calculate_snr(clean, noisy)
        metrics['si_snr_db'] = self.calculate_si_snr(clean, noisy)
        
        # 可懂度指标
        metrics['stoi'] = self.calculate_stoi(clean, noisy, self.sr)
        
        # 频谱距离指标
        spectral_metrics = self.calculate_spectral_distance(clean, noisy)
        metrics.update(spectral_metrics)
        
        # 时域指标
        metrics['mse'] = np.mean((clean - noisy) ** 2)
        metrics['mae'] = np.mean(np.abs(clean - noisy))
        
        # 幅度指标
        clean_rms = np.sqrt(np.mean(clean ** 2))
        noisy_rms = np.sqrt(np.mean(noisy ** 2))
        metrics['rms_ratio'] = clean_rms / (noisy_rms + 1e-12)
        
        return metrics
    
    def evaluate_recovery_quality(self, original: np.ndarray, recovered: np.ndarray) -> Dict[str, float]:
        """
        评估恢复质量
        
        Args:
            original: 原始干净信号
            recovered: 恢复信号
            
        Returns:
            恢复质量评估指标
        """
        metrics = self.calculate_all_metrics(original, recovered)
        
        # 添加恢复特有的指标
        noise_reduction = -metrics['snr_db']  # 负SNR表示噪声减少
        metrics['noise_reduction_db'] = noise_reduction
        
        # 信号保持度（余弦相似度）
        metrics['signal_preservation'] = metrics['cosine_similarity']
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "音频质量指标"):
        """打印格式化的指标"""
        print(f"\n📊 {title}")
        print("=" * 50)
        
        # SNR指标
        print(f"信噪比 (SNR):           {metrics.get('snr_db', 0):.2f} dB")
        print(f"尺度不变SNR (SI-SNR):   {metrics.get('si_snr_db', 0):.2f} dB")
        
        # 可懂度指标
        print(f"短时可懂度 (STOI):      {metrics.get('stoi', 0):.3f}")
        
        # 频谱指标
        print(f"余弦相似度:             {metrics.get('cosine_similarity', 0):.3f}")
        print(f"KL散度:                 {metrics.get('kl_divergence', 0):.3f}")
        
        # 时域指标
        print(f"均方误差 (MSE):         {metrics.get('mse', 0):.6f}")
        print(f"平均绝对误差 (MAE):     {metrics.get('mae', 0):.6f}")
        
        # 特殊指标
        if 'noise_reduction_db' in metrics:
            print(f"噪声减少:               {metrics['noise_reduction_db']:.2f} dB")
        if 'signal_preservation' in metrics:
            print(f"信号保持度:             {metrics['signal_preservation']:.3f}")


def demo_metrics():
    """演示音频指标计算"""
    print("🎵 音频质量评估演示")
    print("=" * 40)
    
    # 创建测试信号
    fs = 16000
    t = np.linspace(0, 2, fs * 2)
    
    # 生成测试信号
    clean = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    noise = 0.1 * np.random.randn(len(clean))
    noisy = clean + noise
    
    # 计算指标
    metrics_calc = AudioMetrics(fs)
    metrics = metrics_calc.calculate_all_metrics(clean, noisy)
    
    # 打印结果
    metrics_calc.print_metrics(metrics, "测试信号质量指标")
    
    print(f"\n💡 指标说明:")
    print(f"- SNR > 10dB: 质量良好")
    print(f"- STOI > 0.7: 可懂度良好")
    print(f"- 余弦相似度 > 0.9: 信号相似度很高")


if __name__ == "__main__":
    demo_metrics()
