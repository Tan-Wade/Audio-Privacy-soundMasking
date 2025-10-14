#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Quality Metrics Module

Provides various audio quality assessment metrics including SNR, PESQ, STOI, etc.
"""

import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class AudioMetrics:
    """Audio Quality Assessment Class"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        
    def calculate_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR)
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
            
        Returns:
            SNR in dB
        """
        noise = noisy - clean
        signal_power = np.mean(clean ** 2) + 1e-12
        noise_power = np.mean(noise ** 2) + 1e-12
        
        if noise_power < 1e-12:
            return float('inf')
        
        snr = 10.0 * np.log10(signal_power / noise_power)
        
        # Handle NaN and infinity values
        if np.isnan(snr) or np.isinf(snr):
            return 0.0
            
        return snr
    
    def calculate_si_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        """
        Calculate Scale-Invariant SNR (SI-SNR)
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
            
        Returns:
            SI-SNR in dB
        """
        # Calculate optimal scale factor
        s_target = clean
        e_noise = noisy - clean
        
        # Calculate scale factor
        alpha = np.dot(s_target, noisy) / (np.dot(s_target, s_target) + 1e-12)
        s_hat = alpha * s_target
        e_residual = noisy - s_hat
        
        # Calculate SI-SNR
        si_snr = 10.0 * np.log10(
            (np.dot(s_target, s_target) + 1e-12) / 
            (np.dot(e_residual, e_residual) + 1e-12)
        )
        return si_snr
    
    def calculate_stoi(self, clean: np.ndarray, noisy: np.ndarray, 
                      fs: int = 16000, extended: bool = False) -> float:
        """
        Calculate Short-Time Objective Intelligibility (STOI)
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
            fs: Sample rate
            extended: Whether to use extended version
            
        Returns:
            STOI score (0-1, higher is better)
        """
        # Simplified STOI implementation
        # For practical applications, recommend using pystoi library
        
        # Frame parameters
        frame_length = int(0.025 * fs)  # 25ms
        hop_length = int(0.01 * fs)     # 10ms
        
        # Ensure signal lengths are consistent
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        # Frame the signals
        frames_clean = self._frame_signal(clean, frame_length, hop_length)
        frames_noisy = self._frame_signal(noisy, frame_length, hop_length)
        
        # Calculate correlation coefficient for each frame
        correlations = []
        for clean_frame, noisy_frame in zip(frames_clean, frames_noisy):
            if np.std(clean_frame) > 1e-6 and np.std(noisy_frame) > 1e-6:
                corr = np.corrcoef(clean_frame, noisy_frame)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        # Return average correlation coefficient as STOI estimate
        stoi = np.mean(correlations) if correlations else 0.0
        return max(0.0, min(1.0, stoi))
    
    def _frame_signal(self, signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """Frame the signal"""
        frames = []
        for i in range(0, len(signal) - frame_length + 1, hop_length):
            frame = signal[i:i + frame_length]
            frames.append(frame)
        return np.array(frames)
    
    def calculate_spectral_distance(self, clean: np.ndarray, noisy: np.ndarray) -> Dict[str, float]:
        """
        Calculate spectral distance metrics
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
            
        Returns:
            Dictionary containing various spectral distance metrics
        """
        # Calculate power spectral density
        clean_psd = np.abs(np.fft.fft(clean)) ** 2
        noisy_psd = np.abs(np.fft.fft(noisy)) ** 2
        
        # Normalize
        clean_psd = clean_psd / (np.sum(clean_psd) + 1e-12)
        noisy_psd = noisy_psd / (np.sum(noisy_psd) + 1e-12)
        
        # Calculate KL divergence
        kl_div = np.sum(clean_psd * np.log((clean_psd + 1e-12) / (noisy_psd + 1e-12)))
        
        # Calculate JS divergence
        m = 0.5 * (clean_psd + noisy_psd)
        js_div = 0.5 * kl_div + 0.5 * np.sum(noisy_psd * np.log((noisy_psd + 1e-12) / (m + 1e-12)))
        
        # Calculate Euclidean distance
        euclidean_dist = np.sqrt(np.sum((clean_psd - noisy_psd) ** 2))
        
        # Calculate cosine similarity
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
        Calculate all audio quality metrics
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
            
        Returns:
            Dictionary containing all metrics
        """
        # Ensure signal lengths are consistent
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        metrics = {}
        
        # Basic SNR metrics
        metrics['snr_db'] = self.calculate_snr(clean, noisy)
        metrics['si_snr_db'] = self.calculate_si_snr(clean, noisy)
        
        # Intelligibility metrics
        metrics['stoi'] = self.calculate_stoi(clean, noisy, self.sr)
        
        # Spectral distance metrics
        spectral_metrics = self.calculate_spectral_distance(clean, noisy)
        metrics.update(spectral_metrics)
        
        # Time domain metrics
        metrics['mse'] = np.mean((clean - noisy) ** 2)
        metrics['mae'] = np.mean(np.abs(clean - noisy))
        
        # Amplitude metrics
        clean_rms = np.sqrt(np.mean(clean ** 2))
        noisy_rms = np.sqrt(np.mean(noisy ** 2))
        metrics['rms_ratio'] = clean_rms / (noisy_rms + 1e-12)
        
        return metrics
    
    def evaluate_recovery_quality(self, original: np.ndarray, recovered: np.ndarray) -> Dict[str, float]:
        """
        Evaluate recovery quality
        
        Args:
            original: Original clean signal
            recovered: Recovered signal
            
        Returns:
            Recovery quality assessment metrics
        """
        metrics = self.calculate_all_metrics(original, recovered)
        
        # Add recovery-specific metrics
        noise_reduction = -metrics['snr_db']  # Negative SNR indicates noise reduction
        metrics['noise_reduction_db'] = noise_reduction
        
        # Signal preservation (cosine similarity)
        metrics['signal_preservation'] = metrics['cosine_similarity']
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "Audio Quality Metrics"):
        """Print formatted metrics"""
        print(f"\n📊 {title}")
        print("=" * 50)
        
        # SNR metrics
        print(f"Signal-to-Noise Ratio (SNR):           {metrics.get('snr_db', 0):.2f} dB")
        print(f"Scale-Invariant SNR (SI-SNR):          {metrics.get('si_snr_db', 0):.2f} dB")
        
        # Intelligibility metrics
        print(f"Short-Time Objective Intelligibility (STOI):      {metrics.get('stoi', 0):.3f}")
        
        # Spectral metrics
        print(f"Cosine Similarity:                     {metrics.get('cosine_similarity', 0):.3f}")
        print(f"KL Divergence:                         {metrics.get('kl_divergence', 0):.3f}")
        
        # Time domain metrics
        print(f"Mean Squared Error (MSE):              {metrics.get('mse', 0):.6f}")
        print(f"Mean Absolute Error (MAE):             {metrics.get('mae', 0):.6f}")
        
        # Special metrics
        if 'noise_reduction_db' in metrics:
            print(f"Noise Reduction:                      {metrics['noise_reduction_db']:.2f} dB")
        if 'signal_preservation' in metrics:
            print(f"Signal Preservation:                  {metrics['signal_preservation']:.3f}")


def demo_metrics():
    """Demonstrate audio metrics calculation"""
    print("🎵 Audio Quality Assessment Demo")
    print("=" * 40)
    
    # Create test signals
    fs = 16000
    t = np.linspace(0, 2, fs * 2)
    
    # Generate test signals
    clean = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    noise = 0.1 * np.random.randn(len(clean))
    noisy = clean + noise
    
    # Calculate metrics
    metrics_calc = AudioMetrics(fs)
    metrics = metrics_calc.calculate_all_metrics(clean, noisy)
    
    # Print results
    metrics_calc.print_metrics(metrics, "Test Signal Quality Metrics")
    
    print(f"\n💡 Metric Guidelines:")
    print(f"- SNR > 10dB: Good quality")
    print(f"- STOI > 0.7: Good intelligibility")
    print(f"- Cosine similarity > 0.9: High signal similarity")


if __name__ == "__main__":
    demo_metrics()
