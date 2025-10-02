#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Privacy Protection System - Quick Demo Script
音频隐私保护系统 - 快速演示脚本
Quick demonstration of audio masking and recovery functionality
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from audio_privacy_system import AudioPrivacySystem

def quick_demo():
    """Quick demo functionality 快速演示功能"""
    print("🎵 Audio Privacy Protection System - Quick Demo")
    print("=" * 50)
    
    # Initialize system 初始化系统
    system = AudioPrivacySystem(sample_rate=16000, target_snr_db=0.0)
    
    # Check files in dataset/input directory 检查dataset/input目录中的文件
    input_files = []
    for ext in ['*.wav', '*.m4a', '*.mp3', '*.flac']:
        input_files.extend(system.input_dir.glob(ext))
    
    if input_files:
        print(f"✓ Found {len(input_files)} input audio files")
        print("Processing first file for demo...")
        
        first_file = input_files[0]
        print(f"✓ Processing file: {first_file.name}")
        
        # Process audio 处理音频
        result = system.process_audio_pair(str(first_file), mask_type="multi_tone")
        
        print(f"\n📊 Processing Results:")
        print(f"   File: {first_file.name}")
        print(f"   Input SNR: {result['metrics']['input_snr_db']:.2f} dB")
        print(f"   Recovery SNR: {result['metrics']['output_snr_db']:.2f} dB")
        print(f"   SNR improvement: {result['metrics']['improvement_db']:.2f} dB")
        
        print(f"\n📁 Output Files:")
        for key, path in result['output_files'].items():
            print(f"   {key}: {path}")
            
        if len(input_files) > 1:
            print(f"\n💡 Tip: {len(input_files)-1} more files to process")
            print("Use the following command for batch processing:")
            print("   python audio_privacy_system.py --batch dataset/input")
            
    else:
        print("⚠️  No input audio files found")
        print("Please place your audio files in dataset/input/ directory and run again.")
        print("\n💡 Usage instructions:")
        print("1. Record speech containing 8-digit numbers")
        print("2. Place files in dataset/input/ directory")
        print("3. Run this script again")
        print("\nSupported audio formats: .wav, .m4a, .mp3, .flac")
    
    print(f"\n🎯 System Description:")
    print("1. Clean speech: Original clear speech signal")
    print("2. Masking noise: Voice-like noise used to mask original speech")
    print("3. Mixed signal: What eavesdroppers would record (unclear)")
    print("4. Recovered signal: Clear speech recovered by authorized parties")
    print("\n💡 Core principle: Only authorized parties know the exact masking parameters to reverse recovery")

if __name__ == "__main__":
    quick_demo()
