#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频隐私保护系统 - 快速演示脚本
快速演示音频掩蔽和恢复功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from audio_privacy_system import AudioPrivacySystem

def quick_demo():
    """快速演示功能"""
    print("🎵 音频隐私保护系统 - 快速演示")
    print("=" * 50)
    
    # 初始化系统
    system = AudioPrivacySystem(sample_rate=16000, target_snr_db=0.0)
    
    # 检查现有文件
    clean_file = "01_clean.wav"
    
    if os.path.exists(clean_file):
        print(f"✓ 发现现有音频文件: {clean_file}")
        print("开始处理...")
        
        # 处理音频
        result = system.process_audio_pair(clean_file, "demo")
        
        print(f"\n📊 处理结果:")
        print(f"   输入SNR: {result['metrics']['input_snr_db']:.2f} dB")
        print(f"   恢复后SNR: {result['metrics']['output_snr_db']:.2f} dB")
        print(f"   SNR改善: {result['metrics']['improvement_db']:.2f} dB")
        
        print(f"\n📁 输出文件:")
        for key, path in result['output_files'].items():
            print(f"   {key}: {path}")
            
    else:
        print("⚠️  未发现现有音频文件")
        print("请将你的8位数字录音文件命名为 '01_clean.wav' 并放在项目根目录，然后重新运行。")
        print("\n💡 使用说明:")
        print("1. 录制一段包含8位数字的语音")
        print("2. 将文件重命名为 '01_clean.wav'")
        print("3. 放在项目根目录")
        print("4. 重新运行此脚本")
    
    print(f"\n🎯 系统说明:")
    print("1. 干净语音: 原始清晰的语音信号")
    print("2. 掩蔽噪声: 类语音样式的噪声，用于掩盖原始语音")
    print("3. 混合信号: 被监听方录到的声音（含混不清）")
    print("4. 恢复信号: 授权方使用已知参数恢复的清晰语音")
    print("\n💡 核心原理: 只有授权方知道掩蔽噪声的精确参数，可以反向恢复原始语音")

if __name__ == "__main__":
    quick_demo()
