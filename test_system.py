#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频隐私保护系统测试脚本
Test Script for Audio Privacy Protection System
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from audio_privacy_system import AudioPrivacySystem
from audio_metrics import AudioMetrics

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试基本功能")
    print("=" * 40)
    
    # 初始化系统
    system = AudioPrivacySystem(sample_rate=16000, target_snr_db=0.0)
    
    # 生成测试信号
    duration = 2.0
    t = np.linspace(0, duration, int(system.sr * duration), endpoint=False)
    
    # 生成包含多个频率成分的测试信号
    test_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +      # A4音符
        0.3 * np.sin(2 * np.pi * 880 * t) +      # A5音符
        0.2 * np.sin(2 * np.pi * 1320 * t)       # E6音符
    )
    
    # 归一化
    test_signal = test_signal / (np.max(np.abs(test_signal)) + 1e-9)
    
    print(f"✓ 生成测试信号: {len(test_signal)/system.sr:.2f}秒")
    
    # 生成掩蔽噪声
    mask = system.generate_voice_like_mask(len(test_signal))
    print(f"✓ 生成掩蔽噪声")
    
    # 混合信号
    mixed, scaled_mask = system.mix_signals(test_signal, mask)
    print(f"✓ 混合信号")
    
    # LMS恢复
    recovered, filter_taps = system.lms_recovery(mixed, scaled_mask)
    print(f"✓ LMS恢复")
    
    # 计算指标
    snr_input = system.calculate_snr(test_signal, scaled_mask)
    snr_after = system.calculate_snr(test_signal, recovered - test_signal)
    improvement = snr_after - snr_input
    
    print(f"\n📊 测试结果:")
    print(f"输入SNR: {snr_input:.2f} dB")
    print(f"恢复后SNR: {snr_after:.2f} dB")
    print(f"SNR改善: {improvement:.2f} dB")
    
    # 保存测试文件
    test_dir = system.output_dir / "test"
    test_dir.mkdir(exist_ok=True)
    
    system.save_audio(test_dir / "test_clean.wav", test_signal)
    system.save_audio(test_dir / "test_mask.wav", scaled_mask)
    system.save_audio(test_dir / "test_mixed.wav", mixed)
    system.save_audio(test_dir / "test_recovered.wav", recovered)
    
    print(f"✓ 测试文件已保存到: {test_dir}")
    
    return improvement > 5.0  # 期望至少5dB的改善

def test_audio_metrics():
    """测试音频质量评估"""
    print("\n🧪 测试音频质量评估")
    print("=" * 40)
    
    metrics_calc = AudioMetrics(sample_rate=16000)
    
    # 生成测试信号
    duration = 2.0
    t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
    
    clean = np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.randn(len(clean))
    noisy = clean + noise
    
    # 计算所有指标
    metrics = metrics_calc.calculate_all_metrics(clean, noisy)
    
    print("✓ 计算音频质量指标")
    metrics_calc.print_metrics(metrics, "测试音频质量指标")
    
    # 验证指标合理性
    assert metrics['snr_db'] > 10, f"SNR过低: {metrics['snr_db']:.2f} dB"
    assert metrics['stoi'] > 0.8, f"STOI过低: {metrics['stoi']:.3f}"
    assert metrics['cosine_similarity'] > 0.9, f"余弦相似度过低: {metrics['cosine_similarity']:.3f}"
    
    print("✓ 音频质量指标测试通过")
    return True

def test_batch_processing():
    """测试批量处理功能"""
    print("\n🧪 测试批量处理功能")
    print("=" * 40)
    
    system = AudioPrivacySystem()
    
    # 创建测试文件
    test_dir = system.output_dir / "batch_test"
    test_dir.mkdir(exist_ok=True)
    
    # 生成测试信号
    duration = 1.0
    t = np.linspace(0, duration, int(system.sr * duration), endpoint=False)
    
    test_files = []
    for i in range(2):
        # 生成不同频率的测试信号
        freq = 440 + i * 100  # 440Hz, 540Hz
        test_signal = np.sin(2 * np.pi * freq * t)
        test_signal = test_signal / (np.max(np.abs(test_signal)) + 1e-9)
        
        # 保存测试文件
        test_file = test_dir / f"test_{i+1}.wav"
        system.save_audio(test_file, test_signal)
        test_files.append(str(test_file))
    
    print(f"✓ 创建 {len(test_files)} 个测试文件")
    
    # 测试批量处理
    results = system.batch_process(test_files)
    
    print(f"✓ 批量处理 {len(results)} 个文件")
    
    valid_results = [r for r in results if r is not None]
    assert len(valid_results) > 0, "批量处理失败"
    
    print(f"✓ 成功处理 {len(valid_results)} 个文件")
    
    return True

def test_existing_files():
    """测试现有文件处理"""
    print("\n🧪 测试现有文件处理")
    print("=" * 40)
    
    system = AudioPrivacySystem()
    
    # 检查现有文件
    existing_files = []
    for filename in ['01_clean.wav', '02_mask.wav', '03_mixed.wav', '04_recovered.wav']:
        if os.path.exists(filename):
            existing_files.append(filename)
    
    if existing_files:
        print(f"✓ 发现 {len(existing_files)} 个现有文件")
        
        # 尝试处理clean文件
        if '01_clean.wav' in existing_files:
            try:
                result = system.process_audio_pair('01_clean.wav', 'test_existing')
                print(f"✓ 成功处理现有文件")
                print(f"  - 输入SNR: {result['metrics']['input_snr_db']:.2f} dB")
                print(f"  - 恢复后SNR: {result['metrics']['output_snr_db']:.2f} dB")
                print(f"  - SNR改善: {result['metrics']['improvement_db']:.2f} dB")
                return True
            except Exception as e:
                print(f"✗ 处理现有文件失败: {e}")
                return False
    else:
        print("⚠️  未发现现有文件，跳过此测试")
        return True

def run_all_tests():
    """运行所有测试"""
    print("🚀 音频隐私保护系统 - 完整测试")
    print("=" * 60)
    
    tests = [
        ("基本功能测试", test_basic_functionality),
        ("音频质量评估测试", test_audio_metrics),
        ("批量处理测试", test_batch_processing),
        ("现有文件处理测试", test_existing_files),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 {test_name}")
            result = test_func()
            results.append((test_name, result, None))
            print(f"✅ {test_name} - {'通过' if result else '失败'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name} - 失败: {e}")
    
    # 汇总结果
    print(f"\n📋 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {test_name}")
        if error:
            print(f"    错误: {error}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统功能正常")
    else:
        print("⚠️  部分测试失败，请检查系统配置")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
