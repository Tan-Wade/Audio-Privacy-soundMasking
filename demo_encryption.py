#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合加密演示脚本
Hybrid Encryption Demo Script

演示如何使用混合加密保护音频隐私系统的掩蔽参数
"""

import os
import sys
from pathlib import Path
from audio_privacy_system import AudioPrivacySystem

def main():
    print("=" * 60)
    print("音频隐私保护系统 - 混合加密演示")
    print("Audio Privacy Protection System - Hybrid Encryption Demo")
    print("=" * 60)
    print()
    
    # 步骤1: 生成接收方密钥对
    print("📋 步骤 1: 生成接收方的RSA密钥对")
    print("-" * 60)
    
    system = AudioPrivacySystem(enable_encryption=True)
    
    receiver_name = "alice"
    keypair = system.generate_keypair_for_receiver(receiver_name)
    
    public_key_path = keypair['public_key']
    private_key_path = keypair['private_key']
    
    print(f"\n✓ 接收方 '{receiver_name}' 的密钥对已生成")
    print(f"  公钥路径: {public_key_path}")
    print(f"  私钥路径: {private_key_path}")
    print()
    
    # 步骤2: 处理音频并使用公钥加密参数
    print("📋 步骤 2: 处理音频并使用公钥加密掩蔽参数")
    print("-" * 60)
    
    # 查找输入音频文件
    input_files = list(system.input_dir.glob("*.wav"))
    
    if not input_files:
        print("❌ 错误: dataset/input/ 目录中没有找到音频文件")
        print("请将音频文件放入 dataset/input/ 目录")
        return
    
    # 使用第一个音频文件进行演示
    input_audio = str(input_files[0])
    print(f"使用音频文件: {Path(input_audio).name}")
    print()
    
    # 处理音频（发送方视角）
    result = system.process_audio_pair(
        clean_path=input_audio,
        output_prefix="demo_encrypted",
        mask_type="multi_tone",
        receiver_public_key=public_key_path
    )
    
    print(f"\n✓ 音频处理完成")
    print(f"  混合音频: {result['output_files']['mixed']}")
    print(f"  加密参数: {result['output_files']['mask_params']}")
    print()
    
    # 步骤3: 授权方使用私钥解密并恢复音频
    print("📋 步骤 3: 授权方使用私钥解密参数并恢复音频")
    print("-" * 60)
    
    mixed_audio_path = result['output_files']['mixed']
    params_file_path = result['output_files']['mask_params']
    recovered_output_path = str(Path(system.output_dir) / "demo_encrypted_recovered_decrypted.wav")
    
    # 授权恢复（接收方视角）
    recovery_result = system.authorized_recovery(
        mixed_audio_path=mixed_audio_path,
        params_path=params_file_path,
        output_path=recovered_output_path,
        receiver_private_key=private_key_path
    )
    
    print(f"\n✓ 授权恢复完成")
    print(f"  恢复音频: {recovery_result['recovered_audio']}")
    print()
    
    # 步骤4: 演示安全性
    print("📋 步骤 4: 安全性演示")
    print("-" * 60)
    
    print("\n🔒 安全特性:")
    print("  1. 掩蔽参数使用AES-256-GCM加密（对称加密）")
    print("  2. AES密钥使用RSA-2048-OAEP加密（非对称加密）")
    print("  3. 只有拥有私钥的授权接收方才能解密参数")
    print("  4. 未授权第三方即使截获参数文件也无法解密")
    print()
    
    print("📊 数据流:")
    print("  发送方 -> 使用公钥加密参数 -> 加密的JSON文件")
    print("  接收方 -> 使用私钥解密参数 -> 恢复干净音频")
    print("  窃听者 -> 无法解密参数 -> 只能听到混合音频")
    print()
    
    # 步骤5: 对比测试（无授权情况）
    print("📋 步骤 5: 无授权场景演示")
    print("-" * 60)
    
    print("\n❌ 如果没有私钥，无法解密参数:")
    print("  - 窃听者截获混合音频和加密参数文件")
    print("  - 由于没有私钥，无法解密参数")
    print("  - 无法重新生成掩蔽信号")
    print("  - 无法从混合音频中恢复干净语音")
    print("  - 只能听到不清晰的混合音频")
    print()
    
    print("=" * 60)
    print("✓ 演示完成！")
    print("=" * 60)
    print()
    
    print("📁 生成的文件:")
    print(f"  - 公钥: {public_key_path}")
    print(f"  - 私钥: {private_key_path}")
    print(f"  - 混合音频: {mixed_audio_path}")
    print(f"  - 加密参数: {params_file_path}")
    print(f"  - 恢复音频: {recovered_output_path}")
    print()
    
    print("💡 提示:")
    print("  - 您可以播放混合音频，听起来应该是模糊不清的")
    print("  - 您可以播放恢复音频，听起来应该接近原始音频")
    print("  - 您可以查看加密参数文件，内容应该是加密的")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

