# 音频隐私保护系统 (Audio Privacy Protection System)

基于声音掩蔽技术的智能手机音频隐私保护实现

## 🚀 快速运行

**一键运行（处理所有输入文件）**：
```bash
python audio_privacy_system.py
```

**指定单个文件**：
```bash
python audio_privacy_system.py --input dataset/input/your_file.wav
```

**使用原始掩蔽类型**：
```bash
python audio_privacy_system.py --mask-type voice_like
```

---

## 📖 项目简介

本项目复现了论文 "Exploiting Sound Masking for Audio Privacy in Smartphones" 的核心思想，实现了一个音频隐私保护系统。

### 🎯 核心功能

1. **音频掩蔽**: 对干净语音施加掩蔽噪声（类似"加密"）
2. **混合信号生成**: 生成含混的混合信号（模拟被监听方录到的声音）
3. **授权恢复**: 授权方使用已知参数进行反向恢复
4. **隐私保护**: 非授权方只能听到含混不清的混合信号

### 🔧 技术原理

- **掩蔽噪声生成**: 使用类语音样式的噪声，比白噪声更有效
- **自适应滤波**: 采用LMS算法进行授权恢复
- **信噪比控制**: 可调节掩蔽强度
- **语音优化**: 针对中文语音特征进行参数优化

## 🛠️ 详细使用

### 1. 环境准备

```bash
# 安装依赖
pip install numpy soundfile scipy

# 或者使用conda
conda install numpy scipy
pip install soundfile
```

**依赖包说明**：
- `numpy>=1.20.0` - 数值计算核心库
- `scipy>=1.7.0` - 科学计算库（音频处理）
- `soundfile>=0.10.0` - 音频文件读写

**可选依赖**（用于更高级的音频处理）：
- `librosa>=0.8.0` - 音频特征提取
- `pystoi>=0.3.0` - STOI计算
- `pesq>=0.0.3` - PESQ计算

### 2. 运行演示

```bash
# 快速演示（推荐）
python demo.py

# 完整功能演示
python audio_privacy_system.py

# 查看帮助
python audio_privacy_system.py --help
```

## 📁 文件结构

```
Sound-Masking/
├── audio_privacy_system.py    # 主系统实现
├── audio_metrics.py           # 音频质量评估模块
├── README.md                 # 项目说明文档
├── dataset/                  # 数据集目录
│   ├── input/               # 输入音频文件
│   └── output/              # 输出结果文件
├── 01_clean.wav             # 示例：干净语音
├── 02_mask.wav              # 示例：掩蔽噪声
├── 03_mixed.wav             # 示例：混合信号
├── 04_recovered.wav         # 示例：恢复语音
```

## 🎵 使用场景

### 场景1：使用现有录音

1. 将你的8位数字录音文件放在 `dataset/input/` 目录
2. 运行 `python audio_privacy_system.py` 处理所有文件
3. 查看 `dataset/output/` 目录下的结果文件

**示例**：
```bash
# 处理单个文件（默认使用多音调掩蔽）
python audio_privacy_system.py --input dataset/input/female-voice.m4a

# 批量处理
python audio_privacy_system.py --batch dataset/input/

# 使用原始掩蔽类型
python audio_privacy_system.py --input dataset/input/file.wav --mask-type voice_like

# 调整掩蔽强度
python audio_privacy_system.py --input dataset/input/file.wav --snr -5.0
```

### 场景2：批量处理

```python
from audio_privacy_system import AudioPrivacySystem

system = AudioPrivacySystem()

# 批量处理多个文件
clean_files = ['file1.wav', 'file2.wav', 'file3.wav']
results = system.batch_process(clean_files)

# 查看结果
for result in results:
    if result:
        print(f"文件: {result['input_file']}")
        print(f"SNR改善: {result['metrics']['improvement_db']:.2f} dB")
```

## ⚙️ 参数配置

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_rate` | 16000 | 采样率（Hz） |
| `target_snr_db` | 0.0 | 目标信噪比（dB），越小掩蔽效果越强 |
| `filter_order` | 128 | LMS滤波器阶数 |
| `learning_rate` | 0.01 | LMS学习率 |

### 调整掩蔽强度

```python
# 强掩蔽（SNR = -5dB）
system = AudioPrivacySystem(target_snr_db=-5.0)

# 中等掩蔽（SNR = 0dB）
system = AudioPrivacySystem(target_snr_db=0.0)

# 弱掩蔽（SNR = 5dB）
system = AudioPrivacySystem(target_snr_db=5.0)
```

### 选择掩蔽类型

```python
# 多音调掩蔽（默认，类似多个说话人）
result = system.process_audio_pair(input_file, mask_type="multi_tone")

# 原始类语音掩蔽（适合展示概念）
result = system.process_audio_pair(input_file, mask_type="voice_like")
```

**掩蔽类型说明**：
- `multi_tone`: 多音调掩蔽，类似多个说话人同时说话（默认）
- `voice_like`: 原始类语音掩蔽，适合展示概念

### 调整恢复质量

```python
# 高质量恢复（更多滤波器阶数）
recovered, _ = system.lms_recovery(mixed, mask_ref, filter_order=256)

# 快速恢复（较少滤波器阶数）
recovered, _ = system.lms_recovery(mixed, mask_ref, filter_order=64)
```

## 📊 性能指标

### SNR指标
- **输入SNR**: 混合信号中干净语音与掩蔽噪声的信噪比
- **恢复后SNR**: 恢复信号与原始信号的信噪比
- **SNR改善**: 恢复效果的性能提升

### 可懂度指标
- **STOI**: 短时客观可懂度（0-1，越高越好）
- **余弦相似度**: 信号相似度（0-1，越高越好）

### 预期效果
- **SNR改善**: 通常可达到5-15dB
- **STOI**: 恢复后通常>0.8
- **授权方**: 可以清晰听到恢复后的语音
- **非授权方**: 只能听到含混不清的混合信号

## 🔧 高级用法

### 自定义掩蔽噪声

```python
# 生成特定类型的掩蔽噪声
def custom_mask_generator(system, length):
    # 使用白噪声
    white_noise = np.random.randn(length)
    
    # 或使用特定频率的噪声
    t = np.linspace(0, length/system.sr, length)
    tone_noise = np.sin(2 * np.pi * 1000 * t)
    
    return tone_noise.astype(np.float32)

# 使用自定义掩蔽
system = AudioPrivacySystem()
custom_mask = custom_mask_generator(system, len(clean_signal))
mixed, scaled_mask = system.mix_signals(clean_signal, custom_mask)
```

## 🐛 常见问题

### Q1: 恢复效果不好怎么办？

**A**: 尝试以下方法：
1. 增加LMS滤波器阶数：`filter_order=256`
2. 调整学习率：`learning_rate=0.005`
3. 确保掩蔽信号质量良好
4. 检查信号长度是否足够（建议>1秒）

### Q2: 掩蔽效果不够强怎么办？

**A**: 降低目标SNR：
```python
system = AudioPrivacySystem(target_snr_db=-10.0)  # 强掩蔽
```

### Q3: 处理速度太慢怎么办？

**A**: 优化参数：
```python
# 减少滤波器阶数
recovered, _ = system.lms_recovery(mixed, mask_ref, filter_order=64)

# 使用更快的掩蔽噪声生成
mask = np.random.randn(len(clean))  # 简单白噪声
```

### Q4: 音频文件格式不支持？

**A**: 确保安装soundfile：
```bash
pip install soundfile
```

或者转换音频格式：
```bash
# 使用ffmpeg转换
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 🔬 技术细节

### 掩蔽噪声生成

1. 生成白噪声
2. 带通滤波到语音频带（200-4000Hz）
3. 添加音节式调制（模拟语音能量变化）
4. 归一化处理

### LMS恢复算法

1. 使用已知掩蔽信号作为参考
2. 自适应学习混合信号的传输特性
3. 估计并消除掩蔽成分
4. 恢复原始干净语音

## 🎯 适用场景

- 语音通话隐私保护
- 智能设备音频隐私
- 敏感信息传输
- 音频数据安全

## 🔒 安全考虑

1. **掩蔽参数保密**: 确保掩蔽噪声的参数只有授权方知道
2. **传输安全**: 在安全通道中传输掩蔽参数
3. **密钥管理**: 考虑使用密钥管理系统管理掩蔽参数
4. **定期更新**: 定期更换掩蔽参数以提高安全性

## 📚 参考文献

- 原始论文：Exploiting Sound Masking for Audio Privacy in Smartphones
- LMS算法：Least Mean Squares Adaptive Filtering
- 语音处理：Speech Signal Processing
- 音频质量评估：Audio Quality Metrics

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目基于MIT许可证开源。
