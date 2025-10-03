# Audio Privacy Protection System

An implementation of sound masking for smartphone audio privacy based on the paper "Exploiting Sound Masking for Audio Privacy in Smartphones".

## Quick Start

Process all input files:
```bash
python audio_privacy_system.py
```

Process specific file:
```bash
python audio_privacy_system.py --input dataset/input/your_file.wav
```

Use original masking type:
```bash
python audio_privacy_system.py --mask-type voice_like
```

## Installation

Install required dependencies:
```bash
pip install numpy soundfile scipy
```

Optional packages for advanced audio metrics:
```bash
pip install librosa pystoi pesq
```

## Core Features

- **Audio Masking**: Apply mask noise to clean speech (similar to encryption)
- **Signal Mixing**: Generate mixed signals that sound incomprehensible to eavesdroppers
- **Authorized Recovery**: Authorized users can recover original speech using known parameters
- **Privacy Protection**: Unauthorized listeners only hear muffled mixed signals

## File Structure

```
Sound-Masking/
├── audio_privacy_system.py    # Main system implementation
├── audio_metrics.py           # Audio quality evaluation module
├── dataset/                  # Dataset directory
│   ├── input/               # Input audio files
│   └── output/              # Output result files
```

## Usage Examples

### Single File Processing
```bash
# Process with multi-tone masking (default)
python audio_privacy_system.py --input dataset/input/file.wav

# Adjust masking intensity
python audio_privacy_system.py --input dataset/input/file.wav --snr -5.0
```

### Batch Processing
```python
from audio_privacy_system import AudioPrivacySystem

system = AudioPrivacySystem()
clean_files = ['file1.wav', 'file2.wav', 'file3.wav']
results = system.batch_process(clean_files)

for result in results:
    if result:
        print(f"File: {result['input_file']}")
        print(f"SNR improvement: {result['metrics']['improvement_db']:.2f} dB")
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Sampling rate (Hz) |
| `target_snr_db` | 0.0 | Target SNR (dB), lower values = stronger masking |
| `filter_order` | 128 | LMS filter order |
| `learning_rate` | 0.01 | LMS learning rate |

### Masking Types
- `multi_tone`: Multi-tone masking, sounds like multiple speakers (default)
- `voice_like`: Original voice-like masking, good for concept demonstration

### Adjusting Masking Strength
```python
# Strong masking (SNR = -5dB)
system = AudioPrivacySystem(target_snr_db=-5.0)

# Medium masking (SNR = 0dB) 
system = AudioPrivacySystem(target_snr_db=0.0)

# Weak masking (SNR = 5dB)
system = AudioPrivacySystem(target_snr_db=5.0)
```

## Performance Metrics

- **SNR Improvement**: Typically 5-15dB
- **STOI**: Usually >0.8 after recovery
- **Cosine Similarity**: Signal similarity measure (0-1, higher is better)

Expected results:
- **Authorized users**: Can clearly hear recovered speech
- **Unauthorized users**: Only hear incomprehensible mixed signals

## Troubleshooting

### Poor Recovery Quality
1. Increase LMS filter order: `filter_order=256`
2. Adjust learning rate: `learning_rate=0.005`
3. Ensure mask signal quality is good
4. Check signal length (recommended >1 second)

### Weak Masking Effect
Lower target SNR:
```python
system = AudioPrivacySystem(target_snr_db=-10.0)
```

### Unsupported Audio Format
Install soundfile:
```bash
pip install soundfile
```

Or convert format:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Technical Details

### Mask Generation
1. Generate white noise
2. Bandpass filter to speech frequency range (200-4000Hz)
3. Add syllabic modulation (simulate speech energy changes)
4. Normalize processing

### LMS Recovery Algorithm
1. Use known mask signal as reference
2. Adaptively learn transmission characteristics of mixed signal
3. Estimate and remove mask components
4. Recover original clean speech

## License

This project is open source under the MIT License.
