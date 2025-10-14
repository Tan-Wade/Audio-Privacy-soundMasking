# Audio Privacy Protection System

This is for the CSEC5614 assignment 2 (Activity Code 06_Group 4).

This is an implementation of sound masking for smartphone audio privacy.

## 🚀 Quick Start

Here are the two main ways to run the audio privacy protection system:

### Method 1: One-step Completion (Recommended)
This method runs the main system and automatically handles key generation (if no keys exist) and encryption of all generated parameters.

```bash
# Run the main system and automatically encrypt all parameters
python audio_privacy_system.py --enable-encryption
```

### Method 2: Step-by-step Execution
This method allows you to first process audio and generate unencrypted parameters, then batch encrypt them separately.

```bash
# 1. Pre-process audio (generate unencrypted parameters)
python audio_privacy_system.py

# 2. Batch encrypt all parameter files
python encryption_module.py
```

### Additional Options

Process specific file:
```bash
python audio_privacy_system.py --input dataset/input/your_file.wav
```

Process with encryption:
```bash
python audio_privacy_system.py --enable-encryption --input dataset/input/your_file.wav
```

**Environment Configuration:**

Open `audio_privacy_system.py` and modify the `ENVIRONMENT` variable in the `main()` function:

```python
# In main() function:
ENVIRONMENT = "dev"   # Dev mode: saves all files including mask audio
# or
ENVIRONMENT = "prod"  # Production mode: saves only parameters, not mask audio
```

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy soundfile scipy cryptography
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
- **🆕 Parameter-based Masking**: Transmit lightweight parameters (< 300 bytes) instead of full mask audio files (saves 99.9%)
- **🆕 Environment Modes**: Dev/Prod modes for development and production deployment
- **🆕 Cryptographically Secure**: Uses secure random seeds with timestamp and identifier for tracking
- **🔒 Hybrid Encryption**: RSA-2048 + AES-256-GCM encryption for mask parameters (NEW!)

## 🚀 New: Parameter-based Masking System

Traditional masking systems require transmitting the entire mask audio file (hundreds of KB). Our new parameter-based approach reduces this to just ~240 bytes!

### Key Benefits

- **99.9% size reduction**: 193 KB → 241 bytes
- **800x compression**: Transmit parameters instead of audio
- **Enhanced security**: Each session uses unique cryptographic seed
- **Flexible deployment**: Separate dev/prod modes for different scenarios

### How It Works

Instead of sending the mask audio file, we send lightweight parameters:

```json
{
  "seed": 1234567890,              // Cryptographic random seed
  "length": 96597,                 // Audio length
  "sample_rate": 16000,            // Sample rate
  "scale_factor": 0.3385,          // Scale factor
  "mask_type": "multi_tone",       // Mask type
  "timestamp": 1760411088,         // Unix timestamp (prevents replay attacks)
  "identifier": "uuid-...",        // Unique session ID
  "version": "1.0",                // Protocol version
  "target_snr_db": 0.0            // Target SNR
}
```

The authorized receiver uses these parameters to regenerate the exact same mask noise and recover the speech.

### Usage

**Sender (Alice):**
```python
from audio_privacy_system import AudioPrivacySystem

# Initialize system
system = AudioPrivacySystem()

# Process audio
result = system.process_audio_pair("speech.wav")

# Get parameters (only 241 bytes!)
mask_params = result['mask_params']

# Encrypt and send: mixed_audio + encrypted(mask_params)
```

**Receiver (Bob):**
```python
# Load parameters
mask_params = system.load_mask_params("received_params.json")

# Regenerate mask from parameters
scaled_mask = system.regenerate_mask_from_params(mask_params)

# Load mixed audio and recover
mixed, _ = system.load_audio("received_mixed.wav")
recovered, _ = system.lms_recovery(mixed, scaled_mask)
```

For detailed documentation, see [PARAMETER_BASED_MASKING.md](PARAMETER_BASED_MASKING.md).

## 🔐 Hybrid Encryption (Optional)

Protect mask parameters with RSA-2048 + AES-256-GCM encryption:

```bash
# Method 1: One-step encryption (recommended)
python audio_privacy_system.py --enable-encryption

# Method 2: Manual key generation and encryption
# 1. Generate receiver's keypair
python audio_privacy_system.py --enable-encryption --generate-keypair alice

# 2. Process audio with encryption (auto-generates keys if none exist)
python audio_privacy_system.py --enable-encryption --input dataset/input/voice.wav

# 3. Batch encrypt existing parameter files
python encryption_module.py

# 4. Receiver: Decrypt and recover
python audio_privacy_system.py --enable-encryption --recover \
  --mixed-audio dataset/output/audio/mixed/*_mixed.wav \
  --params-file dataset/output/encryption/params/*_mask_params_encrypted.json \
  --private-key dataset/output/encryption/keys/*_private.pem \
  --output recovered.wav
```

**Security Features**:
- **Automatic key generation**: System generates keys automatically if none exist
- **Dual file output**: Both plain and encrypted parameter files are saved
- **Batch encryption**: Can encrypt all existing parameter files at once
- **Size**: Encrypted params are ~640 bytes (vs ~240 bytes plain)
- **Protection**: Only authorized receivers with private keys can decrypt and recover audio

### Output File Organization

When using encryption, the system generates two versions of parameter files:

```
dataset/output/encryption/params/
├── file_mask_params_multi_tone.json          # Plain JSON (for development)
└── file_mask_params_multi_tone_encrypted.json # Encrypted version (for transmission)
```

- **Plain files**: Human-readable JSON for development and debugging
- **Encrypted files**: Secure versions for actual transmission to authorized parties
- **Automatic detection**: The system automatically detects encrypted vs plain files during recovery

## File Structure

```
Sound-Masking/
├── audio_privacy_system.py           # Main system implementation
├── audio_metrics.py                  # Audio quality evaluation module
├── encryption_module.py              # Hybrid encryption module (RSA+AES)
├── requirements.txt                  # Python dependencies
├── PARAMETER_BASED_MASKING.md        # Detailed documentation for parameter-based masking
├── README.md                         # This file
├── dataset/                          # Dataset directory
│   ├── input/                       # Input audio files
│   └── output/                      # Output result files (organized by type)
│       ├── audio/                   # Audio files
│       │   ├── clean/              # Clean speech files
│       │   ├── mixed/              # Mixed signals (what eavesdroppers hear)
│       │   ├── recovered/          # Recovered speech (authorized party)
│       │   └── masks/              # Mask audio files (dev mode only)
│       └── encryption/             # Encryption related files
│           ├── keys/               # RSA keypairs (if using encryption)
│           │   ├── *_private.pem  # Private keys (keep secret!)
│           │   └── *_public.pem   # Public keys (share with sender)
│           └── params/             # Parameter files
│               ├── *.json         # Plain parameter files
│               └── *_encrypted.json # Encrypted parameter files
```

## Usage Examples

### Single File Processing
```bash
# Process with multi-tone masking (default)
python audio_privacy_system.py --input dataset/input/file.wav

# Adjust masking intensity
python audio_privacy_system.py --input dataset/input/file.wav --snr -5.0

# Process with encryption (auto-generates keys if none exist)
python audio_privacy_system.py \
  --enable-encryption \
  --input dataset/input/file.wav
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
| `ENVIRONMENT` | "dev" | Environment mode: "dev" (saves all files) or "prod" (parameters only) |
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
