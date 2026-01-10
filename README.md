# Real-Time YouTube Transcription with Whisper.cpp

This project enables real-time transcription of YouTube videos using your system's audio input. It captures audio through a loopback device and transcribes it.

## Features
- Real-time audio capture from system audio output (loopback)
- Automatic transcription using Whisper.cpp models
- Support for multiple languages
- Timestamped transcription output
- Automatic saving of transcriptions to text files

## Requirements

This project requires Python 3.12 and the following dependencies:

```txt
faster-whisper>=1.0.0
sounddevice>=0.4.6
numpy>=1.24.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Setup Instructions

### 1. Install Required Libraries
```bash
pip install faster-whisper sounddevice numpy
```

### 2. Download Whisper Model
The application requires a whisper.cpp model file. You can download one of the following models:

- **Large model** (recommended for accuracy):
  ```bash
  curl -L -o models/ggml-large-v3-turbo-q5_0.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
  ```

- **Medium model** (better balance of speed/accuracy):
  ```bash
  curl -L -o models/ggml-medium.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
  ```

- **Base model** (fastest, less accurate):
  ```bash
  curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
  ```

### 3. Configure Audio Input

This application requires a loopback audio device to capture system audio. For Windows users, you'll need to:

1. **Enable Stereo Mix** (if available):
   - Open Windows Sound Settings
   - Go to "Recording" tab
   - Right-click on your default audio device and select "Properties"
   - In the "Listen" tab, check "Listen to this device"
   - Look for "Stereo Mix" in the list

2. **Install Virtual Audio Cable** (VB-Cable):
   - Download from [vb-audio.com](https://www.vb-audio.com/Cable/)
   - Install and configure it as your loopback device
   - Set it as default recording device

## Usage

Run the transcription script:
```bash
python transcribe.py
```

The program will:
1. List available audio devices
2. Prompt you to select a loopback device
3. Start capturing audio from the selected device
4. Transcribe audio in real-time with timestamps
5. Save output to a timestamped text file

Press `Ctrl+C` to stop transcription.

## Output Format

Transcriptions are saved in text files with the following format:
```
[00:01:23] This is the transcribed text from the audio.
[00:01:45] Another sentence from the video.
```

## Configuration Options

You can modify these settings at the top of `transcribe.py`:

- `MODEL_PATH`: Path to your whisper.cpp model file
- `LANGUAGE`: Language of the video (default: "en")
- `SAMPLE_RATE`: Audio sample rate (default: 16000)
- `CHUNK_DURATION`: Audio chunk duration in seconds before transcription (default: 5)

## Troubleshooting

### No Audio Devices Found
If no loopback devices are detected:
1. Ensure your audio drivers are up to date
2. Verify that Stereo Mix is enabled or VB-Cable is properly installed
3. Check that the selected device has input capabilities

### Audio Quality Issues
- Make sure system volume is at a reasonable level
- Use high-quality audio sources for better transcription accuracy
- Ensure no other applications are using the audio device simultaneously

## Notes

- This application works best with clear, well-recorded audio
- Transcription quality depends heavily on the audio source quality
- The application automatically resamples audio to 16000Hz for optimal performance with Whisper models

## Dependencies

- Python 3.12+
- `sounddevice` - Audio input/output library
- `numpy` - Numerical computing library
- `faster-whisper` - Fast Whisper transcription library
- `scipy` - Scientific computing (included in requirements.txt)


*Note: This application requires system audio capture capabilities. On Windows, this typically means having either Stereo Mix enabled or a virtual audio cable installed.*