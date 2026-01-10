"""
RealTime Youtube Transcription
Capture system audio and transcribe with whisper.cpp
"""

import sys
import threading
import queue
import time
import numpy as np
from scipy import signal
from datetime import datetime
from pathlib import Path

# Configuration
MODEL_PATH = "models/ggml-large-v3-turbo-q5_0.bin"  # Path to the whisper.cpp template
LANGUAGE = "en"      # Video Language (fr, en, etc.)
SAMPLE_RATE = 16000
CHUNK_DURATION = 5   # Seconds of audio before transcription


def find_loopback_device():
    """Find loopback (system audio) on Windows."""
    import sounddevice as sd

    devices = sd.query_devices()
    loopback_device = None

    print("\n=== Available audio devices ===")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            is_loopback = 'loopback' in device['name'].lower() or 'stereo mix' in device['name'].lower() or 'wasapi' in device['name'].lower()
            marker = " <-- LOOPBACK" if is_loopback else ""
            print(f"  [{i}] {device['name']}{marker}")
            if is_loopback and loopback_device is None:
                loopback_device = i

    return loopback_device


def main():
    print("=" * 60)
    print("   REAL-TIME YOUTUBE TRANSCRIPT")
    print("=" * 60)

    # Importing libraries (with progress messages)
    print("\n[1/3] Loading librairies...")

    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: Sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print("[2/3] Loading whisper.cpp...")

    try:
        from pywhispercpp.model import Model
    except ImportError:
        print("ERROR: pywhispercpp not installed. Run: pip install pywhispercpp")
        sys.exit(1)

    # Verify that the model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("\nDownload the model with:")
        print(f"  curl -L -o {MODEL_PATH} https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin")
        print("\nOr for a smaller/faster model:")
        print(f"  curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin")
        sys.exit(1)

    # Load the whisper.cpp model
    print(f"[3/3] Loading model '{MODEL_PATH}'...")
    model = Model(str(model_path), n_threads=8)
    print("Model loaded!")

    # Find the loopback device
    loopback_idx = find_loopback_device()

    print("\n" + "=" * 60)

    if loopback_idx is not None:
        print(f"Loopback device detected: [{loopback_idx}]")
        use_default = input("Use this device? (O/n): ").strip().lower()
        if use_default == 'n':
            loopback_idx = None

    if loopback_idx is None:
        try:
            loopback_idx = int(input("Enter the number of the device to use: "))
        except ValueError:
            print("Invalid number!")
            sys.exit(1)

    # Prepare the backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"transcription_{timestamp}.txt")

    print(f"\n=== START TRANSCRIPTION ===")
    print(f"Output file: {output_file}")
    print(f"Language: {LANGUAGE or 'auto-détection'}")
    print(f"Press Ctrl+C to stop\n")
    print("-" * 60)

    # Queue for storing audio
    audio_queue = queue.Queue()
    transcription_text = []
    start_time = time.time()

    # Callback to capture audio
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Audio status: {status}]", file=sys.stderr)
        audio_queue.put(indata.copy())

    # Transcription thread
    stop_event = threading.Event()
    actual_sample_rate = [16000]  # Will be updated after the stream opens

    def transcription_worker():
        audio_buffer = np.array([], dtype=np.float32)
        samples_needed = int(SAMPLE_RATE * CHUNK_DURATION)

        while not stop_event.is_set():
            try:
                # Retrieve audio from the queue
                chunk = audio_queue.get(timeout=0.5)
                # Convert to mono if needed
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)
                audio_buffer = np.concatenate([audio_buffer, chunk.flatten()])

                # Calculate how many samples are needed based on the current sample rate
                sr = actual_sample_rate[0]
                samples_at_current_sr = int(sr * CHUNK_DURATION)

                # Transcribe when you have enough audio
                if len(audio_buffer) >= samples_at_current_sr:
                    # Extract the chunk
                    audio_data = audio_buffer[:samples_at_current_sr].astype(np.float32)

                    # Resample to 16000Hz if necessary
                    if sr != SAMPLE_RATE:
                        audio_data = signal.resample_poly(audio_data, SAMPLE_RATE, sr)

                    # Normalize if necessary (sounddevice already returns float32 between -1 and 1)
                    max_val = np.abs(audio_data).max()
                    if max_val > 1.0:
                        audio_data = audio_data / max_val

                    # Check if the audio contains sound (no silence)
                    if np.abs(audio_data).mean() > 0.001:
                        # Transcribe with whisper.cpp
                        segments = model.transcribe(audio_data, language=LANGUAGE)

                        for segment in segments:
                            text = segment.text.strip()
                            if text:
                                # Timestamp = time elapsed since the beginning
                                elapsed = time.time() - start_time
                                hours = int(elapsed // 3600)
                                minutes = int((elapsed % 3600) // 60)
                                seconds = int(elapsed % 60)
                                timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                                line = f"[{timestamp_str}] {text}"
                                print(line)
                                transcription_text.append(line)

                                # Save as you go
                                with open(output_file, "a", encoding="utf-8") as f:
                                    f.write(line + "\n")

                    # Keep a small overlap so that words are not cut off (1 second).
                    audio_buffer = audio_buffer[samples_at_current_sr - sr:]

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Erreur transcription: {e}]", file=sys.stderr)

    # Start audio capture
    try:
        device_info = sd.query_devices(loopback_idx)
        device_sr = int(device_info['default_samplerate'])
        channels = min(2, device_info['max_input_channels'])

        # Adjust the sample rate if necessary
        actual_sr = device_sr if device_sr in [16000, 44100, 48000] else 44100
        actual_sample_rate[0] = actual_sr  # Update for the worker

        with sd.InputStream(
            device=loopback_idx,
            samplerate=actual_sr,
            channels=channels,
            dtype=np.float32,
            callback=audio_callback,
            blocksize=int(actual_sr * 0.5)  # 500ms blocks
        ):
            # If the sample rate is different from 16000, we will have to resample.
            if actual_sr != SAMPLE_RATE:
                print(f"[Note: Resampling de {actual_sr}Hz vers {SAMPLE_RATE}Hz]")

            # Start the transcription thread
            transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
            transcription_thread.start()

            # Wait Ctrl+C
            print("Listening... (Ctrl+C to stop)")
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("End of transcription...")
        stop_event.set()

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nIf you do not have a loopback device, you must:")
        print("1. Enable ‘Stereo Mix’ in Windows sound settings")
        print("2. Or install a virtual audio cable (VB-Cable)")
        sys.exit(1)

    # Final summary
    print(f"\nTranscript saved in: {output_file}")
    print(f"Number of segments: {len(transcription_text)}")

    if transcription_text:
        print("\nWould you like to generate a summary? (o/N): ", end="")
        try:
            response = input().strip().lower()
            if response == 'o':
                generate_summary(output_file)
        except:
            pass


def generate_summary(transcript_file):
    """Generates a summary of the transcript."""
    print("\n=== SUMMARY GENERATION ===")

    # Read the transcript
    with open(transcript_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    if not full_text.strip():
        print("Transcript empty, no summary possible.")
        return

    # Extract only the text (without timestamps)
    lines = full_text.strip().split("\n")
    text_only = " ".join(line.split("] ", 1)[1] if "] " in line else line for line in lines)

    print(f"\nTotal text: {len(text_only)} characters")
    print("\nTo generate a summary, you can:")
    print("1. Copy the contents of the file into ChatGPT/Claude")
    print("2. Use a local summary script (requires ollama or similar)")

    # Save a version without timestamps to facilitate summarization
    summary_input_file = transcript_file.with_suffix('.clean.txt')
    with open(summary_input_file, "w", encoding="utf-8") as f:
        f.write(text_only)

    print(f"\nSaved version without timestamps: {summary_input_file}")


if __name__ == "__main__":
    main()
