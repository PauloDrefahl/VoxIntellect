import sounddevice as sd
import numpy as np
import wave
import whisper
import ollama
import soundfile as sf
from TTS.api import TTS
import threading

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
SAMPLE_RATE = 44100  # CD-quality
CHANNELS = 1
AUDIO_FILE = "recorded_audio.wav"
DURATION_LIMIT = 60  # Max duration safeguard in seconds 

# ------------------------------------------------------------------------------
# Load the TTS model once
# ------------------------------------------------------------------------------
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
MALE_SPEAKER = "p228"  # Example male voice

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def record_audio():
    """Records audio when user presses Enter to start and Enter again to stop."""
    print("🎤 Press Enter to start recording or type 'q' then Enter to quit.")
    start_input = input().strip().lower()
    if start_input == 'q':
        return False  # Signal to quit

    print("🎙 Recording... Press Enter to stop.")

    frames = []
    recording = True

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status)
        frames.append(indata.copy())

    def wait_for_stop():
        nonlocal recording
        input()
        recording = False

    stop_thread = threading.Thread(target=wait_for_stop)
    stop_thread.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        while recording:
            sd.sleep(100)

    audio_data = np.concatenate(frames, axis=0)

    # Save to WAV
    with wave.open(AUDIO_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    print(f"✅ Audio saved as {AUDIO_FILE}")
    return True

def transcribe_audio():
    print("📝 Transcribing audio with Whisper...")
    model = whisper.load_model("tiny")
    result = model.transcribe(AUDIO_FILE)
    text = result["text"]
    print(f"🎙 Transcription: {text}")
    return text

def ask_llama(question):
    print("🤖 Processing with LLaMA...")
    system_instruction = (
        "You are Albert Einstein, a brilliant theoretical physicist and mathematician. "
        "When explaining math or other concepts, avoid using any math symbols or special characters. "
        "Use only words and numbers in plain text. Keep your answers short. "
        "If you need to do a calculation, provide only the final numeric result "
        "or just the necessary concept. Do not include detailed symbolic math steps."
    )
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question}
        ]
    )
    answer = response["message"]["content"]
    print(f"🧠 LLaMA's Explanation: {answer}")
    return answer

def speak_text(text):
    tts.tts_to_file(
        text=text,
        file_path="response.wav",
        speaker=MALE_SPEAKER,
        speed=1.8
    )
    data, samplerate = sf.read("response.wav")
    sd.play(data, samplerate)
    sd.wait()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    while True:
        proceed = record_audio()
        if not proceed:
            print("👋 Exiting. Goodbye!")
            break
        question = transcribe_audio()
        reply = ask_llama(question)
        speak_text(reply)
