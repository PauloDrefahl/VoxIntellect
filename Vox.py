import sounddevice as sd
import numpy as np
import wave
import soundfile as sf
import threading
from faster_whisper import WhisperModel
from TTS.api import TTS
from llama_cpp import Llama

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_FILE = "recorded_audio.wav"
DURATION_LIMIT = 60
MODEL_PATH = "/home/paulodrefahl/Desktop/llama.cpp/models/phi-2.gguf"

# ------------------------------------------------------------------------------
# Load Models Once
# ------------------------------------------------------------------------------
# Whisper on GPU
whisper_model = WhisperModel("tiny", compute_type="float16", device="cpu")

# TTS on GPU
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
tts.to("cuda")

# LLaMA.cpp model using GPU
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_gpu_layers=-1,
    verbose=False
)

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def record_audio():
    print("🎤 Press Enter to start recording or type 'q' then Enter to quit.")
    start_input = input().strip().lower()
    if start_input == 'q':
        return False

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

    with wave.open(AUDIO_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    print(f"✅ Audio saved as {AUDIO_FILE}")
    return True

def transcribe_audio():
    print("📝 Transcribing audio with Faster-Whisper...")
    segments, _ = whisper_model.transcribe(AUDIO_FILE)
    text = ""
    for segment in segments:
        text += segment.text
    print(f"🎙 Transcription: {text}")
    return text

def ask_llama(question):
    print("🤖 Answering with GPU-powered LLaMA.cpp...")
    prompt = f"You are Albert Einstein. Explain this in simple terms: {question}"
    response = llm(prompt, max_tokens=150)
    answer = response["choices"][0]["text"].strip()
    print(f"🧠 LLaMA's Answer: {answer}")
    return answer

def speak_text(text):
    tts.tts_to_file(
        text=text,
        file_path="response.wav",
        speaker=None,
        speed=1.8
    )
    data, samplerate = sf.read("response.wav")
    sd.play(data, samplerate)
    sd.wait()

# ------------------------------------------------------------------------------
# Main Loop
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
