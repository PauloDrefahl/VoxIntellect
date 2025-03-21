import sounddevice as sd
import numpy as np
import wave
import whisper
import ollama
import soundfile as sf
from TTS.api import TTS

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
SAMPLE_RATE = 44100  # CD-quality
CHANNELS = 1
DURATION_LIMIT = 60  # Max recording duration (seconds)
AUDIO_FILE = "recorded_audio.wav"

# ------------------------------------------------------------------------------
# Load the multi-speaker model *once* at the top to avoid re-initializing.
# ------------------------------------------------------------------------------
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)

# If you're curious which speakers exist, uncomment to print them:
# print("Available speakers:", tts.speakers)
# Example output might be: ['p225', 'p226', 'p227', 'p228', ...]

# Choose a male speaker from the list (often "p228" is male).
MALE_SPEAKER = "p228"

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def record_audio():
    """Records audio until the user presses Enter."""
    print("🎤 Recording... Press Enter to stop.")
    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(status)
        frames.append(indata.copy())

    # Open input stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        input()  # Wait for user input (press Enter to stop)
    
    print("🛑 Recording stopped.")

    # Convert recorded data to NumPy array
    audio_data = np.concatenate(frames, axis=0)

    # Save as a WAV file
    with wave.open(AUDIO_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    print(f"✅ Audio saved as {AUDIO_FILE}")


def transcribe_audio():
    """Uses Whisper to transcribe the recorded audio."""
    print("📝 Transcribing audio with Whisper...")
    model = whisper.load_model("tiny")
    result = model.transcribe(AUDIO_FILE)
    text = result["text"]
    print(f"🎙 Transcription: {text}")
    return text


def ask_llama(question):
    """Sends the transcribed question to LLaMA (Ollama) and gets a response."""

    print("🤖 Processing with LLaMA...")

    system_instruction = (
        "You are Sir Isaac Newton, a brilliant mathematician. "
        "When explaining math or other concepts, avoid using any math symbols or special characters. "
        "Use only words and numbers in plain text. "
        "Keep your answers short. "
        "If you need to do a calculation, provide only the final numeric result "
        "or just the necessary concept. Do not include detailed symbolic math steps."
    )

    # We now supply two messages: one “system” style and one from the "user".
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
    """
    Speaks the text using Coqui TTS (multi-speaker VITS).
    We'll pick a male speaker and speed up the speech slightly.
    """
    # Example: speed=1.2 => ~20% faster. If this doesn't work, try duration_factor=0.85
    tts.tts_to_file(
        text=text,
        file_path="response.wav",
        speaker=MALE_SPEAKER,
        speed=1.8  # <--- For a slightly faster rate
        # or: duration_factor=0.85
    )

    # Play the generated WAV
    data, samplerate = sf.read("response.wav")
    sd.play(data, samplerate)
    sd.wait()


# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    record_audio()
    question_text = transcribe_audio()
    explanation = ask_llama(question_text)
    speak_text(explanation)
