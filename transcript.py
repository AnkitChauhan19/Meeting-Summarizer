import os
import whisper
import torch

AUDIO_FOLDER = "amicorpus/audio"
TRANSCRIPT_FOLDER = "amicorpus/transcripts"

os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

""" 
Load Whisper model for transcripting audio 
files and move it to GPU (if available) 
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

"""
Looping over all the audio files in the AUDIO_FOLDER 
and using the Whisper Base model to transcript these
files and store the transcription in the TRANSCRIPT_FOLDER
"""
for filename in os.listdir(AUDIO_FOLDER):
    if filename.endswith(".wav"):
        audio_path = os.path.join(AUDIO_FOLDER, filename)
        output_path = os.path.join(TRANSCRIPT_FOLDER, filename.replace(".wav", ".txt"))

        print(f"Transcribing {filename} on {device}...")

        result = model.transcribe(audio_path, language="en", fp16=(device == "cuda"))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

print("Transcription complete.")