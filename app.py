import os
import whisper
import torch
import regex as re
from summarize import load_trained_model, summarize_transcript, extract_sections
from flask import Flask, request, render_template

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"m4a", "mp4", "wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_model = whisper.load_model("base").to(device)

model, tokenizer = load_trained_model()

@app.route("/")
def upload_form():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return render_template("upload.html", error="No file selected")
    
    file = request.files["file"]
    if file.filename == "":
        return render_template("upload.html", error="No file selected")
    
    if file and allowed_file(file.filename):
        try:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            
            print("Starting transcription...")
            transcription = transcribe(filepath)
            
            print("Generating summary...")
            summary = summarize(transcription)
            
            os.remove(filepath)
            
            summary = separate_sections(summary)
            abstract = summary['ABSTRACT']
            actions = summary['ACTIONS']
            decisions = summary['DECISIONS']
            problems = summary['PROBLEMS']

            return render_template("summary.html", 
                                 abstract=abstract,
                                 actions=actions,
                                 decisions=decisions,
                                 problems=problems, 
                                 transcription=transcription,
                                 filename=file.filename)
                                 
        except Exception as e:
            print(f"Error processing file: {e}")
            return render_template("upload.html", error="Error processing file. Please try again.")
    else:
        return render_template("upload.html", error="Invalid file type. Please upload .m4a, .mp4, or .wav files.")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe(filepath):
    print(f"Transcribing: {filepath}...")
    result = whisper_model.transcribe(filepath, language="en", fp16=(device == "cuda"))
    return result["text"]

def summarize(text, max_length=5120):
    final_summary = ""

    final_summary = summarize_transcript(transcript=text, model=model, tokenizer=tokenizer)

    return final_summary

def separate_sections(summary):
    result = extract_sections(summary)

    return result

if __name__ == "__main__":
    app.run(port=5000, debug=True)