# Meeting Summarizer

An AI-powered web application that automatically transcribes and summarizes meeting audio recordings. Built with state-of-the-art models like OpenAI Whisper for transcription and LED (Longformer Encoder-Decoder) for abstractive summarization.

---

## Project Overview

The **Meeting Summarizer** takes `.m4a`, `.mp4`, or `.wav` meeting audio files as input, transcribes the audio to text, and generates a structured summary with sections like:
- **[ABSTRACT]**
- **[ACTIONS]**
- **[DECISIONS]**
- **[PROBLEMS]**

It includes a web interface built with Flask, where users can upload files and receive real-time summaries and full transcripts.

---

## Features

- 🎙️ **Audio Transcription** using OpenAI Whisper (base model)
- 📝 **Structured Summarization** using fine-tuned `allenai/led-large-16384`
- 🌐 **Web Interface** for easy file upload and summary display
- 📁 **Support for `.m4a`, `.mp4`, `.wav` formats**
- 🔄 **Copy / Download Summary** options
- ✨ Clean UI with auto-handling of empty summary sections

---

## Project Workflow: Building the Summarizer

### 1. **Data Preparation**
- Used the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/download/) (Headset mix Audio version of ES2002a-ES2011d meetings).
- Used the Whisper base model to generate transcripts from meeting audio files and stored them in a new folder `(transcript.py)`.
- Parsed the summaries from AMI Corpus and aligned transcripts with structured labels like `[ABSTRACT]`, `[ACTIONS]`, `[DECISIONS]`, `[PROBLEMS]` and stored in `ami_aligned_sections.json` `(parse.py)`.

### 2. **Model Training** `(hf_model.py)`
- Fine-tuned the `allenai/led-large-16384` model using Hugging Face `Trainer`.
- Trained on `ami_aligned_sections.json` transcript-summary pairs.
- Saved the trained model to a directory like `led_summarization_model`.

### 3. **Audio Transcription Module**
- Loaded the base Whisper model.
- Transcribed input `.m4a`, `.mp4`, or `.wav` audio files using `whisper.transcribe()`.

### 4. **Summarization Module** `(summarize.py)`
- Tokenized and truncated transcription to fit the LED model’s input limit (16,384 tokens).
- Ran summarization using the trained LED model.

### 5. **Web Interface (Flask)**
- Users upload audio files via `upload.html`.
- After processing, `summary.html` displays the transcription and structured summary.
- Buttons provided to copy/download the result.
- Optional enhancements: auto-delete uploaded files, input validation, error handling.

---

## Technologies Used

- **Python 3**
- **Flask**
- **Whisper (for transcription)**
- **HuggingFace Transformers (`LEDForConditionalGeneration`)**
- **HTML/CSS/JavaScript (Frontend)**

---

## Installation & Setup

- Cloning Repository and Installing Dependencies
```
bash
# Clone the repository
git clone https://github.com/AnkitChauhan19/Meeting-Summarizer.git
cd Meeting-Summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
- Training Model
```
bash
# Train the model and store it in a directory (change directory name if needed)
python hf_model.py
```
- Running app
```
bash

python app.py
```
