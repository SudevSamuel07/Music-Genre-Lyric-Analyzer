Music Genre Classification and Lyric Transcription


This project is a deep learning application that analyzes an audio file to perform two main tasks:

Music Genre Classification: It identifies the genre of the music (e.g., Pop, Rock, Classical) by analyzing its sound features.

Lyric Transcription: It uses a state-of-the-art speech recognition model to transcribe the lyrics from the song.

This provides a comprehensive analysis of a music track, combining both Music Information Retrieval (MIR) and Natural Language Processing (NLP).

Features
Hybrid AI Approach: Combines a CNN for Music Information Retrieval (genre classification) with a state-of-the-art Transformer model (Whisper) for Natural Language Processing (lyric transcription).

CNN for Genre Classification: Utilizes a Convolutional Neural Network (CNN) trained on the GTZAN dataset to classify music based on its Mel-Frequency Cepstral Coefficients (MFCCs).

Whisper for Transcription: Employs OpenAI's powerful Whisper model for accurate, state-of-the-art speech-to-text transcription.

Self-Contained Script: A single Python script handles data preprocessing, model training, genre prediction, and lyric transcription.

How It Works
The project follows a two-pronged approach to analyze an audio file:
Music Genre Classification and Lyric Transcription (Web App)

This project analyzes an audio file and returns two results:

- Predicted music genre (using a CNN trained on MFCC features)
- Transcribed lyrics (using OpenAI Whisper)

Recent changes
- A minimal Flask web application was added to provide a simple UI and HTTP endpoint for analysis.
- New files: `app.py`, `features.py`, `templates/index.html`, and `requirements.txt`.

Files of interest
- `app.py` — Flask app that serves the web UI and an `/analyze` endpoint.
- `features.py` — audio feature extraction, model loading, genre prediction, and Whisper transcription helpers.
- `templates/index.html` — simple web UI (upload or provide URL).
- `genre_classifier.h5` — pretrained Keras model (binary, must be present to predict).
- `data.json` — dataset metadata (mapping of class indices to genre names). Required by `features`.

Requirements
- Python 3.8+ (project was developed and tested with Python 3.10–3.13)
- FFmpeg (must be installed and on PATH for Whisper to decode many audio formats)

Install

1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1    # Windows PowerShell
```

2. Install Python dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Note: `requirements.txt` installs `tensorflow`, `librosa`, and Whisper (via Git). Whisper and TensorFlow are large; installation may take time.

Run the web app

```powershell
# from project root
python app.py
```

The app starts on http://localhost:5000. Use the web form to upload an audio file or paste a direct URL to an audio file (mp3, wav, m4a, flac, ogg).

How it works (brief)
- The app extracts MFCC segments from the input audio (`features.process_audio`) and feeds them to the pre-trained Keras model (`genre_classifier.h5`) for per-segment predictions. A majority vote produces the final genre.
- Whisper is used to transcribe the audio. The code now validates audio content and uses `whisper.load_audio` + `whisper.pad_or_trim` to avoid zero-length tensor errors.

Notes & troubleshooting
- Ensure `genre_classifier.h5` and `data.json` are present in the project root. The web UI will return a friendly error if the model or mapping is missing.
- Whisper requires FFmpeg. On Windows, download FFmpeg and add its `bin` directory to your PATH.
- If you get an error like "Loaded audio is empty (ffmpeg/decoder likely failed)", try converting the file with ffmpeg and re-uploading:

```powershell
ffmpeg -y -i "uploads\input.mp3" -ar 16000 -ac 1 "uploads\input_conv.wav"
```

- If you run the Flask app under the VS Code debugger, the built-in reloader can raise SystemExit in the parent process. The app disables the reloader when started via `app.py` to avoid this problem.

- If TensorFlow or protobuf emits many warnings at startup, they are usually informational. To reduce noise for development you can set environment variables before running the app:

```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
python app.py
```

Development notes
- The app currently expects a pre-trained `genre_classifier.h5`. The original ETL/training script in the repo can generate `data.json` and train a model if you prefer to retrain — training can be slow and memory intensive.
- The transcription step loads a Whisper model (by default `base`) which can be slow to load; subsequent requests are faster once the model is cached in memory.

Next improvements you might consider
- Add an API JSON endpoint for programmatic use.
- Add background processing (Celery/RQ) to avoid long request timeouts for Whisper on large inputs.
- Add a small test that runs inference on a bundled short audio sample.

License & credits
- Uses GTZAN dataset (if you train) and OpenAI Whisper for ASR.

---
If you need me to update the README further (add screenshots, deployment steps, or an API reference), tell me what to include.
