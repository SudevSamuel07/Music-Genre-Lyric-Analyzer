import os
import tempfile
import requests
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename  

from features import process_audio, load_keras_model, load_genre_mapping, predict_genre, transcribe_audio_whisper

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "flac", "ogg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB limit


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # either file upload or url
    file = request.files.get("file")
    url = request.form.get("url")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(tmp_path)
    elif url:
        # attempt to download URL to a temp file
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            suffix = os.path.splitext(url.split("?")[0])[1] or ".mp3"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=app.config["UPLOAD_FOLDER"]) 
            with open(tmp.name, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            tmp_path = tmp.name
        except Exception as e:
            return render_template("index.html", error=f"Failed to download URL: {e}")
    else:
        return render_template("index.html", error="No valid file upload or URL provided.")

    # load model
    model = load_keras_model()
    if model is None:
        return render_template("index.html", error="Saved model not found. Please train the model or place 'genre_classifier.h5' in the project folder.")

    # load mapping
    try:
        mapping = load_genre_mapping()
    except Exception as e:
        return render_template("index.html", error=f"Failed to load data.json mapping: {e}")

    # extract features
    audio_features = process_audio(tmp_path)
    if audio_features is None:
        return render_template("index.html", error="Could not extract audio features from the provided file.")

    # predict
    predicted_genre = predict_genre(model, audio_features, mapping)

    # transcribe (returns text, error)
    transcription, transcribe_error = transcribe_audio_whisper(tmp_path)

    # If transcription failed, surface an error message in the UI but keep genre
    if transcribe_error:
        return render_template("index.html", result=True, genre=predicted_genre, transcription="", error=transcribe_error)

    return render_template("index.html", result=True, genre=predicted_genre, transcription=transcription)


if __name__ == "__main__":
    # When running under the VS Code debugger the Flask reloader will spawn a
    # child process and cause a SystemExit in the parent. Disable the reloader
    # when debugging to avoid SystemExit: 3. If you want the reloader locally,
    # run the script outside of the debugger or set use_reloader=True.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
