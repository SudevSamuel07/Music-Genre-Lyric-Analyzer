import json
import os
import numpy as np
import librosa
import whisper
import tensorflow.keras as keras

# Paths (same defaults used by original script)
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "genre_classifier.h5")

SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def load_genre_mapping(json_path=DATA_PATH):
    """Loads the genre mapping from the json file."""
    with open(json_path, "r") as fp:
        data = json.load(fp)
    return data["mapping"]


def process_audio(audio_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """Extracts MFCCs from a single audio file. Returns array(shape=(segments, num_vectors, num_mfcc))."""
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = int(np.ceil(samples_per_segment / hop_length))
    try:
        signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Could not load file {audio_path}: {e}")
        return None
    all_mfccs = []
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        segment = signal[start:finish]
        if len(segment) < samples_per_segment:
            # skip incomplete segment at end
            continue
        mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        if mfcc.shape[0] == num_mfcc_vectors_per_segment:
            all_mfccs.append(mfcc.tolist())
    if len(all_mfccs) == 0:
        return None
    return np.array(all_mfccs)


def predict_genre(model, audio_features, genre_mapping):
    """Predicts the genre of an audio file using a majority vote over its segments."""
    audio_features = audio_features[..., np.newaxis]
    predictions = model.predict(audio_features)
    predicted_indices = np.argmax(predictions, axis=1)
    most_common_prediction_index = np.bincount(predicted_indices).argmax()
    predicted_genre = genre_mapping[most_common_prediction_index]
    return predicted_genre


# Whisper model is expensive to load; keep one instance in this module.
_whisper_model = None


def transcribe_audio_whisper(audio_path, model_name="base"):
    """Transcribes audio using Whisper safely.

    Returns a tuple: (transcribed_text, error_message_or_None)
    """
    global _whisper_model
    # basic file checks
    if not os.path.exists(audio_path):
        return "", "Audio file not found."
    if os.path.getsize(audio_path) < 1024:
        return "", "Audio file is too small or empty (possible download/encoding error)."

    try:
        if _whisper_model is None:
            print(f"Loading Whisper model '{model_name}' (this may take a while)...")
            _whisper_model = whisper.load_model(model_name)

        # Load audio as a numpy array and ensure it has content
        audio = whisper.load_audio(audio_path)
        if audio.size == 0:
            return "", "Loaded audio is empty (ffmpeg/decoder likely failed)."

        # pad or trim to acceptable length (prevents zero-length tensors)
        audio = whisper.pad_or_trim(audio)

        # Do the transcription
        result = _whisper_model.transcribe(audio_path)
        return result.get("text", ""), None
    except Exception as e:
        # return the exception message so the UI can show it
        return "", f"Whisper transcription failed: {e}"


def load_keras_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        return None
    return keras.models.load_model(model_path)
