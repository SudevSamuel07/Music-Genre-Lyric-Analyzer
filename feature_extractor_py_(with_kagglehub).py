import json
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import librosa
import whisper # New import for transcription



# Path to the JSON file that stores MFCCs and genre labels
DATA_PATH = "data.json"
# Path to save/load the trained model
MODEL_PATH = "genre_classifier.h5"
# Path to the new audio file you want to predict
AUDIO_PATH = "C:\\Users\\sudev\\Downloads\\Sapphire (Mp3 Song)-(SambalpuriStar.In).mp3"

# Audio processing parameters
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# --- Functions for Data Loading and Training (Unchanged) ---

def load_data(data_path):
    """Loads training dataset from json file."""
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    print("Data successfully loaded!")
    return X, y

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs."""
    fig, axs = plt.subplots(2)
    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    plt.show()

def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets."""
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    """Generates CNN model."""
    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# --- Functions for Prediction and Transcription ---

def load_genre_mapping(json_path):
    """Loads the genre mapping from the json file."""
    with open(json_path, "r") as fp:
        data = json.load(fp)
    return data["mapping"]

def process_audio(audio_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """Extracts MFCCs from a single audio file."""
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = np.ceil(samples_per_segment / hop_length)
    try:
        signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Could not load file {audio_path}: {e}")
        return None
    all_mfccs = []
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == num_mfcc_vectors_per_segment:
            all_mfccs.append(mfcc.tolist())
    return np.array(all_mfccs)

def predict_genre(model, audio_features, genre_mapping):
    """Predicts the genre of an audio file using a majority vote over its segments."""
    audio_features = audio_features[..., np.newaxis]
    predictions = model.predict(audio_features)
    predicted_indices = np.argmax(predictions, axis=1)
    most_common_prediction_index = np.bincount(predicted_indices).argmax()
    predicted_genre = genre_mapping[most_common_prediction_index]
    return predicted_genre

def transcribe_audio(audio_path):
    """Transcribes the audio file to text using the Whisper ASR model."""
    print("\n--- Transcribing Lyrics with Whisper ---")
    try:
        # Load the base Whisper model
        model = whisper.load_model("base")
        
        # Transcribe the audio file
        result = model.transcribe(audio_path)
        lyrics = result["text"]
        
        print("Transcription successful!")
        return lyrics
    except Exception as e:
        print(f"Could not transcribe audio: {e}")
        return "Lyrics could not be transcribed."


if __name__ == "__main__":
    
    # --- Part 1: Train the Genre Classification Model ---
    print("--- Starting Model Training ---")
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    plot_history(history)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    # Save the trained model
    model.save(MODEL_PATH)
    print(f"\nModel saved successfully to {MODEL_PATH}")

    # --- Part 2: Analyze Your Song ---
    print("\n--- Analyzing New Song ---")
    genre_mapping = load_genre_mapping(DATA_PATH)
    print(f"\nProcessing {AUDIO_PATH} for genre classification...")
    audio_features = process_audio(AUDIO_PATH)
    
    predicted_genre = "Not determined"
    if audio_features is not None and audio_features.shape[0] > 0:
        predicted_genre = predict_genre(model, audio_features, genre_mapping)
    else:
        print("Could not extract features for genre classification.")

    # Transcribe the lyrics from the same audio file
    transcribed_lyrics = transcribe_audio(AUDIO_PATH)

    # --- Part 3: Final Combined Output ---
    print("\n\n--- FINAL ANALYSIS ---")
    print(f"Predicted Genre: {predicted_genre}")
    print(f"Transcribed Lyrics: {transcribed_lyrics}")

