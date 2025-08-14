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

1. Genre Classification (Music Information Retrieval)
This part of the project answers the question: "What does the music sound like?"

Feature Extraction: The audio signal is loaded and processed using the librosa library. It's converted into a series of MFCC spectrograms, which are visual representations of the audio's timbre and frequency.

Model Training: A CNN is trained on thousands of these MFCC "images" from the GTZAN dataset. The model learns to recognize the unique visual patterns associated with each of the 10 different music genres.

Prediction: For a new song, its MFCCs are generated and fed into the trained CNN, which predicts the most likely genre.

2. Lyric Transcription (Natural Language Processing)
This part of the project answers the question: "What are the lyrics about?"

ASR Model: The project uses OpenAI's Whisper, a pre-trained model designed for robust Automatic Speech Recognition (ASR).

Transcription: The Whisper model processes the raw audio waveform of the song and converts the sung vocals into a text string, providing the song's lyrics.

Technology Stack
Dataset
GTZAN Genre Collection: A popular benchmark dataset consisting of 1,000 audio tracks, each 30 seconds long, covering 10 music genres.

Software and Libraries
Python 3.x

TensorFlow / Keras: For building and training the CNN model for genre classification.

Librosa: For audio processing and extracting MFCC features.

OpenAI Whisper: For transcribing audio to text.

scikit-learn: For splitting the dataset into training and testing sets.

NumPy: For numerical operations.

Matplotlib: For plotting the model's training history.

FFmpeg: A command-line tool required by Whisper for audio/video processing.

Setup and Installation
Follow these steps to set up the project on your local machine.

Clone the repository:

git clone <your-repository-url>
cd <your-repository-name>

Create a Python virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required Python libraries:

pip install tensorflow numpy scikit-learn matplotlib librosa
pip install git+[https://github.com/openai/whisper.git](https://github.com/openai/whisper.git)

Install FFmpeg:

On macOS (using Homebrew):

brew install ffmpeg

On Debian/Ubuntu:

sudo apt update && sudo apt install ffmpeg

On Windows: Download the binaries from the official website and add the bin folder to your system's PATH.

Download the Dataset:

Download the GTZAN dataset from Kaggle.

Extract the archive and place the genres_original folder inside a Data directory in the root of the project.

Usage
The project is structured to run in two main phases from a single script.

Phase 1: Data Preprocessing and Model Training

First, you need to generate the data.json file from the GTZAN dataset.

Then, the script will automatically train the genre classification model and save it as genre_classifier.h5.

Phase 2: Analysis of a New Song

After training, the script will load the saved model.

It will then analyze the audio file specified by the AUDIO_PATH variable in the script.

Finally, it will print the predicted genre and the transcribed lyrics.

To run the entire pipeline, simply execute the main Python script:

python your_script_name.py

Note: The first run will take a significant amount of time due to data preprocessing and model training. Subsequent runs will be much faster as they can load the pre-trained model.
