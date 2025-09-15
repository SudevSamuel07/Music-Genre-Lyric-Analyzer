# Music Genre & Lyrics Analyzer

- Genre detection
- Lyric analysis

## Installation

Instructions for installing the project.

## Usage

### Technical Features
- **CNN-based Genre Classification**: Uses MFCC (Mel-Frequency Cepstral Coefficients) feature extraction
- **Segment-based Prediction**: Analyzes audio in multiple segments with majority voting
- **OpenAI Whisper Integration**: State-of-the-art speech recognition for lyrics transcription
- **Model Caching**: Efficient model loading and reuse for better performance
- **Large File Support**: Handles audio files up to 200MB

## 🏗️ Project Structure

```
Music-Genre-Lyric-Analyzer/
├── app.py                                      # Flask web application
├── features.py                                 # Core audio processing and ML functions
├── feature_extractor_py_(with_kagglehub).py   # Original training script
├── requirements.txt                            # Python dependencies
├── genre_classifier.h5                        # Pre-trained CNN model
├── data.json                                   # Genre mapping and dataset metadata
├── templates/
│   └── index.html                             # Web interface template
├── uploads/                                   # Temporary file storage
└── README.md                                  # This file
```

## License

Details about the license.

