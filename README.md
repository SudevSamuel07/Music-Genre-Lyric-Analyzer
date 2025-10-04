        # Music Genre & Lyrics Analyzer
   
A comprehensive music analysis web application that combines machine learning and AI to provide dual insights into audio files:

- **Genre Classification**: Uses a Convolutional Neural Network (CNN) trained on MFCC features to predict music genres
- **Lyrics Transcription**: Leverages OpenAI's Whisper model for accurate speech-to-text conversion of song vocals

## 🎵 Features

### Core Functionality
- **Multi-format Audio Support**: Handles MP3, WAV, M4A, FLAC, and OGG files
- **Dual Input Methods**: Upload files directly or provide URLs to remote audio files
- **Real-time Analysis**: Instant genre prediction and lyrics transcription
- **Web Interface**: User-friendly Flask-based web application
- **Robust Error Handling**: Comprehensive validation and user-friendly error messages

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

## 🔧 Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.10-3.13)
- FFmpeg (required for Whisper audio processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/SudevSamuel07/Music-Genre-Lyric-Analyzer.git
cd Music-Genre-Lyric-Analyzer
```

### Step 2: Create Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install FFmpeg
**Windows:**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract and add the `bin` directory to your system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

## 🚀 Usage

### Starting the Web Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface
1. Open your browser and navigate to `http://localhost:5000`
2. Choose one of two input methods:
   - **File Upload**: Click "Choose File" and select an audio file
   - **URL Input**: Paste a direct link to an audio file
3. Click "Analyze" to process the file
4. View results:
   - **Predicted Genre**: AI-predicted music genre
   - **Transcribed Lyrics**: Extracted lyrics from the audio

### Supported Audio Formats
- MP3 (recommended)
- WAV
- M4A
- FLAC
- OGG

## 🧠 How It Works

### Genre Classification Pipeline
1. **Audio Loading**: Load audio file and resample to 22,050 Hz
2. **Segmentation**: Split 30-second clips into 10 segments
3. **Feature Extraction**: Extract 13 MFCC coefficients per segment
4. **CNN Prediction**: Feed features through pre-trained CNN model
5. **Majority Voting**: Combine segment predictions for final genre

### Lyrics Transcription Pipeline
1. **Audio Validation**: Check file existence and size
2. **Whisper Loading**: Load cached OpenAI Whisper model
3. **Audio Processing**: Use `whisper.load_audio()` and `pad_or_trim()`
4. **Transcription**: Convert speech to text using Whisper's ASR
5. **Error Handling**: Return meaningful error messages on failure

## 🔧 Technical Details

### Dependencies
- **Flask**: Web framework for the user interface
- **TensorFlow/Keras**: CNN model loading and inference
- **librosa**: Audio processing and MFCC extraction
- **OpenAI Whisper**: Speech recognition and transcription
- **NumPy**: Numerical operations
- **scikit-learn**: Machine learning utilities
- **requests**: HTTP client for URL downloads

### Model Architecture
- **Input**: MFCC features (shape: segments × time_frames × 13)
- **CNN Layers**: Convolutional layers for pattern recognition
- **Output**: 10 music genre classes (based on GTZAN dataset)

### Performance Optimizations
- Model caching to avoid reloading
- Efficient audio processing with librosa
- Chunked file downloads for URLs
- Segment-based analysis for robustness

## 🛠️ Configuration

### Environment Variables (Optional)
```bash
# Reduce TensorFlow logging verbosity
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
```

### Application Settings
- **Upload Size Limit**: 200MB (configurable in `app.py`)
- **Whisper Model**: "base" (can be changed to "small", "medium", "large")
- **Audio Parameters**: 22,050 Hz sample rate, 30-second duration

## 🐛 Troubleshooting

### Common Issues

**1. "Saved model not found" Error**
- Ensure `genre_classifier.h5` is in the project root
- Run the training script if model is missing

**2. "Failed to load data.json mapping" Error**  
- Verify `data.json` exists and contains genre mappings
- Check file permissions and format

**3. Whisper Transcription Failures**
- Install FFmpeg and ensure it's in PATH
- Check audio file format and integrity
- Try converting to WAV format: `ffmpeg -i input.mp3 output.wav`

**4. "Loaded audio is empty" Error**
- Audio file may be corrupted or unsupported
- Try a different audio file or format
- Check file size (minimum 1KB required)

**5. High Memory Usage**
- Whisper models can be memory-intensive
- Consider using smaller Whisper models ("tiny", "small")
- Close other applications if running out of memory

### Performance Tips
- First request may be slow due to model loading
- Subsequent requests are much faster (models cached)
- Use shorter audio clips for faster processing
- Ensure stable internet connection for URL downloads

## 🔮 Future Enhancements

- [ ] REST API endpoint for programmatic access
- [ ] Background processing with Celery/RQ
- [ ] Multiple genre prediction confidence scores
- [ ] Audio visualization and waveform display
- [ ] Batch processing for multiple files
- [ ] Genre-specific lyrics analysis
- [ ] Docker containerization
- [ ] Cloud deployment options

## 📊 Model Information

The genre classification model was trained on the GTZAN dataset, which includes:
- **Genres**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **Training Data**: 1,000 30-second audio clips (100 per genre)
- **Features**: 13 MFCC coefficients extracted from audio segments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **GTZAN Dataset**: George Tzanetakis for the music genre dataset
- **OpenAI Whisper**: For the powerful speech recognition model
- **librosa**: For comprehensive audio processing capabilities
- **TensorFlow**: For the machine learning framework

---

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/SudevSamuel07/Music-Genre-Lyric-Analyzer).

