# 🎙️ Arabic Letters Audio Recognition

This application uses machine learning to recognize and classify Arabic letters based on their audio pronunciation. Leveraging Support Vector Machine (SVM) and advanced audio feature extraction techniques, the system achieves 63% accuracy in identifying Arabic letters from about 1400 spoken audio.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 📝 Overview

The Arabic Letters Audio Recognition system is designed to identify Arabic letters from audio input. It features:

- A user-friendly desktop application built with CustomTkinter
- Real-time audio recording and processing capabilities
- Advanced audio feature extraction using librosa
- Support Vector Machine classification model trained on a diverse dataset
- Ability to upload existing audio files or record new ones

## 🌟 Features

- **Interactive GUI**: Modern, intuitive interface for easy interaction
- **Real-time Recognition**: Record and instantly analyze Arabic letter pronunciations
- **File Upload**: Process pre-recorded audio files in WAV, MP3, or OGG formats
- **Advanced Audio Processing**: Sophisticated feature extraction including MFCCs, spectral features, rhythm features, and more
- **Confidence Metrics**: Visual indicators for prediction confidence
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🎬 Demo

![Application Screenshot](https://github.com/AlharthAlhajHussein/Classifying-Arabic-Letters-From-Audio-Dataset-Using-SVM/blob/main/Images/screenshot.png)

[▶️ Watch Demo Video](https://www.youtube.com/watch?v=9vnx0FEfwnI)

## 📂 Project Structure

```
📦 Arabic Letter Recognition
 ┣ 📂 SVM Python
 ┃ ┣ 📜 Arabic Letter Recognition APP.py  # Main application
 ┃ ┣ 📜 SVM.ipynb                        # Model training notebook
 ┃ ┣ 📜 utilities.py                      # Feature extraction utilities
 ┃ ┣ 📜 arabic_letter_audio_classifier.pkl # Trained model
 ┃ ┣ 📜 label_encoder.pkl                 # Label encoder for Arabic letters
 ┃ ┗ 📜 requirements.txt                  # Dependencies
 ┃
 ┣ 📂 Cleaned Audio Dataset               # Training dataset
 ┃ ┣ 📂 Alif
 ┃ ┣ 📂 Ba
 ┃ ┣ ...
 ┃ 
 ┗ 📂 Images                              # Project images
   ┗ 📜 screenshot.png                    # App screenshot
```

## 🛠️ Technology Stack

- **Python**: Core programming language
- **librosa**: Audio processing and feature extraction
- **scikit-learn**: Machine learning algorithms and pipeline
- **CustomTkinter**: Modern GUI framework
- **sounddevice & soundfile**: Audio recording and file handling
- **numpy**: Numerical computations
- **matplotlib**: Data visualization

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AlharthAlhajHussein/Classifying-Arabic-Letters-From-Audio-Dataset-Using-SVM.git
   cd Classifying-Arabic-Letters-From-Audio-Dataset-Using-SVM
   ```

2. Install dependencies:
   ```bash
   cd "SVM Python"
   pip install -r requirements.txt
   ```

## 🚀 Usage

To run the application:

```bash
python "SVM Python/Arabic Letter Recognition APP.py"
```

The application offers two main functionalities:
1. **Record Audio**: Click the "Record" button and pronounce an Arabic letter for 1 second
2. **Upload Audio**: Click "Upload Audio File" to analyze a pre-recorded audio file

## 📊 Model Information

- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Feature Extraction**: 
  - MFCCs with delta and delta-delta coefficients
  - Spectral features (centroid, bandwidth, contrast, flatness, rolloff)
  - Rhythm features (tempo, beat strength)
  - Zero crossing rate
  - Chroma features
  - Tonnetz (tonal centroid features)
- **Hyperparameters**: C=100, gamma=0.0003
- **Performance**: High accuracy across different Arabic letters

## 📚 Dataset

The model was trained on a dataset of Arabic letter pronunciations, with multiple samples for each letter. The audio files were processed and cleaned to ensure consistent quality and duration. You could also use the Dataset from [Kaggle](https://www.kaggle.com/datasets/alharthalhajhussein/arabic-letters-as-audio-data).

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is open source and available under the [Aleppo University License](LICENSE).

## 👨‍💻 Author

**Alharth Alhaj Hussein**

Connect with me:
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alharth-alhaj-hussein-023417241)  
- [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AlharthAlhajHussein)   
- [![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Alharth.Alhaj.Hussein)
- [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/alharthalhajhussein)

---

If you find this project useful, please consider giving it a ⭐
