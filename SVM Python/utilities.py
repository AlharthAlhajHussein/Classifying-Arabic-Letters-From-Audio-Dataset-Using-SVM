import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def extract_features_v2(file_path, n_mfcc=20, n_mels=128, frames=41):
    """
    Extract advanced audio features from a file
    
    Parameters:
    file_path (str): Path to the audio file
    n_mfcc (int): Number of MFCCs to extract (increased from 13 to 20)
    n_mels (int): Number of Mel bands to generate (increased from 40 to 128)
    frames (int): Number of frames to consider
    
    Returns:
    np.array: Feature vector with enhanced acoustic characteristics
    """
    try:
        # Load the audio file with a fixed duration
        y, sr = librosa.load(file_path, sr=22050)  # Fixed sample rate for consistency
        
        # Check if audio is empty or contains only zeros/silence
        if np.mean(np.abs(y)) < 0.001 or len(y) < sr * 0.1:  # Less than 100ms or very low amplitude
            return None  # Skip this file
        
        # Apply pre-emphasis filter to boost high frequencies
        y_emphasized = librosa.effects.preemphasis(y)
        
        # Remove silent parts (noise reduction)
        y_filtered, _ = librosa.effects.trim(y_emphasized, top_db=20)
        
        # Check duration and trim or pad as needed
        target_duration = 1.0  # 1 second
        if len(y_filtered) > sr * target_duration:
            y_filtered = y_filtered[:int(sr * target_duration)]  # Trim to 1 second
        elif len(y_filtered) < sr * target_duration:
            padding = np.zeros(int(sr * target_duration) - len(y_filtered))
            y_filtered = np.concatenate([y_filtered, padding])  # Pad to 1 second
        
        # Window the signal to avoid spectral leakage
        y_windowed = y_filtered * np.hamming(len(y_filtered))
        
        # Create feature list to store all extracted features
        features = []
        
        # 1. MFCCs with delta and delta-delta (captures temporal changes)
        mfccs = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels)
        mfcc_delta = librosa.feature.delta(mfccs)  # First-order derivatives
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)  # Second-order derivatives
        
        # Process MFCCs with statistics
        for mfcc_features in [mfccs, mfcc_delta, mfcc_delta2]:
            features.append(np.mean(mfcc_features, axis=1))  # Mean of each coefficient
            features.append(np.std(mfcc_features, axis=1))   # Standard deviation
            features.append(np.min(mfcc_features, axis=1))   # Minimum
            features.append(np.max(mfcc_features, axis=1))   # Maximum
        
        # 2. Spectral Features
        # 2.1 Spectral Centroid (weighted mean of frequencies)
        spectral_centroid = librosa.feature.spectral_centroid(y=y_filtered, sr=sr)
        features.append(np.mean(spectral_centroid, axis=1))
        features.append(np.std(spectral_centroid, axis=1))
        
        # 2.2 Spectral Bandwidth (variance around centroid)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_filtered, sr=sr)
        features.append(np.mean(spectral_bandwidth, axis=1))
        features.append(np.std(spectral_bandwidth, axis=1))
        
        # 2.3 Spectral Contrast (valley-to-peak energy difference in each band)
        spectral_contrast = librosa.feature.spectral_contrast(y=y_filtered, sr=sr, n_bands=6)
        features.append(np.mean(spectral_contrast, axis=1))
        features.append(np.std(spectral_contrast, axis=1))
        
        # 2.4 Spectral Flatness (tonal vs. noise-like)
        spectral_flatness = librosa.feature.spectral_flatness(y=y_filtered)
        features.append(np.mean(spectral_flatness, axis=1))
        features.append(np.std(spectral_flatness, axis=1))
        
        # 2.5 Spectral Rolloff (frequency below which X% of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_filtered, sr=sr, roll_percent=0.85)
        spectral_rolloff_60 = librosa.feature.spectral_rolloff(y=y_filtered, sr=sr, roll_percent=0.60)
        features.append(np.mean(spectral_rolloff, axis=1))
        features.append(np.std(spectral_rolloff, axis=1))
        features.append(np.mean(spectral_rolloff_60, axis=1))
        
        # 3. Rhythm Features
        # 3.1 Tempo and Beat Strength
        onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features.append(np.array([tempo]))
        features.append(np.array([np.mean(onset_env)]))  # Beat strength
        features.append(np.array([np.std(onset_env)]))   # Beat variation
        
        # 4. Zero Crossing Rate (related to frequency content)
        zcr = librosa.feature.zero_crossing_rate(y_filtered)
        features.append(np.mean(zcr, axis=1))
        features.append(np.std(zcr, axis=1))
        
        # 5. Root Mean Square Energy (volume/loudness variation)
        rms = librosa.feature.rms(y=y_filtered)
        features.append(np.mean(rms, axis=1))
        features.append(np.std(rms, axis=1))
        
        # 6. Mel Spectrogram for enhanced spectral information
        mel_spectrogram = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Take statistics from different frequency bands (low, mid, high)
        mel_bands = np.array_split(mel_spectrogram_db, 4, axis=0)  # Split into 4 frequency bands
        for band in mel_bands:
            features.append(np.mean(band, axis=1))
            features.append(np.std(band, axis=1))
        
        # 7. Chroma Features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(y=y_filtered, sr=sr, n_chroma=12)
        features.append(np.mean(chroma, axis=1))
        features.append(np.std(chroma, axis=1))
        
        # 8. Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=y_filtered, sr=sr)
        features.append(np.mean(tonnetz, axis=1))
        features.append(np.std(tonnetz, axis=1))
        
        # 9. Spectral Flux (spectral change over time)
        spectral_flux = np.diff(mel_spectrogram, axis=1)
        features.append(np.array([np.mean(np.abs(spectral_flux))]))
        
        # 10. Formant frequencies (helpful for vowel distinction)
        # Approximate formants using spectral peaks
        S = np.abs(librosa.core.stft(y_filtered))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Find the most prominent frequencies
        frame_peaks = []
        for frame in range(S.shape[1]):
            spectrum = S[:, frame]
            peak_indices = librosa.util.peak_pick(spectrum, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=0.5, wait=10)
            if len(peak_indices) > 0:
                # Sort by amplitude and take top 3 (if available)
                sorted_peaks = sorted([(spectrum[i], freqs[i]) for i in peak_indices], reverse=True)
                frame_peaks.extend([freq for _, freq in sorted_peaks[:3]])
        
        # Add formant statistics if found
        if frame_peaks:
            features.append(np.array([np.mean(frame_peaks)]))
            features.append(np.array([np.std(frame_peaks)]))
        else:
            features.append(np.array([0]))
            features.append(np.array([0]))
        
        # Flatten and concatenate all features
        feature_vector = np.concatenate([feat.flatten() for feat in features])
        
        # Apply feature scaling (standardization)
        feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-10)
        
        return feature_vector
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Define a function to extract features from audio files
def extract_features(file_path, n_mfcc=13, n_mels=40, frames=41):
    """
    Extract audio features from a file
    
    Parameters:
    file_path (str): Path to the audio file
    n_mfcc (int): Number of MFCCs to extract
    n_mels (int): Number of Mel bands to generate
    frames (int): Number of frames to consider
    
    Returns:
    np.array: Feature vector
    """
    try:
        # Load the audio file with a fixed duration
        y, sr = librosa.load(file_path, sr=None)
        
        # Check if audio is empty or contains only zeros/silence
        if np.mean(np.abs(y)) < 0.001 or len(y) < sr * 0.1:  # Less than 100ms or very low amplitude
            return None  # Skip this file
        
        # Check duration and trim or pad as needed
        target_duration = 1.0  # 1 second
        if len(y) > sr * target_duration:
            y = y[:int(sr * target_duration)]  # Trim to 1 second
        elif len(y) < sr * target_duration:
            padding = np.zeros(int(sr * target_duration) - len(y))
            y = np.concatenate([y, padding])  # Pad to 1 second
        
        # Extract features
        # 1. MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        # 2. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_processed = np.mean(spectral_centroid.T, axis=0)
        
        # 3. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_processed = np.mean(spectral_rolloff.T, axis=0)
        
        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_processed = np.mean(zcr.T, axis=0)
        
        # 5. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_processed = np.mean(chroma.T, axis=0)
        
        # Combine all features
        feature_vector = np.concatenate([
            mfccs_processed, 
            spectral_centroid_processed,
            spectral_rolloff_processed,
            zcr_processed,
            chroma_processed
        ])
        
        return feature_vector
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to load data from all folders
def load_dataset(root_directory):
    """
    Load data from all folders in the root directory
    
    Parameters:
    root_directory (str): Path to the root directory containing class folders
    
    Returns:
    tuple: X (features), y (labels)
    """
    features = []
    labels = []
    skipped_files = 0
    
    print("Loading and processing audio files...")
    
    # Count total files for progress tracking
    total_files = sum([len(files) for r, d, files in os.walk(root_directory) if files])
    processed_files = 0
    
    # Iterate through each folder (class)
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing folder: {folder_name}")
        
        # Iterate through each file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                
                # Extract features from the audio file
                feature_vector = extract_features_v2(file_path)
                
                # If features were successfully extracted
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(folder_name)
                else:
                    skipped_files += 1
                
                processed_files += 1
                if processed_files % 100 == 0:
                    print(f"Processed {processed_files}/{total_files} files...")
    
    print(f"Finished processing. Skipped {skipped_files} files due to silence or errors.")
    
    return np.array(features), np.array(labels)


# Function to visualize some audio examples
def visualize_sample_audios(root_directory, num_samples=3):
    """
    Visualize waveform and spectrogram for a few sample audio files
    
    Parameters:
    root_directory (str): Path to the root directory
    num_samples (int): Number of samples to visualize
    """
    plt.figure(figsize=(15, 10))
    
    sample_count = 0
    folders = os.listdir(root_directory)
    
    for folder_name in folders:
        folder_path = os.path.join(root_directory, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        if files:
            file_path = os.path.join(folder_path, files[0])
            y, sr = librosa.load(file_path, sr=None)
            
            # Plot waveform
            plt.subplot(num_samples, 2, sample_count*2 + 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform: {folder_name}')
            
            # Plot spectrogram
            plt.subplot(num_samples, 2, sample_count*2 + 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram: {folder_name}')
            
            sample_count += 1
            if sample_count >= num_samples:
                break
    
    plt.tight_layout()
    plt.show()
