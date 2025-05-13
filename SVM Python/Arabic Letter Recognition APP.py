import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
import joblib
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont
import librosa

# Import the extract_features_v2 function from the provided code
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
        print(f"Error extracting features: {e}")
        return None

def predict_letter(file_path):
    """
    Predict the Arabic letter from an audio file
    
    Parameters:
    file_path (str): Path to the audio file
    
    Returns:
    tuple: (predicted_letter, confidence)
    """
    try:
        # Load the trained pipeline and label encoder
        pipeline = joblib.load('arabic_letter_audio_classifier.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Extract features
        features = extract_features_v2(file_path)
        
        if features is not None:
            features = features.reshape(1, -1)
            prediction = pipeline.predict(features)
            probabilities = pipeline.predict_proba(features)
            max_prob = np.max(probabilities)
            predicted_class = label_encoder.inverse_transform(prediction)[0]
            print(f"Predicted letter: ({predicted_class}) with confidence: {max_prob:.2f}")
            return predicted_class, max_prob
        else:
            print("Could not extract features from the audio file.")
            return None, 0
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0

class ArabicLetterRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic Letter Recognition")
        self.root.geometry("600x600")  # Made taller to fit all elements
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create a frame for content
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Title Label
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="Arabic Letter Recognition", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(20, 30))
        
        # Instructions
        self.instruction_label = ctk.CTkLabel(
            self.main_frame,
            text="Press the record button and pronounce an Arabic letter for 1 second",
            font=ctk.CTkFont(size=14)
        )
        self.instruction_label.pack(pady=(0, 20))
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Ready",
            font=ctk.CTkFont(size=14),
            text_color="#4CAF50"
        )
        self.status_label.pack(pady=(0, 20))
        
        # Record button
        self.record_button = ctk.CTkButton(
            self.main_frame,
            text="Record (1s)",
            command=self.start_recording,
            width=200,
            height=50,
            font=ctk.CTkFont(size=16),
            fg_color="#E53935",
            hover_color="#C62828"
        )
        self.record_button.pack(pady=(0, 20))
        
        # Upload button (added with clear visibility)
        self.upload_button = ctk.CTkButton(
            self.main_frame,
            text="Upload Audio File",
            command=self.upload_audio,
            width=200,
            height=40,
            font=ctk.CTkFont(size=16),
            fg_color="#2196F3",
            hover_color="#1976D2"
        )
        self.upload_button.pack(pady=(0, 30))
        
        # Result frame
        self.result_frame = ctk.CTkFrame(self.main_frame)
        self.result_frame.pack(pady=10, fill="x", padx=40)
        
        # Result title
        self.result_title = ctk.CTkLabel(
            self.result_frame,
            text="Recognition Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.result_title.pack(pady=(10, 20))
        
        # Letter display
        self.letter_frame = ctk.CTkFrame(self.result_frame, fg_color=("#E0E0E0", "#2D2D2D"))
        self.letter_frame.pack(pady=10, padx=20)
        
        self.letter_label = ctk.CTkLabel(
            self.letter_frame,
            text="",
            font=ctk.CTkFont(size=60, weight="bold"),
            width=100,
            height=100
        )
        self.letter_label.pack(pady=10, padx=40)
        
        # Confidence display
        self.confidence_label = ctk.CTkLabel(
            self.result_frame,
            text="Confidence: --",
            font=ctk.CTkFont(size=16)
        )
        self.confidence_label.pack(pady=(5, 5))
        
        # Confidence progress bar - making sure it's visible and properly styled
        self.confidence_frame = ctk.CTkFrame(self.result_frame)
        self.confidence_frame.pack(pady=(0, 20), fill="x", padx=40)
        
        self.confidence_progress = ctk.CTkProgressBar(
            self.confidence_frame,
            width=300,
            height=20,
            corner_radius=10,
            progress_color="#4CAF50",
            mode="determinate"
        )
        self.confidence_progress.pack(pady=10, fill="x")
        self.confidence_progress.set(0)  # Initialize at 0
        
        # Setup variables
        self.recording = False
        self.sample_rate = 22050  # Same as used in feature extraction
        self.duration = 1  # 1 second recording
        self.temp_audio_file = "temp_recording.wav"
        
        # Check if model files exist and show warning if not
        if not os.path.exists('arabic_letter_audio_classifier.pkl') or not os.path.exists('label_encoder.pkl'):
            self.show_model_warning()
    
    def show_model_warning(self):
        """Show warning if model files are missing"""
        warning_window = ctk.CTkToplevel(self.root)
        warning_window.title("Warning")
        warning_window.geometry("400x200")
        warning_window.transient(self.root)
        warning_window.grab_set()
        
        warning_frame = ctk.CTkFrame(warning_window)
        warning_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        warning_label = ctk.CTkLabel(
            warning_frame,
            text="Model files are missing!\n\nPlease ensure 'arabic_letter_audio_classifier.pkl'\nand 'label_encoder.pkl' are in the same directory.",
            font=ctk.CTkFont(size=14),
            text_color="#FF5722"
        )
        warning_label.pack(pady=20)
        
        ok_button = ctk.CTkButton(
            warning_frame,
            text="OK",
            command=warning_window.destroy
        )
        ok_button.pack(pady=10)
    
    def update_status(self, text, color="#4CAF50"):
        """Update the status label with text and color"""
        self.status_label.configure(text=text, text_color=color)
        self.root.update()
    
    def update_confidence_display(self, confidence):
        """Update the confidence display with the given confidence value"""
        if confidence is None:
            self.confidence_label.configure(text="Confidence: --")
            self.confidence_progress.set(0)
            return
            
        # Update text display
        self.confidence_label.configure(text=f"Confidence: {confidence:.2f}")
        
        # Update progress bar value (between 0 and 1)
        self.confidence_progress.set(confidence)
        
        # Update progress bar color based on confidence level
        if confidence > 0.8:
            self.confidence_progress.configure(progress_color="#4CAF50")  # Green for high confidence
        elif confidence > 0.5:
            self.confidence_progress.configure(progress_color="#FFC107")  # Yellow for medium confidence
        else:
            self.confidence_progress.configure(progress_color="#F44336")  # Red for low confidence
    
    def record_audio(self):
        """Record audio for the specified duration"""
        try:
            self.update_status("Recording...", "#E53935")
            # Record audio
            audio_data = sd.rec(int(self.sample_rate * self.duration), 
                               samplerate=self.sample_rate, 
                               channels=1, 
                               dtype='float32')
            sd.wait()  # Wait until recording is done
            
            # Save to temp file
            sf.write(self.temp_audio_file, audio_data, self.sample_rate)
            self.update_status("Processing...", "#2196F3")
            
            # Predict letter
            letter, confidence = predict_letter(self.temp_audio_file)
            
            # Update UI with results
            if letter:
                self.letter_label.configure(text=letter)
                self.update_confidence_display(confidence)
                self.update_status("Ready", "#4CAF50")
            else:
                self.letter_label.configure(text="?")
                self.update_confidence_display(None)
                self.update_status("Failed to recognize", "#FF9800")
        except Exception as e:
            print(f"Error during recording: {e}")
            self.update_status(f"Error: {str(e)[:30]}...", "#F44336")
        finally:
            self.recording = False
            self.record_button.configure(state="normal")
    
    def start_recording(self):
        """Start recording in a separate thread"""
        if not self.recording:
            self.recording = True
            self.record_button.configure(state="disabled")
            threading.Thread(target=self.record_audio, daemon=True).start()
    
    def upload_audio(self):
        """Allow user to upload an audio file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=(("Audio files", "*.wav *.mp3 *.ogg"), ("All files", "*.*"))
            )
            
            if file_path:
                self.update_status("Processing uploaded file...", "#2196F3")
                
                # Run prediction in a separate thread to keep UI responsive
                def process_file():
                    try:
                        # Predict letter from uploaded file
                        letter, confidence = predict_letter(file_path)
                        
                        # Update UI with results
                        if letter:
                            self.letter_label.configure(text=letter)
                            self.update_confidence_display(confidence)
                            self.update_status(f"Analyzed: {os.path.basename(file_path)}", "#4CAF50")
                        else:
                            self.letter_label.configure(text="?")
                            self.update_confidence_display(None)
                            self.update_status("Failed to recognize audio", "#FF9800")
                    except Exception as e:
                        print(f"Error processing file: {e}")
                        self.update_status(f"Error: {str(e)[:30]}...", "#F44336")
                
                threading.Thread(target=process_file, daemon=True).start()
        except Exception as e:
            print(f"Error during file upload: {e}")
            self.update_status(f"Error: {str(e)[:30]}...", "#F44336")

def main():
    # Create and run the application
    root = ctk.CTk()
    app = ArabicLetterRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()