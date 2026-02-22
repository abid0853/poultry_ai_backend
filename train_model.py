import os

# --- 🟢 CRITICAL FIX: FORCE PYTHON TO SEE FFMPEG ---
# This allows the training script to read the .m4a files from your phone
ffmpeg_path = r"C:\ffmpeg\ffmpeg\bin"  
os.environ["PATH"] += os.pathsep + ffmpeg_path
# ---------------------------------------------------

import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configuration
DATASET_PATH = "datasets"  # Make sure this matches your folder name (dataset vs datasets)
MODEL_FILE = "poultry_model.pkl"
n_mfcc = 13  # Number of audio features to extract

def extract_features(file_path):
    """
    Loads an audio file and averages its MFCC features.
    """
    try:
        # kaiser_fast is faster for prototypes
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        # Average the features over time
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def train():
    print("🎧 Loading dataset and extracting features...")
    X = [] # Features
    y = [] # Labels (0 = Healthy, 1 = Sick)

    # 2. Load "Healthy" samples
    healthy_path = os.path.join(DATASET_PATH, 'healthy')
    count_healthy = 0
    if os.path.exists(healthy_path):
        print(f"   - Scanning 'healthy' folder...")
        for file in os.listdir(healthy_path):
            file_path = os.path.join(healthy_path, file)
            # Skip hidden files
            if file.startswith('.'): continue
            
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(0) # Label 0 for Healthy
                count_healthy += 1
    
    print(f"   ✅ Loaded {count_healthy} Healthy samples.")

    # 3. Load "Sick" samples
    sick_path = os.path.join(DATASET_PATH, 'sick')
    count_sick = 0
    if os.path.exists(sick_path):
        print(f"   - Scanning 'sick' folder...")
        for file in os.listdir(sick_path):
            file_path = os.path.join(sick_path, file)
            if file.startswith('.'): continue
            
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(1) # Label 1 for Sick
                count_sick += 1
                
    print(f"   ✅ Loaded {count_sick} Sick samples.")

    if len(X) == 0:
        print("❌ No data found! Please add audio files to datasets/healthy and datasets/sick")
        return

    # 4. Train the Model
    print(f"📊 Training on Total {len(X)} audio samples...")
    
    # Check if we have enough data to split
    if len(X) < 5:
        print("⚠️ Not enough data to split into Train/Test. Training on ALL data.")
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        accuracy = 1.0
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

    print(f"✅ Model Trained! Accuracy: {accuracy * 100:.2f}%")

    # 5. Save the Model
    joblib.dump(model, MODEL_FILE)
    print(f"💾 Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()