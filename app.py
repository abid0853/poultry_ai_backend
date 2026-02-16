import os
from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np

# --- 🟢 RENDER OPTIMIZATION ---
# We remove the Windows-specific FFmpeg path. 
# On Render (Linux), FFmpeg is installed globally and found automatically.
# ------------------------------

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'uploads'
MODEL_FILE = "poultry_model.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable for the model
model = None

def load_model():
    """Load the trained model on startup"""
    global model
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Warning: poultry_model.pkl not found. Ensure it is uploaded to GitHub!")

def extract_features(file_path):
    """Audio feature extraction using librosa"""
    try:
        # Load audio using default backend (FFmpeg on Render)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Librosa Error: {e}")
        return None

# Load model when app starts
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save to the temporary uploads folder on Render
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        if model is None:
            return jsonify({"status": "Model Not Found", "confidence": "0%"})

        features = extract_features(filepath)
        
        if features is None:
            return jsonify({"error": "Audio processing failed. Check FFmpeg logs."}), 500

        # AI Prediction
        prediction = model.predict([features])[0] 
        probabilities = model.predict_proba([features])[0]
        
        is_sick = (prediction == 1)
        confidence_score = probabilities[prediction] * 100

        result = {
            "status": "Respiratory Infection" if is_sick else "Healthy",
            "confidence": f"{confidence_score:.1f}%",
            "risk_level": "High" if is_sick else "Low"
        }

        # Cleanup: Remove the file after processing to save disk space on Render
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use environment port for Render compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)