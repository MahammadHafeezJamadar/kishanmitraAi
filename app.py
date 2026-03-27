from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json, io, threading, time
import requests as req_lib
import os  # <-- Port ke liye zaroori hai

app = Flask(__name__)

# CORS ko properly configure kiya taaki Lovable se connect ho sake
CORS(app, resources={r"/*": {"origins": "*"}})

# Model loading ko try-except mein dala taaki server crash na ho
try:
    model = tf.keras.models.load_model("crop_disease_model.h5")
    with open("model_config.json") as f:
        cfg = json.load(f)
    CLASS_NAMES = cfg["class_names"]
    IMG_SIZE = cfg["image_size"]
    print("✅ Model and Config loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ... (Tera TREATMENTS wala part yahan same rahega, use change mat karna) ...

# ✅ Keep-Alive Fix: Ismein domain name change karne ki zaroorat nahi padegi
def keep_alive():
    time.sleep(60)
    while True:
        try:
            # Localhost ping is better for internal keep-alive
            req_lib.get("http://localhost:10000/api/health", timeout=10)
            print("✅ Internal Keep-alive ping sent")
        except:
            pass
        time.sleep(600) # 10 minutes

threading.Thread(target=keep_alive, daemon=True).start()

@app.route("/")
def home():
    return jsonify({
        "message": "KrishiMitra AI Backend is running! 🌱",
        "status": "ok"
    })

@app.route("/api/predict", methods=["POST", "OPTIONS"]) # OPTIONS header for CORS
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    # Frontend se 'image' ya 'file' kuch bhi aaye, ye handle kar lega
    file = request.files.get("image") or request.files.get("file")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Preprocessing
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Prediction
        pred = model.predict(arr)[0]
        top_idx = int(np.argmax(pred))
        
        # Index string hai ya int, config ke hisaab se check karega
        disease = CLASS_NAMES.get(str(top_idx)) or CLASS_NAMES.get(top_idx)
        confidence = round(float(pred[top_idx]) * 100, 2)

        t = TREATMENTS.get(disease, DEFAULT_TREATMENT)

        return jsonify({
            "status": "success",
            "disease": disease,
            "confidence": confidence,
            "treatment": t["treatment"],
            "treatment_hindi": t["treatment_hindi"],
            "severity": t["severity"]
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # 🚨 RENDER FIX: Render hamesha port badalta rehta hai
    # Isliye os.environ use karna compulsory hai
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
