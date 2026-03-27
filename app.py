from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json, io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("crop_disease_model.h5")
with open("model_config.json") as f:
    cfg = json.load(f)

CLASS_NAMES = cfg["class_names"]
IMG_SIZE = cfg["image_size"]

TREATMENTS = {
    "Tomato___Early_blight": {"treatment": "Mancozeb spray karo", "treatment_hindi": "मैंकोज़ेब स्प्रे करें", "severity": "Medium"},
    "Tomato___healthy": {"treatment": "Plant healthy hai!", "treatment_hindi": "पौधा स्वस्थ है!", "severity": "None"},
    "Rice_Leaf Blast": {"treatment": "Tricyclazole spray karo", "treatment_hindi": "ट्राइसाइक्लाज़ोल स्प्रे करें", "severity": "High"},
    "Mango_Anthracnose": {"treatment": "Carbendazim spray karo", "treatment_hindi": "कार्बेन्डाज़िम स्प्रे करें", "severity": "High"},
}

@app.route("/")
def home():
    return jsonify({
        "message": "KrishiMitra AI Backend is running! 🌱",
        "status": "ok",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict (POST)"
        }
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    # ✅ Bug 1 Fix: dono key accept karo
    file = request.files.get("image") or request.files.get("file")
    if not file:
        return jsonify({"error": "No image"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(np.array(img) / 255.0, 0)
        pred = model.predict(arr)[0]

        top_idx = int(np.argsort(pred)[::-1][0])
        disease = CLASS_NAMES[str(top_idx)]
        confidence = round(float(pred[top_idx]) * 100, 2)
        t = TREATMENTS.get(disease, {
            "treatment": "Krishi Kendra se sampark karo",
            "treatment_hindi": "कृषि केंद्र से संपर्क करें",
            "severity": "Unknown"
        })

        # ✅ Bug 2 Fix: frontend jo expect karta woh format
        return jsonify({
            "status": "success",
            "disease": disease,
            "disease_hindi": disease.replace("_", " "),
            "confidence": confidence,
            "treatment": t["treatment"],
            "treatment_hindi": t["treatment_hindi"],
            "severity": t["severity"],
            "is_healthy": "healthy" in disease.lower()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "classes": len(CLASS_NAMES)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
