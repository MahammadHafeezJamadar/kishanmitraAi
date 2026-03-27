
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
    "Tomato___Early_blight": {"treatment": "Mancozeb spray karo", "severity": "Medium"},
    "Tomato___healthy": {"treatment": "Plant healthy hai!", "severity": "None"},
    "Rice_Leaf Blast": {"treatment": "Tricyclazole spray karo", "severity": "High"},
    "Mango_Anthracnose": {"treatment": "Carbendazim spray karo", "severity": "High"},
}

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img)/255.0, 0)
    pred = model.predict(arr)[0]
    top3 = np.argsort(pred)[::-1][:3]
    out = []
    for idx in top3:
        d = CLASS_NAMES[str(idx)]
        t = TREATMENTS.get(d, {"treatment": "Krishi Kendra", "severity": "Unknown"})
        out.append({"disease": d, "confidence": round(float(pred[idx])*100,2),
                    "treatment": t["treatment"], "severity": t["severity"]})
    return jsonify({"status": "success", "predictions": out})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "classes": len(CLASS_NAMES)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
