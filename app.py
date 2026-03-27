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
    "Tomato___Early_blight": {
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide spray",
        "treatment_hindi": "मैंकोज़ेब या क्लोरोथालोनिल फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Tomato___Late_blight": {
        "treatment": "Apply Metalaxyl + Mancozeb spray, remove infected leaves",
        "treatment_hindi": "मेटालैक्सिल + मैंकोज़ेब स्प्रे करें, संक्रमित पत्तियां हटाएं",
        "severity": "High"
    },
    "Tomato___healthy": {
        "treatment": "Plant is healthy! Keep watering regularly.",
        "treatment_hindi": "पौधा स्वस्थ है! नियमित पानी देते रहें।",
        "severity": "None"
    },
    "Tomato___Leaf_Miner": {
        "treatment": "Use Spinosad or Abamectin insecticide spray",
        "treatment_hindi": "स्पिनोसैड या अबामेक्टिन कीटनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Tomato___Septoria_leaf_spot": {
        "treatment": "Apply Copper-based fungicide, avoid overhead watering",
        "treatment_hindi": "कॉपर आधारित फफूंदनाशक लगाएं, ऊपर से पानी देना बंद करें",
        "severity": "Medium"
    },
    "Tomato___Spider_mites": {
        "treatment": "Apply Abamectin or Neem oil spray",
        "treatment_hindi": "अबामेक्टिन या नीम तेल स्प्रे करें",
        "severity": "Medium"
    },
    "Tomato___Target_Spot": {
        "treatment": "Apply Azoxystrobin fungicide spray",
        "treatment_hindi": "एज़ोक्सीस्ट्रोबिन फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "treatment": "Remove infected plants, control whitefly with Imidacloprid",
        "treatment_hindi": "संक्रमित पौधे हटाएं, इमिडाक्लोप्रिड से सफेद मक्खी नियंत्रित करें",
        "severity": "High"
    },
    "Tomato___Tomato_mosaic_virus": {
        "treatment": "Remove infected plants, disinfect tools, control aphids",
        "treatment_hindi": "संक्रमित पौधे हटाएं, औजार साफ करें, माहू नियंत्रित करें",
        "severity": "High"
    },
    "Tomato___Bacterial_spot": {
        "treatment": "Apply Copper hydroxide spray, avoid overhead irrigation",
        "treatment_hindi": "कॉपर हाइड्रॉक्साइड स्प्रे करें, ऊपर से सिंचाई बंद करें",
        "severity": "High"
    },
    "Rice_Leaf Blast": {
        "treatment": "Apply Tricyclazole or Isoprothiolane fungicide",
        "treatment_hindi": "ट्राइसाइक्लाज़ोल या आइसोप्रोथायोलेन फफूंदनाशक स्प्रे करें",
        "severity": "High"
    },
    "Rice_Neck Blast": {
        "treatment": "Spray Tricyclazole at booting stage, use resistant varieties",
        "treatment_hindi": "बूटिंग चरण में ट्राइसाइक्लाज़ोल स्प्रे करें, प्रतिरोधी किस्में उगाएं",
        "severity": "High"
    },
    "Rice_Brown Spot": {
        "treatment": "Apply Mancozeb or Iprodione fungicide spray",
        "treatment_hindi": "मैंकोज़ेब या आइप्रोडायोन फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Rice_Sheath Blight": {
        "treatment": "Apply Hexaconazole or Propiconazole fungicide",
        "treatment_hindi": "हेक्साकोनाज़ोल या प्रोपिकोनाज़ोल फफूंदनाशक लगाएं",
        "severity": "High"
    },
    "Rice_Bacterial Leaf Blight": {
        "treatment": "Apply Copper oxychloride, drain field water",
        "treatment_hindi": "कॉपर ऑक्सीक्लोराइड लगाएं, खेत का पानी निकालें",
        "severity": "High"
    },
    "Rice_healthy": {
        "treatment": "Plant is healthy! Continue proper fertilization.",
        "treatment_hindi": "पौधा स्वस्थ है! उचित उर्वरक देते रहें।",
        "severity": "None"
    },
    "Mango_Anthracnose": {
        "treatment": "Apply Carbendazim or Mancozeb fungicide spray",
        "treatment_hindi": "कार्बेन्डाज़िम या मैंकोज़ेब फफूंदनाशक स्प्रे करें",
        "severity": "High"
    },
    "Mango_Powdery_Mildew": {
        "treatment": "Apply Sulfur or Hexaconazole fungicide spray",
        "treatment_hindi": "सल्फर या हेक्साकोनाज़ोल फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Mango_Sooty_Mould": {
        "treatment": "Control insects with Imidacloprid, wash leaves with water",
        "treatment_hindi": "इमिडाक्लोप्रिड से कीट नियंत्रित करें, पत्तियां पानी से धोएं",
        "severity": "Medium"
    },
    "Mango_Die_Back": {
        "treatment": "Prune infected branches, apply Copper oxychloride paste",
        "treatment_hindi": "संक्रमित शाखाएं काटें, कॉपर ऑक्सीक्लोराइड पेस्ट लगाएं",
        "severity": "High"
    },
    "Mango_healthy": {
        "treatment": "Plant is healthy! Water and fertilize regularly.",
        "treatment_hindi": "पौधा स्वस्थ है! नियमित पानी और खाद दें।",
        "severity": "None"
    },
    "Wheat_Leaf_Rust": {
        "treatment": "Apply Propiconazole or Tebuconazole fungicide",
        "treatment_hindi": "प्रोपिकोनाज़ोल या टेबुकोनाज़ोल फफूंदनाशक लगाएं",
        "severity": "High"
    },
    "Wheat_Yellow_Rust": {
        "treatment": "Apply Propiconazole fungicide spray immediately",
        "treatment_hindi": "तुरंत प्रोपिकोनाज़ोल फफूंदनाशक स्प्रे करें",
        "severity": "High"
    },
    "Wheat_Stem_Rust": {
        "treatment": "Apply Mancozeb or Propiconazole fungicide",
        "treatment_hindi": "मैंकोज़ेब या प्रोपिकोनाज़ोल फफूंदनाशक लगाएं",
        "severity": "High"
    },
    "Wheat_healthy": {
        "treatment": "Plant is healthy! Ensure proper irrigation.",
        "treatment_hindi": "पौधा स्वस्थ है! उचित सिंचाई सुनिश्चित करें।",
        "severity": "None"
    },
    "Corn___Common_rust": {
        "treatment": "Apply Azoxystrobin or Propiconazole fungicide",
        "treatment_hindi": "एज़ोक्सीस्ट्रोबिन या प्रोपिकोनाज़ोल फफूंदनाशक लगाएं",
        "severity": "Medium"
    },
    "Corn___Northern_Leaf_Blight": {
        "treatment": "Apply Mancozeb or Azoxystrobin fungicide spray",
        "treatment_hindi": "मैंकोज़ेब या एज़ोक्सीस्ट्रोबिन फफूंदनाशक स्प्रे करें",
        "severity": "High"
    },
    "Corn___healthy": {
        "treatment": "Plant is healthy! Continue good farming practices.",
        "treatment_hindi": "पौधा स्वस्थ है! अच्छी खेती जारी रखें।",
        "severity": "None"
    },
    "Potato___Early_blight": {
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide spray",
        "treatment_hindi": "मैंकोज़ेब या क्लोरोथालोनिल फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Potato___Late_blight": {
        "treatment": "Apply Metalaxyl + Mancozeb spray immediately",
        "treatment_hindi": "तुरंत मेटालैक्सिल + मैंकोज़ेब स्प्रे करें",
        "severity": "High"
    },
    "Potato___healthy": {
        "treatment": "Plant is healthy! Keep soil moist.",
        "treatment_hindi": "पौधा स्वस्थ है! मिट्टी नम रखें।",
        "severity": "None"
    },
}

DEFAULT_TREATMENT = {
    "treatment": "Contact your local Krishi Kendra for expert advice",
    "treatment_hindi": "विशेषज्ञ सलाह के लिए अपने स्थानीय कृषि केंद्र से संपर्क करें",
    "severity": "Unknown"
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
    # Accept both "image" and "file" keys
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

        t = TREATMENTS.get(disease, DEFAULT_TREATMENT)

        return jsonify({
            "status": "success",
            "disease": disease,
            "disease_hindi": disease.replace("___", " - ").replace("_", " "),
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
