from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json, io, threading, time
import requests as req_lib

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("crop_disease_model.h5")
with open("model_config.json") as f:
    cfg = json.load(f)

CLASS_NAMES = cfg["class_names"]
IMG_SIZE = cfg["image_size"]

# ✅ Sabhi 49 EXACT class names — model_config.json se match
TREATMENTS = {
    "Apple___Apple_scab": {
        "treatment": "Apply Captan or Myclobutanil fungicide spray every 7-10 days",
        "treatment_hindi": "हर 7-10 दिन में कैप्टन या माइक्लोब्यूटेनिल फफूंदनाशक स्प्रे करें",
        "severity": "High"
    },
    "Apple___Black_rot": {
        "treatment": "Remove infected fruit and branches, apply Captan fungicide",
        "treatment_hindi": "संक्रमित फल और शाखाएं हटाएं, कैप्टन फफूंदनाशक लगाएं",
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "treatment": "Apply Myclobutanil fungicide spray, remove nearby cedar trees",
        "treatment_hindi": "माइक्लोब्यूटेनिल फफूंदनाशक स्प्रे करें, पास के देवदार पेड़ हटाएं",
        "severity": "Medium"
    },
    "Apple___healthy": {
        "treatment": "Plant is healthy! Maintain regular pruning and watering.",
        "treatment_hindi": "पौधा स्वस्थ है! नियमित छंटाई और पानी देते रहें।",
        "severity": "None"
    },
    "Blueberry___healthy": {
        "treatment": "Plant is healthy! Ensure acidic soil pH 4.5-5.5.",
        "treatment_hindi": "पौधा स्वस्थ है! मिट्टी का pH 4.5-5.5 बनाए रखें।",
        "severity": "None"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "treatment": "Apply Sulfur or Potassium bicarbonate fungicide spray",
        "treatment_hindi": "सल्फर या पोटेशियम बाइकार्बोनेट फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Cherry_(including_sour)___healthy": {
        "treatment": "Plant is healthy! Ensure proper sunlight and drainage.",
        "treatment_hindi": "पौधा स्वस्थ है! उचित धूप और जल निकासी सुनिश्चित करें।",
        "severity": "None"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "treatment": "Apply Azoxystrobin or Pyraclostrobin fungicide spray",
        "treatment_hindi": "एज़ोक्सीस्ट्रोबिन या पाइराक्लोस्ट्रोबिन फफूंदनाशक स्प्रे करें",
        "severity": "High"
    },
    "Corn_(maize)___Common_rust_": {
        "treatment": "Apply Propiconazole or Azoxystrobin fungicide",
        "treatment_hindi": "प्रोपिकोनाज़ोल या एज़ोक्सीस्ट्रोबिन फफूंदनाशक लगाएं",
        "severity": "Medium"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "treatment": "Apply Mancozeb or Azoxystrobin fungicide spray",
        "treatment_hindi": "मैंकोज़ेब या एज़ोक्सीस्ट्रोबिन फफूंदनाशक स्प्रे करें",
        "severity": "High"
    },
    "Corn_(maize)___healthy": {
        "treatment": "Plant is healthy! Continue good farming practices.",
        "treatment_hindi": "पौधा स्वस्थ है! अच्छी खेती जारी रखें।",
        "severity": "None"
    },
    "Grape___Black_rot": {
        "treatment": "Apply Myclobutanil or Mancozeb fungicide, remove mummified fruit",
        "treatment_hindi": "माइक्लोब्यूटेनिल या मैंकोज़ेब फफूंदनाशक लगाएं, सूखे फल हटाएं",
        "severity": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "treatment": "Prune infected wood, consult Krishi Kendra for treatment",
        "treatment_hindi": "संक्रमित लकड़ी काटें, कृषि केंद्र से परामर्श लें",
        "severity": "High"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "treatment": "Apply Copper oxychloride or Mancozeb fungicide",
        "treatment_hindi": "कॉपर ऑक्सीक्लोराइड या मैंकोज़ेब फफूंदनाशक लगाएं",
        "severity": "Medium"
    },
    "Grape___healthy": {
        "treatment": "Plant is healthy! Maintain proper trellising and pruning.",
        "treatment_hindi": "पौधा स्वस्थ है! उचित जाली और छंटाई बनाए रखें।",
        "severity": "None"
    },
    "Mango_Anthracnose": {
        "treatment": "Apply Carbendazim or Mancozeb fungicide spray",
        "treatment_hindi": "कार्बेन्डाज़िम या मैंकोज़ेब फफूंदनाशक स्प्रे करें",
        "severity": "High"
    },
    "Mango_Bacterial Canker": {
        "treatment": "Apply Copper oxychloride spray, prune infected branches",
        "treatment_hindi": "कॉपर ऑक्सीक्लोराइड स्प्रे करें, संक्रमित शाखाएं काटें",
        "severity": "High"
    },
    "Mango_Cutting Weevil": {
        "treatment": "Apply Chlorpyrifos or Carbaryl insecticide spray",
        "treatment_hindi": "क्लोरपायरीफॉस या कार्बेरिल कीटनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Mango_Die Back": {
        "treatment": "Prune infected branches, apply Copper oxychloride paste",
        "treatment_hindi": "संक्रमित शाखाएं काटें, कॉपर ऑक्सीक्लोराइड पेस्ट लगाएं",
        "severity": "High"
    },
    "Mango_Gall Midge": {
        "treatment": "Apply Dimethoate or Imidacloprid insecticide spray",
        "treatment_hindi": "डाइमेथोएट या इमिडाक्लोप्रिड कीटनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Mango_Healthy": {
        "treatment": "Plant is healthy! Water and fertilize regularly.",
        "treatment_hindi": "पौधा स्वस्थ है! नियमित पानी और खाद दें।",
        "severity": "None"
    },
    "Mango_Powdery Mildew": {
        "treatment": "Apply Sulfur or Hexaconazole fungicide spray",
        "treatment_hindi": "सल्फर या हेक्साकोनाज़ोल फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Mango_Sooty Mould": {
        "treatment": "Control insects with Imidacloprid, wash leaves with water",
        "treatment_hindi": "इमिडाक्लोप्रिड से कीट नियंत्रित करें, पत्तियां पानी से धोएं",
        "severity": "Medium"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "treatment": "Remove infected trees, control psyllid with Imidacloprid",
        "treatment_hindi": "संक्रमित पेड़ हटाएं, इमिडाक्लोप्रिड से सिला कीट नियंत्रित करें",
        "severity": "High"
    },
    "Peach___Bacterial_spot": {
        "treatment": "Apply Copper hydroxide spray, avoid overhead irrigation",
        "treatment_hindi": "कॉपर हाइड्रॉक्साइड स्प्रे करें, ऊपर से सिंचाई बंद करें",
        "severity": "High"
    },
    "Peach___healthy": {
        "treatment": "Plant is healthy! Ensure well-drained soil.",
        "treatment_hindi": "पौधा स्वस्थ है! अच्छी जल निकासी वाली मिट्टी सुनिश्चित करें।",
        "severity": "None"
    },
    "Pepper,_bell___Bacterial_spot": {
        "treatment": "Apply Copper-based bactericide, use disease-free seeds",
        "treatment_hindi": "कॉपर आधारित जीवाणुनाशक लगाएं, रोगमुक्त बीज उपयोग करें",
        "severity": "High"
    },
    "Pepper,_bell___healthy": {
        "treatment": "Plant is healthy! Maintain proper watering schedule.",
        "treatment_hindi": "पौधा स्वस्थ है! उचित पानी देने का समय बनाए रखें।",
        "severity": "None"
    },
    "Potato___Early_blight": {
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide spray",
        "treatment_hindi": "मैंकोज़ेब या क्लोरोथालोनिल फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Potato___Late_blight": {
        "treatment": "Apply Metalaxyl + Mancozeb spray immediately, destroy infected plants",
        "treatment_hindi": "तुरंत मेटालैक्सिल + मैंकोज़ेब स्प्रे करें, संक्रमित पौधे नष्ट करें",
        "severity": "High"
    },
    "Potato___healthy": {
        "treatment": "Plant is healthy! Keep soil moist and weed-free.",
        "treatment_hindi": "पौधा स्वस्थ है! मिट्टी नम और खरपतवार मुक्त रखें।",
        "severity": "None"
    },
    "Raspberry___healthy": {
        "treatment": "Plant is healthy! Prune old canes after harvest.",
        "treatment_hindi": "पौधा स्वस्थ है! फसल के बाद पुरानी शाखाएं काटें।",
        "severity": "None"
    },
    "Rice_Bacterial leaf blight": {
        "treatment": "Apply Copper oxychloride, drain field water completely",
        "treatment_hindi": "कॉपर ऑक्सीक्लोराइड लगाएं, खेत का पानी पूरी तरह निकालें",
        "severity": "High"
    },
    "Rice_Brown spot": {
        "treatment": "Apply Mancozeb or Iprodione fungicide spray",
        "treatment_hindi": "मैंकोज़ेब या आइप्रोडायोन फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Rice_Leaf smut": {
        "treatment": "Use disease-free seeds, apply Carbendazim seed treatment",
        "treatment_hindi": "रोगमुक्त बीज उपयोग करें, कार्बेन्डाज़िम बीज उपचार करें",
        "severity": "Medium"
    },
    "Soybean___healthy": {
        "treatment": "Plant is healthy! Maintain proper crop rotation.",
        "treatment_hindi": "पौधा स्वस्थ है! उचित फसल चक्र बनाए रखें।",
        "severity": "None"
    },
    "Squash___Powdery_mildew": {
        "treatment": "Apply Sulfur or Neem oil spray every 7 days",
        "treatment_hindi": "हर 7 दिन में सल्फर या नीम तेल स्प्रे करें",
        "severity": "Medium"
    },
    "Strawberry___Leaf_scorch": {
        "treatment": "Apply Captan fungicide, remove and destroy infected leaves",
        "treatment_hindi": "कैप्टन फफूंदनाशक लगाएं, संक्रमित पत्तियां हटाकर नष्ट करें",
        "severity": "Medium"
    },
    "Strawberry___healthy": {
        "treatment": "Plant is healthy! Ensure good air circulation.",
        "treatment_hindi": "पौधा स्वस्थ है! अच्छा वायु संचार सुनिश्चित करें।",
        "severity": "None"
    },
    "Tomato___Bacterial_spot": {
        "treatment": "Apply Copper hydroxide spray, avoid overhead irrigation",
        "treatment_hindi": "कॉपर हाइड्रॉक्साइड स्प्रे करें, ऊपर से सिंचाई बंद करें",
        "severity": "High"
    },
    "Tomato___Early_blight": {
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide spray",
        "treatment_hindi": "मैंकोज़ेब या क्लोरोथालोनिल फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Tomato___Late_blight": {
        "treatment": "Apply Metalaxyl + Mancozeb spray, remove infected plants immediately",
        "treatment_hindi": "मेटालैक्सिल + मैंकोज़ेब स्प्रे करें, तुरंत संक्रमित पौधे हटाएं",
        "severity": "High"
    },
    "Tomato___Leaf_Mold": {
        "treatment": "Apply Chlorothalonil fungicide, improve ventilation",
        "treatment_hindi": "क्लोरोथालोनिल फफूंदनाशक लगाएं, वेंटिलेशन बढ़ाएं",
        "severity": "Medium"
    },
    "Tomato___Septoria_leaf_spot": {
        "treatment": "Apply Copper-based fungicide, avoid overhead watering",
        "treatment_hindi": "कॉपर आधारित फफूंदनाशक लगाएं, ऊपर से पानी देना बंद करें",
        "severity": "Medium"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "treatment": "Apply Abamectin or Neem oil spray, increase humidity",
        "treatment_hindi": "अबामेक्टिन या नीम तेल स्प्रे करें, नमी बढ़ाएं",
        "severity": "Medium"
    },
    "Tomato___Target_Spot": {
        "treatment": "Apply Azoxystrobin or Chlorothalonil fungicide spray",
        "treatment_hindi": "एज़ोक्सीस्ट्रोबिन या क्लोरोथालोनिल फफूंदनाशक स्प्रे करें",
        "severity": "Medium"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "treatment": "Remove infected plants, control whitefly with Imidacloprid spray",
        "treatment_hindi": "संक्रमित पौधे हटाएं, इमिडाक्लोप्रिड से सफेद मक्खी नियंत्रित करें",
        "severity": "High"
    },
    "Tomato___Tomato_mosaic_virus": {
        "treatment": "Remove infected plants, disinfect tools, control aphids with Dimethoate",
        "treatment_hindi": "संक्रमित पौधे हटाएं, औजार साफ करें, डाइमेथोएट से माहू नियंत्रित करें",
        "severity": "High"
    },
    "Tomato___healthy": {
        "treatment": "Plant is healthy! Keep watering regularly.",
        "treatment_hindi": "पौधा स्वस्थ है! नियमित पानी देते रहें।",
        "severity": "None"
    },
}

DEFAULT_TREATMENT = {
    "treatment": "Contact your local Krishi Kendra for expert advice",
    "treatment_hindi": "विशेषज्ञ सलाह के लिए अपने स्थानीय कृषि केंद्र से संपर्क करें",
    "severity": "Unknown"
}

# ✅ Keep-Alive — server har 14 min mein khud ko jagata rahega
def keep_alive():
    time.sleep(60)  # startup ke baad 1 min wait karo
    while True:
        try:
            req_lib.get("https://kishanmitraai.onrender.com/api/health", timeout=10)
            print("✅ Keep-alive ping sent")
        except Exception as e:
            print(f"Keep-alive failed: {e}")
        time.sleep(840)  # 14 minutes

threading.Thread(target=keep_alive, daemon=True).start()


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
    # ✅ "image" aur "file" dono accept karta hai
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
            "is_healthy": "healthy" in disease.lower() or "Healthy" in disease
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "classes": len(CLASS_NAMES)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
