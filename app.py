from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load trained CNN model
cnn_model = load_model("cnn_fashion_model.h5")

# Fashion-MNIST class names
class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Home route: render upload page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Health check route
@app.route("/health")
def health():
    return "ok", 200

# Prediction route: returns JSON
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Decode image using OpenCV
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (28, 28))
        img_norm = img_resized.astype("float32") / 255.0
        img_input = img_norm.reshape(1, 28, 28, 1)

        # Predict
        preds = cnn_model.predict(img_input)
        pred_label = np.argmax(preds, axis=1)[0]

        return jsonify({
            "prediction": class_names[pred_label],
            "probabilities": preds.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: Web page form submission route
@app.route("/upload", methods=["POST"])
def upload_predict():
    prediction = None
    file_uploaded = False

    file = request.files.get("file")
    if file:
        file_uploaded = True
        # Decode image using OpenCV
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (28, 28))
        img_norm = img_resized.astype("float32") / 255.0
        img_input = img_norm.reshape(1, 28, 28, 1)

        # Predict
        preds = cnn_model.predict(img_input)
        pred_label = np.argmax(preds, axis=1)[0]
        prediction = class_names[pred_label]

    return render_template(
        "index.html",
        prediction=prediction,
        file_uploaded=file_uploaded
    )

if __name__ == "__main__":
    app.run(debug=True)
