from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
cnn_model = load_model("cnn_fashion_model.h5")

# Fashion-MNIST class names
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

@app.route("/")
def home():
    return render_template("index.html")  # webpage to upload/test images

@app.route("/health")
def health():
    return "ok", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Received /predict request")
        file = request.files["file"]
        print("File received")
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        print("Image decoded")
        img_resized = cv2.resize(img, (28, 28))
        print("Image resized")
        img_norm = img_resized.astype("float32") / 255.0
        img_input = img_norm.reshape(1, 28, 28, 1)
        print("Image preprocessed")

        preds = cnn_model.predict(img_input)
        print("Prediction made")
        pred_label = np.argmax(preds, axis=1)[0]

        return jsonify({
            "prediction": class_names[pred_label],
            "probabilities": preds.tolist()
        })
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)