from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import pickle
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- Model Loading ---
# Load the pickled model. Ensure your file is named 'fashion_model.pkl'
# and is in the same directory as this app.py file.
try:
    with open('fashion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logging.info("Pickle model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file 'fashion_model.pkl' not found. Please check the filename and path.")
    model = None
except Exception as e:
    logging.error(f"Error loading pickle model: {e}")
    model = None

# Fashion-MNIST class names (must be in the same order as the model's training)
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

@app.route("/")
def home():
    """Renders the main page."""
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check endpoint."""
    return "ok", 200

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the image prediction request."""
    if model is None:
        return jsonify({"error": "Model is not loaded or failed to load. Check server logs."}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files["file"]

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # --- Image Preprocessing ---
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img_grayscale = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img_grayscale is None:
            return jsonify({"error": "Could not decode image. It might be corrupted or in an unsupported format."}), 400
        
        img_resized = cv2.resize(img_grayscale, (28, 28))
        img_inverted = cv2.bitwise_not(img_resized)
        img_norm = img_inverted.astype("float32") / 255.0

        # ✨ KEY CHANGE: Flatten the 28x28 image into a 1D array of 784 pixels
        # The model expects a 2D array of shape (n_samples, n_features)
        # For one image, this is (1, 784)
        img_input = img_norm.reshape(1, -1) # -1 automatically calculates it as 784
        logging.info(f"Image preprocessed and reshaped to: {img_input.shape}")

        # --- Prediction ---
        # Use predict_proba to get probabilities for all classes
        probabilities = model.predict_proba(img_input)
        
        # Get the name of the class with the highest probability
        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = class_names[predicted_class_index]
        
        logging.info(f"Prediction successful. Class: {predicted_class_name}")

        return jsonify({
            "prediction": predicted_class_name,
            "probabilities": probabilities.tolist() # Convert numpy array to a standard list for JSON
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # This is for local development. For production (like on Render), use Gunicorn.
    app.run(debug=True)