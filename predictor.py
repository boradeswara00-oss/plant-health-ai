
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_model_v2.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names_v2.json"

IMAGE_SIZE = (224, 224)

# Load the trained model once.
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)


def predict_plant_disease(image):
    """Predict a plant disease from a PIL image."""

    # Convert the uploaded image to RGB.
    image = image.convert("RGB")

    # Resize the image to match the model input.
    image = image.resize(IMAGE_SIZE)

    # Convert the image to a NumPy array.
    image_array = np.asarray(image, dtype=np.float32)

    # Add the batch dimension: (224, 224, 3) -> (1, 224, 224, 3).
    image_array = np.expand_dims(image_array, axis=0)

    # Run the model once.
    predictions = model.predict(image_array, verbose=0)[0]

    # Get the three highest-scoring classes.
    top_indices = np.argsort(predictions)[::-1][:3]

    predicted_index = int(top_indices[0])
    predicted_disease = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    # Prepare the top three predictions for the app.
    top_predictions = [
        {
            "disease": class_names[int(index)],
            "confidence": float(predictions[index]) * 100,
        }
        for index in top_indices
    ]

    return predicted_disease, confidence, top_predictions
