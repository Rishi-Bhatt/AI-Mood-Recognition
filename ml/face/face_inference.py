import numpy as np
import cv2
from tf_keras.models import load_model
import joblib

MODEL_PATH = "models/expressiondetector_modern.keras"
ENCODER_PATH = "models/label_encoder.pkl"

model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = img[..., np.newaxis]
    img = np.expand_dims(img, axis=0)
    return img

def predict_emotion(image):
    img = preprocess_image(image)
    probs = model.predict(img)[0]
    pred_idx = np.argmax(probs)
    emotion = label_encoder.inverse_transform([pred_idx])[0]
    return emotion, probs
