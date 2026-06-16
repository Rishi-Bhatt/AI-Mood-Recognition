import numpy as np
from tf_keras.models import load_model
import joblib

def load_cnn_model(model_path="expressiondetector_modern.keras", encoder_path="label_encoder.pkl"):
    model = load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

def predict_emotion(image, model, label_encoder):
    img = image.astype('float32') / 255.0
    if img.ndim == 2:
        img = img[..., np.newaxis]
    img = np.expand_dims(img, axis=0)
    probs = model.predict(img)[0]
    pred_idx = np.argmax(probs)
    emotion = label_encoder.inverse_transform([pred_idx])[0]
    return emotion, probs
