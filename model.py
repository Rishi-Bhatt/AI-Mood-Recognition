import tensorflow as tf
import numpy as np

class EmotionDetector:
    def __init__(self):
        self.model = tf.keras.models.load_model("expressiondetector_modern.keras", compile=False)
        self.labels = {
            0: 'angry', 1: 'disgust', 2: 'fear',
            3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
        }

    def predict(self, image_data):
        # image_data is expected to be a numpy array, e.g., from an image file
        # Resize and preprocess the image as required by your model
        image = np.array(image_data).reshape(1, 48, 48, 1) / 255.0
        predictions = self.model.predict(image)
        emotion = self.labels[predictions.argmax()]
        return {"emotion": emotion}
