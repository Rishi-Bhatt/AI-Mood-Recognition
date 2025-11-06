# AI Mood Recognition

## Project Description

This project implements an AI-powered mood recognition system. It utilizes a deep learning model to detect and classify human emotions from facial expressions.

## Local Setup

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/AI_Mood_Recognition.git
    cd AI_Mood_Recognition
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    First, create a `requirements.txt` file with the following content (if it doesn't already exist):

    ```
    Flask
    opencv-python
    numpy
    speechrecognition
    librosa
    nltk
    transformers
    tensorflow
    huggingface_hub
    ```

    Then, install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Hugging Face Models:**

    Run the following script to download the necessary models from Hugging Face:

    ```bash
    python download_models.py
    ```

    This script will download `expressiondetector_modern.keras` and ensure the `distilbert-base-uncased-emotion` model is downloaded and cached by the `transformers` library.

5.  **Run the application:**

    Depending on the main entry point, you would typically run:

    ```bash
    python app.py
    # or
    python Flask_dashboard.py
    ```

## Hugging Face Model Usage

This section provides alternative ways to directly use the models. The project's `app.py` already handles the loading of these models as described above.

**Hugging Face Model Links:**
*   Expression Detector: [21f1000330/AI-Mood-Recognition](https://huggingface.co/21f1000330/AI-Mood-Recognition)
*   DistilBERT Emotion Classifier: [21f1000330/distilbert-base-uncased-emotion](https://huggingface.co/21f1000330/distilbert-base-uncased-emotion)

### How to use the Models Directly

#### 1. DistilBERT Emotion Classifier (Text Emotion)

You can leverage the `transformers` library for the text classification model:

```python
from transformers import pipeline

# Load the emotion classification pipeline for text
emotion_classifier = pipeline("text-classification", model="21f1000330/distilbert-base-uncased-emotion", top_k=1)

# Example usage
text_input = "I am feeling very happy today!"
result = emotion_classifier(text_input)
print(result)
```

#### 2. Keras Expression Detector (Facial Expression)

For the Keras facial expression detection model, you can load it directly using `tensorflow.keras` from its local path after downloading:

```python
import tensorflow as tf

# Load the Keras model directly from the local file
model = tf.keras.models.load_model("expressiondetector_modern.keras", compile=False)

# Example inference (assuming preprocessed_image is a 48x48 grayscale numpy array)
# predictions = model.predict(preprocessed_image)
# print(predictions)
```
