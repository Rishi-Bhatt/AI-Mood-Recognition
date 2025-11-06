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

    ```bash
    pip install -r requirements.txt
    ```
    (Note: We will create the `requirements.txt` file in a later step.)

4.  **Run the application:**

    Depending on the main entry point, you would typically run:

    ```bash
    python app.py
    # or
    python Flask_dashboard.py
    ```

## Hugging Face Model Usage

This project utilizes an emotion detection model (`expressiondetector_modern.keras`) that is publicly available on Hugging Face. You can directly use this model for inference in your own applications.

**Hugging Face Model Link:** [21f1000330/AI-Mood-Recognition](https://huggingface.co/21f1000330/AI-Mood-Recognition)

### How to use the Model

To use the deployed model, you can leverage the `transformers` library:

```python
from transformers import pipeline

# Load the emotion classification pipeline using your model
emotion_classifier = pipeline("image-classification", model="21f1000330/AI-Mood-Recognition")

# Example usage (assuming you have an image file or numpy array)
# from PIL import Image
# image = Image.open("path/to/your/image.jpg")
# result = emotion_classifier(image)
# print(result)

# Note: The model expects a 48x48 grayscale image for input. 
# You may need to preprocess your images accordingly before passing them to the pipeline.
```

Alternatively, you can load the model directly using `tensorflow.keras` if you prefer:

```python
import tensorflow as tf

# Load the Keras model directly from Hugging Face
model = tf.keras.models.load_model("hf://21f1000330/AI-Mood-Recognition/expressiondetector_modern.keras", compile=False)

# Example inference (assuming preprocessed_image is a 48x48 grayscale numpy array)
# predictions = model.predict(preprocessed_image)
# print(predictions)
```
