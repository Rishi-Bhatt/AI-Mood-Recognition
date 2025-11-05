import sqlite3
import smtplib
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from datetime import datetime
from email.mime.text import MIMEText
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import speech_recognition as sr
import librosa
from tensorflow import keras
import tensorflow as tf


# Environment and TensorFlow Config

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 


# Flask app setup

app = Flask(__name__)


# Initialize NLTK + Sentiment Analyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


# Hugging Face Emotion Classifier

emotion_classifier = pipeline(
    "text-classification",
    model="21f1000330/distilbert-base-uncased-emotion",
    top_k=1
)


# Load Modern Converted Model

print("Loading modern Keras model...")
model = keras.models.load_model("expressiondetector_modern.keras", compile=False)
print("Model loaded successfully!")


# Face detection setup

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

task_assignment = {
    'angry': "Take deep breaths and relax.",
    'disgust': "Engage in a team discussion for clarity.",
    'fear': "Reassess task workload and provide support.",
    'happy': "Encourage collaboration and brainstorming.",
    'neutral': "Continue with assigned tasks normally.",
    'sad': "Assign lighter tasks or encourage social interaction.",
    'surprise': "Allow creative or unexpected tasks."
}


# Database utilities

def save_mood_to_db(emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mood_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                emotion TEXT
            )
        """)
        cursor.execute("INSERT INTO mood_log (timestamp, emotion) VALUES (?, ?)", (timestamp, emotion))
        conn.commit()
    check_stress_alert()

def check_stress_alert():
    stress_emotions = {"sad", "angry", "fear"}
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT emotion FROM mood_log ORDER BY timestamp DESC LIMIT 5")
        last_moods = [row[0] for row in cursor.fetchall()]

    if len(last_moods) < 5:
        return
    if sum(1 for m in last_moods if m in stress_emotions) >= 4:
        send_stress_alert()

def send_stress_alert():
    sender_email = os.getenv("EMAIL_USER")
    receiver_email = os.getenv("HR_EMAIL")
    password = os.getenv("EMAIL_PASS")

    if not sender_email or not password:
        print("Email credentials missing.")
        return

    subject = "Employee Stress Alert!"
    body = "An employee has shown signs of prolonged stress. Please check in."
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("HR Alert Sent")
    except Exception as e:
        print(f"Email sending error: {e}")


# Emotion Analysis

def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

def analyze_text_emotion(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return "happy"
    elif sentiment['compound'] <= -0.05:
        return "sad"
    else:
        emotion_result = emotion_classifier(text)[0][0]['label']
        mapping = {
            "joy": "happy", "anger": "angry", "fear": "fear",
            "surprise": "surprise", "sadness": "sad",
            "neutral": "neutral", "disgust": "disgust"
        }
        return mapping.get(emotion_result.lower(), "neutral")

def analyze_speech_emotion(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return "happy" if np.mean(mfccs) > 0 else "neutral"

# Flask Routes

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.json.get("text", "")
    emotion = analyze_text_emotion(text)
    save_mood_to_db(emotion)
    return jsonify({"emotion": emotion})

@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    audio_file = request.files['file']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)
    emotion = analyze_speech_emotion(audio_path)
    save_mood_to_db(emotion)
    os.remove(audio_path)
    return jsonify({"emotion": emotion})

detected_faces = []

@app.route('/get_emotion_task')
def get_emotion_task():
    return jsonify(detected_faces or [{"emotion": "No face detected", "task": "N/A"}])

def generate_frames():
    global detected_faces
    webcam = cv2.VideoCapture(0)
    while True:
        success, im = webcam.read()
        if not success:
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
        detected_faces.clear()

        for (x, y, w, h) in faces:
            image = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            emotion = labels[pred.argmax()]
            task = task_assignment[emotion]
            detected_faces.append({'emotion': emotion, 'task': task})
            save_mood_to_db(emotion)
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(im, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', im)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    webcam.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
