import sqlite3
import smtplib
import os
import sys
from datetime import datetime
from email.mime.text import MIMEText
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import librosa
import tf_keras as keras
from transformers import pipeline
from dotenv import load_dotenv

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend", "templates")
sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

from ml.fusion.fusion_layer import fuse

app = Flask(__name__, template_folder=TEMPLATES_DIR)

EMOTION_MAP = {
    "joy": "happy", "anger": "angry", "sadness": "sad",
    "fear": "fear", "disgust": "disgust", "surprise": "surprise",
    "neutral": "neutral"
}

task_assignment = {
    'angry':   "Take deep breaths and relax.",
    'disgust': "Engage in a team discussion for clarity.",
    'fear':    "Reassess task workload and provide support.",
    'happy':   "Encourage collaboration and brainstorming.",
    'neutral': "Continue with assigned tasks normally.",
    'sad':     "Assign lighter tasks or encourage social interaction.",
    'surprise':"Allow creative or unexpected tasks."
}

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# ── Model loading ─────────────────────────────────────────────────────────────
print("Loading text emotion classifier (j-hartmann/emotion-english-distilroberta-base)...")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

print("Loading Whisper ASR (openai/whisper-base)...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base"
)

print("Loading face CNN...")
face_model = keras.models.load_model(
    os.path.join(MODELS_DIR, "expressiondetector_modern.keras"), compile=False
)
print("All models loaded!")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# ── Global state for fusion ───────────────────────────────────────────────────
latest_emotions = {'face': None, 'text': None, 'audio': None}
detected_faces = []

# ── DB helpers ────────────────────────────────────────────────────────────────
def save_mood_to_db(emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(os.path.join(BASE_DIR, "mood_tracking.db")) as conn:
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
    with sqlite3.connect(os.path.join(BASE_DIR, "mood_tracking.db")) as conn:
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

# ── Inference helpers ─────────────────────────────────────────────────────────
def extract_face_features(image):
    return np.array(image).reshape(1, 48, 48, 1) / 255.0

def analyze_text_emotion(text):
    result = emotion_classifier(text)[0][0]
    return EMOTION_MAP.get(result['label'].lower(), "neutral")

def analyze_speech_emotion(audio_path):
    # Load & resample to 16 kHz mono (required by Whisper)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    transcription = asr_pipeline({"array": y, "sampling_rate": sr})['text'].strip()
    if not transcription:
        return "neutral", ""
    emotion = analyze_text_emotion(transcription)
    return emotion, transcription

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.json.get("text", "")
    emotion = analyze_text_emotion(text)
    latest_emotions['text'] = emotion
    save_mood_to_db(emotion)
    return jsonify({"emotion": emotion})

@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    audio_file = request.files['file']
    audio_path = os.path.join(BASE_DIR, "temp_audio.wav")
    audio_file.save(audio_path)
    emotion, transcription = analyze_speech_emotion(audio_path)
    latest_emotions['audio'] = emotion
    save_mood_to_db(emotion)
    try:
        os.remove(audio_path)
    except OSError:
        pass
    return jsonify({"emotion": emotion, "transcription": transcription})

@app.route('/get_emotion_task')
def get_emotion_task():
    return jsonify(detected_faces or [{"emotion": "No face detected", "task": "N/A"}])

@app.route('/fuse')
def fuse_emotions():
    emotion, scores = fuse(
        face_result=latest_emotions['face'],
        text_result=latest_emotions['text'],
        audio_result=latest_emotions['audio']
    )
    task = task_assignment.get(emotion, "Continue with assigned tasks normally.")
    return jsonify({
        "fused_emotion": emotion,
        "task": task,
        "sources": {k: v for k, v in latest_emotions.items()},
        "scores": {k: round(v, 3) for k, v in scores.items()}
    })

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
            img = extract_face_features(image)
            pred = face_model.predict(img, verbose=0)
            emotion = labels[pred.argmax()]
            task = task_assignment[emotion]
            detected_faces.append({'emotion': emotion, 'task': task})
            latest_emotions['face'] = emotion
            save_mood_to_db(emotion)
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(im, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', im)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    webcam.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
