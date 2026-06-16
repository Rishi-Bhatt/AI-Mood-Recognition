# AI Mood Recognition System

A multimodal real-time emotion detection system that combines **face**, **text**, and **speech** analysis to predict a user's emotional state and suggest appropriate tasks.

---

## How it works

| Modality | Model | What it does |
|---|---|---|
| **Face** | Custom CNN (`expressiondetector_modern.keras`) | Detects emotion from live webcam feed |
| **Text** | `j-hartmann/emotion-english-distilroberta-base` | Classifies emotion from typed text |
| **Audio** | `superb/wav2vec2-base-superb-er` | Detects emotion acoustically from recorded speech |
| **Fusion** | Weighted late-fusion | Combines all 3 into one prediction (weights adjustable in UI) |

Emotions detected: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

---

## Project Structure

```
AI-Mood-Recognition/
├── backend/
│   ├── app.py                  # Main Flask app — all routes and model inference
│   ├── Flask_dashboard.py      # Dash analytics dashboard
│   ├── mood_database.py        # SQLite mood logging
│   ├── stress_alerts.py        # Email alert logic
│   └── requirements.txt
├── frontend/
│   └── templates/
│       └── index.html          # Main UI (served by Flask)
├── ml/
│   ├── face/                   # Face inference module
│   ├── audio/                  # Audio inference module
│   ├── text/                   # Text inference module
│   ├── fusion/                 # Multimodal fusion + temporal smoothing + TTS
│   └── training/
│       └── Trained_Model.py    # Script used to train the face CNN
├── models/                     # Model files (gitignored — download separately)
├── notebooks/                  # Jupyter training notebooks
└── download_models.py          # Script to download the face CNN from HuggingFace
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rishi-Bhatt/AI-Mood-Recognition.git
cd AI-Mood-Recognition
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r backend/requirements.txt
```

### 4. Download the face CNN model

The trained face model is **not stored in git** (it's a binary file). Download it from HuggingFace:

```powershell
python download_models.py
```

This downloads `expressiondetector_modern.keras` and `label_encoder.pkl` into `models/`.

> **If you want to train it yourself instead**, run `ml/training/Trained_Model.py` with the FER-2013 dataset placed at `images/train/` and `images/test/`. The audio and text models download automatically from HuggingFace on first run.

### 5. Set up email alerts (optional)

Create a `.env` file in the project root:

```
EMAIL_USER=your.gmail@gmail.com
EMAIL_PASS=your_16_char_app_password
HR_EMAIL=hr@yourcompany.com
```

> Gmail requires an **App Password** (not your regular password). Generate one at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords). If this is not set up, the app still works — stress alerts just won't send.

### 6. Run the app

```powershell
cd backend
python app.py
```

Open **http://localhost:5000** in your browser.

> **First run is slow** — the text (`j-hartmann`) and audio (`wav2vec2`) models download automatically (~500 MB total) and are cached locally for future runs.

---

## Usage

| Feature | How to use |
|---|---|
| **Face detection** | Allow camera access — emotions detected live |
| **Text analysis** | Type in the text box and click "Analyze Text" |
| **Speech analysis** | Click "Record Speech" — records 4 seconds of audio |
| **Combined emotion** | Auto-updates every 3 seconds from the fusion of all 3 |
| **Adjust weights** | Drag the sliders in the "Combined Emotion" card |
| **Mood dashboard** | Run `python backend/Flask_dashboard.py` → http://localhost:5000/dashboard/ |

---

## Stress Alert System

If 4 out of the last 5 detected emotions are `angry`, `sad`, or `fear`, an email alert is sent to the configured HR address. Alerts are rate-limited to once per hour to avoid spam.

---

## Models

| Model | Source | Trained on |
|---|---|---|
| Face CNN | `21f1000330/AI-Mood-Recognition` on HuggingFace | FER-2013 (48×48 grayscale, 7 emotions) |
| Text classifier | `j-hartmann/emotion-english-distilroberta-base` | GoEmotions + others |
| Audio classifier | `superb/wav2vec2-base-superb-er` | IEMOCAP (4 emotions: angry, happy, neutral, sad) |
