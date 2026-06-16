import numpy as np
import librosa

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_mfcc(audio_path, n_mfcc=40, sr=22050):
    y, sample_rate = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def predict_audio_emotion(audio_path):
    """
    Heuristic audio emotion classifier using MFCC features.
    Returns (emotion_label, confidence_dict).
    """
    try:
        mfccs = extract_mfcc(audio_path)
    except Exception as e:
        print(f"Audio load error: {e}")
        return "neutral", {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS}

    mean_energy = np.mean(mfccs)
    std_energy = np.std(mfccs)
    delta = np.mean(np.diff(mfccs))

    scores = {label: 0.0 for label in EMOTION_LABELS}

    if mean_energy > 5 and std_energy > 10:
        scores['angry'] += 0.4
        scores['surprise'] += 0.2
    elif mean_energy > 2:
        scores['happy'] += 0.3
        scores['surprise'] += 0.1
    elif mean_energy < -2:
        scores['sad'] += 0.3
        scores['fear'] += 0.2
    else:
        scores['neutral'] += 0.5

    if delta > 0.5:
        scores['happy'] += 0.15
        scores['surprise'] += 0.1
    elif delta < -0.5:
        scores['sad'] += 0.15
        scores['fear'] += 0.1

    for label in EMOTION_LABELS:
        scores[label] = max(scores[label], 0.05)

    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}

    predicted = max(scores, key=scores.get)
    return predicted, scores
