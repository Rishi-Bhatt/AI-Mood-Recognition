from transformers import pipeline

LABEL_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise"
}

nlp = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)

def predict_text_emotion(text):
    result = nlp(text)[0][0]
    label = LABEL_MAP.get(result['label'].lower(), "neutral")
    score = result['score']
    return label, score
