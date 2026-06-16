"""
Weighted late-fusion of face, text, and audio emotion predictions.
Each modality returns a dict of {emotion: confidence}.
Weights reflect typical modality reliability for this system.
"""

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

MODALITY_WEIGHTS = {
    'face': 0.5,
    'text': 0.35,
    'audio': 0.15,
}


def _uniform_dist():
    n = len(EMOTIONS)
    return {e: 1.0 / n for e in EMOTIONS}


def _label_to_dist(label):
    dist = _uniform_dist()
    if label in dist:
        dist[label] = 0.7
        remaining = 0.3 / (len(EMOTIONS) - 1)
        for e in EMOTIONS:
            if e != label:
                dist[e] = remaining
    return dist


def fuse(face_result=None, text_result=None, audio_result=None):
    """
    Fuse predictions from available modalities.

    Each result can be either:
      - a (label, confidence_dict) tuple, or
      - a plain label string (converted to soft distribution), or
      - None (modality skipped; its weight is redistributed).

    Returns (fused_emotion, fused_scores_dict).
    """
    results = {'face': face_result, 'text': text_result, 'audio': audio_result}
    active_weights = {}
    distributions = {}

    for modality, result in results.items():
        if result is None:
            continue
        if isinstance(result, str):
            distributions[modality] = _label_to_dist(result)
        elif isinstance(result, tuple):
            label, dist = result
            if isinstance(dist, dict):
                distributions[modality] = dist
            else:
                distributions[modality] = _label_to_dist(label)
        active_weights[modality] = MODALITY_WEIGHTS[modality]

    if not active_weights:
        return 'neutral', _uniform_dist()

    total_weight = sum(active_weights.values())
    normalized_weights = {m: w / total_weight for m, w in active_weights.items()}

    fused = {e: 0.0 for e in EMOTIONS}
    for modality, dist in distributions.items():
        w = normalized_weights[modality]
        for emotion in EMOTIONS:
            fused[emotion] += w * dist.get(emotion, 0.0)

    best_emotion = max(fused, key=fused.get)
    return best_emotion, fused
