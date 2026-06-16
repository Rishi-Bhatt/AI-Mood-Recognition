"""
Weighted late-fusion of face, text, and audio emotion predictions.
Weights are passed in at call time so the UI can adjust them live.
"""

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

DEFAULT_WEIGHTS = {
    'face': 0.50,
    'text': 0.35,
    'audio': 0.15,
}


def _uniform_dist():
    n = len(EMOTIONS)
    return {e: 1.0 / n for e in EMOTIONS}


def _label_to_dist(label):
    dist = _uniform_dist()
    if label in dist:
        dist[label] = 0.70
        remaining = 0.30 / (len(EMOTIONS) - 1)
        for e in EMOTIONS:
            if e != label:
                dist[e] = remaining
    return dist


def fuse(face_result=None, text_result=None, audio_result=None, weights=None):
    """
    Fuse predictions from available modalities.

    Each result can be:
      - a plain label string, or
      - a (label, confidence_dict) tuple, or
      - None (modality skipped; its weight is redistributed).

    weights: dict with keys 'face', 'text', 'audio'. Values are unnormalized
             (they are normalized internally so they don't have to sum to 1).

    Returns (fused_emotion, fused_scores_dict).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

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
            distributions[modality] = dist if isinstance(dist, dict) else _label_to_dist(label)
        active_weights[modality] = max(weights.get(modality, 0.0), 0.0)

    if not active_weights or sum(active_weights.values()) == 0:
        return 'neutral', _uniform_dist()

    total = sum(active_weights.values())
    norm_weights = {m: w / total for m, w in active_weights.items()}

    fused = {e: 0.0 for e in EMOTIONS}
    for modality, dist in distributions.items():
        w = norm_weights[modality]
        for emotion in EMOTIONS:
            fused[emotion] += w * dist.get(emotion, 0.0)

    best = max(fused, key=fused.get)
    return best, fused
