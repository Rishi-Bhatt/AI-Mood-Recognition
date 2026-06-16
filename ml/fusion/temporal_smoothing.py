from collections import deque

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class TemporalSmoother:
    """
    Smooths emotion predictions over a sliding window of recent frames.
    Uses majority vote over the window to suppress single-frame noise.
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self._history = deque(maxlen=window_size)

    def update(self, emotion):
        self._history.append(emotion)
        return self.current_emotion()

    def current_emotion(self):
        if not self._history:
            return 'neutral'
        counts = {e: 0 for e in EMOTIONS}
        for e in self._history:
            if e in counts:
                counts[e] += 1
        return max(counts, key=counts.get)

    def reset(self):
        self._history.clear()


def smooth_emotions(history, window_size=10):
    """
    Stateless version: given a list of recent emotion labels, return the smoothed label.
    """
    if not history:
        return 'neutral'
    window = history[-window_size:]
    counts = {e: 0 for e in EMOTIONS}
    for e in window:
        if e in counts:
            counts[e] += 1
    return max(counts, key=counts.get)
