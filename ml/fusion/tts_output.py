"""
Text-to-speech output for emotion-based feedback.
Uses pyttsx3 (offline, cross-platform) with a gTTS fallback.
"""

EMOTION_MESSAGES = {
    'angry':    "You seem frustrated. Take a deep breath and try to relax.",
    'disgust':  "It looks like something is bothering you. Consider stepping away for a moment.",
    'fear':     "You appear anxious. Remember, you can ask for help anytime.",
    'happy':    "Great to see you're in a positive mood! Keep up the energy.",
    'neutral':  "You appear calm and focused. Good work.",
    'sad':      "You seem a bit down. It's okay — consider taking a short break.",
    'surprise': "Something caught you off guard! Channel that energy creatively.",
}


def speak(emotion, blocking=True):
    """
    Speak an emotion-appropriate message. Tries pyttsx3 first, falls back to gTTS.
    Set blocking=False to speak in a background thread.
    """
    message = EMOTION_MESSAGES.get(emotion, f"Detected emotion: {emotion}.")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        if blocking:
            engine.say(message)
            engine.runAndWait()
        else:
            import threading
            def _speak():
                engine.say(message)
                engine.runAndWait()
            threading.Thread(target=_speak, daemon=True).start()
        return True
    except Exception as e:
        print(f"pyttsx3 error: {e}. Trying gTTS...")

    try:
        from gtts import gTTS
        import tempfile
        import os
        import subprocess

        tts = gTTS(text=message, lang='en')
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            tmp_path = f.name
        tts.save(tmp_path)

        if os.name == 'nt':
            subprocess.Popen(['start', '/min', tmp_path], shell=True)
        else:
            subprocess.Popen(['mpg123', tmp_path])
        return True
    except Exception as e:
        print(f"gTTS error: {e}. TTS unavailable.")
        return False


def get_message(emotion):
    return EMOTION_MESSAGES.get(emotion, f"Detected emotion: {emotion}.")
