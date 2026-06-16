"""
Downloads the trained face CNN model from HuggingFace into the models/ directory.
Run this once before starting the app for the first time.

Usage:
    python download_models.py
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID   = "21f1000330/AI-Mood-Recognition"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FILES = [
    "expressiondetector_modern.keras",
    "label_encoder.pkl",
]

print(f"Downloading models from HuggingFace: {REPO_ID}")
for filename in FILES:
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  {filename} already exists, skipping.")
        continue
    print(f"  Downloading {filename} ...")
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR,
        )
        print(f"  {filename} saved to models/")
    except Exception as e:
        print(f"  ERROR downloading {filename}: {e}")
        print(f"  You may need to train the model yourself — see ml/training/Trained_Model.py")

print("\nDone. You can now run: python backend/app.py")
