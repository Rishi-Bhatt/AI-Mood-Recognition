from huggingface_hub import hf_hub_download
from transformers import pipeline
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

print("Downloading expressiondetector_modern.keras...")
try:
    hf_hub_download(
        repo_id='21f1000330/AI-Mood-Recognition',
        filename='expressiondetector_modern.keras',
        local_dir='.'
    )
    print("expressiondetector_modern.keras downloaded successfully!")
except Exception as e:
    print(f"Error downloading expressiondetector_modern.keras: {e}")

print("Initializing DistilBERT pipeline to trigger model download (if not already cached)...")
try:
    # download and cache the distilbert model if it's not already present
    _ = pipeline(
        "text-classification",
        model="21f1000330/distilbert-base-uncased-emotion",
        top_k=1,
        framework="tf" # setting framework to tensorflow
    )
    print("DistilBERT model is ready (downloaded or already cached).")
except Exception as e:
    print(f"Error with DistilBERT model: {e}")

print("All necessary models have been downloaded or are ready.")
