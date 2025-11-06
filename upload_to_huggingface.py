from huggingface_hub import HfApi

api = HfApi()

repo_id = "21f1000330/AI-Mood-Recognition"

# Upload the Keras model file
api.upload_file(
    path_or_fileobj="expressiondetector_modern.keras",
    path_in_repo="expressiondetector_modern.keras",
    repo_id=repo_id,
    repo_type="model",
)

# Upload the custom model.py file
api.upload_file(
    path_or_fileobj="model.py",
    path_in_repo="model.py",
    repo_id=repo_id,
    repo_type="model",
)

# Upload the requirements.txt file
api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=repo_id,
    repo_type="model",
)

print(f"All files uploaded to {repo_id} successfully!")