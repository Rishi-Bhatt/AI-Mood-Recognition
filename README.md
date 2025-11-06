# AI Mood Recognition

## Project Description

This project implements an AI-powered mood recognition system. It utilizes a deep learning model to detect and classify human emotions from facial expressions.

## Local Setup

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/AI_Mood_Recognition.git
    cd AI_Mood_Recognition
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    (Note: We will create the `requirements.txt` file in a later step.)

4.  **Run the application:**

    Depending on the main entry point, you would typically run:

    ```bash
    python app.py
    # or
    python Flask_dashboard.py
    ```

## Hugging Face Deployment

This project's emotion detection model (`expressiondetector_modern.keras`) is deployed on Hugging Face. Users can interact with the model directly through the Hugging Face platform.

### Deploying to Hugging Face

To deploy your model to Hugging Face, follow these steps:

1.  **Install Hugging Face Hub library:**

    ```bash
    pip install huggingface_hub
    ```

2.  **Log in to Hugging Face:**

    ```bash
    huggingface-cli login
    ```
    Follow the prompts to enter your Hugging Face token.

3.  **Create a new model repository:**

    Go to [Hugging Face New Model](https://huggingface.co/new) and create a new model repository. Choose a name like `AI_Mood_Recognition_Model`.

4.  **Push your model files:**

    You will need to push the following files to your Hugging Face model repository:

    *   `expressiondetector_modern.keras` (your trained model)
    *   `model.py` (the custom inference code)
    *   `requirements.txt` (dependencies for the model)

    Here's how you can push them using the `huggingface_hub` library:

    ```python
    from huggingface_hub import HfApi

    api = HfApi()

    # Replace "your-username/AI_Mood_Recognition_Model" with your actual model ID
    repo_id = "your-username/AI_Mood_Recognition_Model"

    api.upload_file(
        path_or_fileobj="expressiondetector_modern.keras",
        path_in_repo="expressiondetector_modern.keras",
        repo_id=repo_id,
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj="model.py",
        path_in_repo="model.py",
        repo_id=repo_id,
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="model",
    )
    ```

    Alternatively, you can use `git LFS` to push larger files:

    ```bash
    git lfs install
    git clone https://huggingface.co/your-username/AI_Mood_Recognition_Model
    cd AI_Mood_Recognition_Model
    cp ../expressiondetector_modern.keras .
    cp ../model.py .
    cp ../requirements.txt .
    git add .
    git commit -m "Add initial model files"
    git push
    ```

Once pushed, your model will be available on Hugging Face, and users will be able to load and use it in their applications.
