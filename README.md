# Colab Local Chat Demo

This repository contains a simple Gradio web app that runs a small local language model (GPT-2) using Hugging Face Transformers. It can be deployed to Hugging Face Spaces (Gradio runtime) or run locally/Colab. The app uses CPU by default.

Files:
- app.py — Gradio application (chat UI)
- requirements.txt — Python dependencies

How to deploy to Hugging Face Spaces
1. Go to https://huggingface.co/spaces and click "Create new Space".
2. Choose "Gradio" as the SDK and select "Public" visibility.
3. Under the "Repository" section click "Import from GitHub" and paste this repository URL:

   https://github.com/shawnsilve8814-droid/colab-local-chat-demo

4. Wait for the Space to build. Once finished you can open the web UI and chat with the model.

Notes & caveats
- GPT-2 is small and will run on CPU, but quality is limited compared to modern chat models.
- If you want a better model, you can edit app.py to load a different model (e.g., `gpt2-medium`) or configure a GPU.
- Running large models may require more compute or a paid Space.
