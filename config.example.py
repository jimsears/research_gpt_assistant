# Example configuration for ResearchGPT Assistant
# Copy this file to config.py and replace placeholder values with your own.
# DO NOT commit your real config.py file to GitHub.

import os
from dotenv import load_dotenv

# Load environment variables if present
load_dotenv()

class Config:
    # Mistral API settings
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "your_api_key_here")
    MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

    # Directory paths
    SAMPLE_PAPERS_DIR = os.getenv("SAMPLE_PAPERS_DIR", "data/sample_papers/")
    PROCESSED_DIR = os.getenv("PROCESSED_DIR", "processed/")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "runs/")

    # Processing parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    OVERLAP = int(os.getenv("OVERLAP", "100"))
