# Example configuration for ResearchGPT Assistant
# Copy this file to config.py and replace placeholder values with your own.
# DO NOT commit your real config.py file to GitHub.

import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # Load environment variables if present
        load_dotenv()

        # Mistral API settings
        self.MISTRAL_API_KEY = "your_api_key_here"  # Replace with your actual key
        self.MODEL_NAME = "mistral-medium"
        self.TEMPERATURE = 0.1
        self.MAX_TOKENS = 1000

        # Directory paths
        self.DATA_DIR = "data/"
        self.SAMPLE_PAPERS_DIR = "data/sample_papers/"
        self.PROCESSED_DIR = "data/processed/"
        self.RESULTS_DIR = "results/"

        # Processing parameters
        self.CHUNK_SIZE = 1000
        self.OVERLAP = 100