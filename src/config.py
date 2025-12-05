"""Configuration for JSON re-contextualization workflow."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model - gpt-4.1-nano is fastest for high-throughput 
GENERATION_MODEL = "gpt-4.1-nano"
MODEL_TEMPERATURE = 0.3

# Workflow
MAX_RETRIES = 3
OUTPUT_DIR = "output"
