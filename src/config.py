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

# Batching optimization
TARGET_TOKENS_PER_BATCH = 600   # Lower target for faster batches (<8s each)
LARGE_TEXT_THRESHOLD = 2000    # Chars - texts above this get split
TOKENS_PER_CHAR = 0.25         # Approximation: ~4 chars per token
