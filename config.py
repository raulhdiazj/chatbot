## Configuration file

import os

CONFIG = {
    # --- Model Parameters ---
    "LLM_MODEL": "meta-llama/Meta-Llama-3-8B-Instruct",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "LLM_TEMPERATURE": 0.2,        # Low temperature to reduce technical hallucinations
    "MAX_NEW_TOKENS": 512,         # Sufficient length for complex code trace responses
    
    # --- Retrieval Settings ---
    "RETRIEVAL_K": 2,              # Number of text chunks to pull from vector store
    "CHUNK_SIZE": 250,
    "CHUNK_OVERLAP": 35,
    
    # --- Short-Term Memory ---
    "MAX_MEMORY_TURNS": 4,         # Keeps context length safe on the free tier window
    
    # --- Persistence Paths ---
    "DB_DIR": "./chroma_db",
    "LOG_FILE": "colabbuddy_analytics.jsonl",
    "PROFILE_FILE": "user_profiles.json"
}