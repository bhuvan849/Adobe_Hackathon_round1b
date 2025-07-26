# download_model.py
from sentence_transformers import SentenceTransformer

# Download the main, powerful model
print("Downloading and caching model 'all-mpnet-base-v2'...")
SentenceTransformer('all-mpnet-base-v2')
print("Model 'all-mpnet-base-v2' downloaded successfully.")

# --- FIX: Download the small test model needed for validation ---
print("Downloading and caching test model 'all-MiniLM-L6-v2'...")
SentenceTransformer('all-MiniLM-L6-v2')
print("Model 'all-MiniLM-L6-v2' downloaded successfully.")