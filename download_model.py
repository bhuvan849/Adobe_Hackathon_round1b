# download_model.py
from sentence_transformers import SentenceTransformer

# This will download the new, more powerful model during the image build process.
print("Downloading and caching model 'all-mpnet-base-v2'...")
SentenceTransformer('all-mpnet-base-v2')
print("Model downloaded successfully.")