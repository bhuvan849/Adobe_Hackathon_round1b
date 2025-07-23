# download_model.py
from sentence_transformers import SentenceTransformer

# This will download the model from the Hugging Face Hub and cache it.
# In the Dockerfile, this runs once during the image build process.
print("Downloading and caching model 'all-MiniLM-L6-v2'...")
SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded successfully.")