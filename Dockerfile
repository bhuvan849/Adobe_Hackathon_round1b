# Use a specific platform for compatibility as required by the hackathon
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# --- FIX: Install build tools (like gcc) needed to compile some Python packages ---
RUN apt-get update && apt-get install -y build-essential

# --- Force Install CPU-Only PyTorch ---
# This installs torch separately using the correct index URL
# to guarantee the small, CPU-only version is installed.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file for the other packages
COPY requirements.txt .

# Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Download the necessary spacy model
RUN python -m spacy download en_core_web_sm

# --- Model Caching Step ---
COPY download_model.py .
RUN python download_model.py
ENV TRANSFORMERS_OFFLINE=1
# Copy the rest of the application source code into the container
COPY . .

# Specify the command to run when the container starts
CMD ["python", "main.py"]