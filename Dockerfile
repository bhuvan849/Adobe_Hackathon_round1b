# Use a specific platform for compatibility as required by the hackathon
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# --- Force Install CPU-Only PyTorch ---
# This is the definitive fix. We install torch separately using the correct index URL
# to guarantee the small, CPU-only version is installed.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file for the other packages
COPY requirements.txt .

# Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Model Caching Step ---
COPY download_model.py .
RUN python download_model.py

# Copy the rest of the application source code into the container
COPY . .

# Specify the command to run when the container starts
CMD ["python", "main.py"]