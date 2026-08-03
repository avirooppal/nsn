FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies if needed by FAISS or other native extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the core package files
COPY pyproject.toml README.md ./
COPY neurosleepnet/ ./neurosleepnet/
COPY demo.py ./

# Install the package and its dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Download the required spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-download the SentenceTransformer model to bake it into the image
# This avoids downloading the model from HuggingFace every time the container starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Run the demonstration script by default
CMD ["python", "demo.py"]
