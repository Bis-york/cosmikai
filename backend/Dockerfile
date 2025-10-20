FROM python:3.11-slim-bookworm

# Install system packages required for scientific Python stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all backend source code
COPY . backend/

# Create empty __init__.py in parent to make backend a proper package
RUN touch /app/__init__.py

# Set Python path so backend can be imported as a package
ENV PYTHONPATH="/app"

# Default environment configuration; can be overridden at runtime
# Points to MongoDB on host machine by default
ENV COSMIKAI_MONGO_URI="mongodb://host.docker.internal:27017/" \
    COSMIKAI_MONGO_DB="exoplanet_DB" \
    COSMIKAI_MONGO_COLLECTION="predictions"

EXPOSE 8000

# Run the API server via the helper that waits for MongoDB
CMD ["python", "backend/server_setup.py", "--host", "0.0.0.0", "--port", "8000"]
