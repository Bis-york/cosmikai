FROM python:3.11-bookworm AS base

# Install system packages required for scientific Python stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files separately for better caching.
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy backend source code and server entrypoint.
COPY backend /app/backend
COPY server_setup.py /app/server_setup.py

# Default environment configuration; can be overridden at runtime.
ENV COSMIKAI_MONGO_URI="mongodb://mongo:27017/" \
    COSMIKAI_MONGO_DB="exoplanet_DB" \
    COSMIKAI_MONGO_COLLECTION="predictions"

EXPOSE 8000

# Run the API server via the helper that waits for MongoDB.
CMD ["python", "server_setup.py", "--host", "0.0.0.0", "--port", "8000"]
