FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        build-essential \
        supervisor \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g serve \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./

ARG BASE_API_URL=api.flyingwaffle.ca
ARG BASE_VISUAL_URL=visuals.flyingwaffle.ca
ARG VISUAL_API_URL=api.flyingwaffle.ca

WORKDIR /app/base_frontend
RUN npm install
RUN VITE_API_BASE_URL=${BASE_API_URL} \
    VITE_VISUAL_BASE_URL=${BASE_VISUAL_URL} \
    npm run build \
    && rm -rf node_modules

WORKDIR /app/visual_frontend
RUN npm install
RUN VITE_API_BASE_URL=${VISUAL_API_URL} \
    npm run build \
    && rm -rf node_modules

WORKDIR /app


COPY docker/supervisord.conf /etc/supervisor/conf.d/cosmikai.conf

ENV COSMIKAI_MONGO_URI=mongodb://localhost:27017/ \
    COSMIKAI_MONGO_DB=exoplanet_DB \
    COSMIKAI_MONGO_COLLECTION=predictions

EXPOSE 8000 5180 5173

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/cosmikai.conf"]
