FROM node:20-alpine

WORKDIR /app

# Install dependencies first for better caching.
COPY package*.json ./
RUN npm install

# Copy the remainder of the front-end source.
COPY . .

# Default environment values suitable for the Docker Compose network.
ENV VITE_HOST=0.0.0.0 \
    VITE_PORT=5180 \
    VITE_STRICT_PORT=true \
    VITE_HTTPS=false \
    VITE_OPEN=false \
    VITE_API_BASE_URL=http://localhost:8000 \
    VITE_VISUAL_BASE_URL=http://localhost:5173

EXPOSE 5180

# Launch the Vite dev server.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5180"]
