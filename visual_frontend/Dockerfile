FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install
RUN chmod -R +x /app/node_modules/.bin/

COPY . .

ENV VITE_HOST=0.0.0.0 \
    VITE_PORT=5173 \
    VITE_STRICT_PORT=true \
    VITE_HTTPS=false \
    VITE_OPEN=false \
    VITE_API_BASE_URL=http://localhost:8000

EXPOSE 5173

CMD ["/app/node_modules/.bin/vite", "--host", "0.0.0.0", "--port", "5173"]
