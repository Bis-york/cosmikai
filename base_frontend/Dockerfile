FROM node:20-alpine

WORKDIR /app

# Install dependencies first for better caching.
COPY package*.json ./
RUN npm install

# Copy the remainder of the front-end source.
COPY . .

EXPOSE 5180

# Launch the Vite dev server.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5180"]
