// src/config.ts
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string;

if (!API_BASE_URL) {
  throw new Error('Missing VITE_API_BASE_URL. Add it to your .env file.');
}

