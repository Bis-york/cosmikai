// src/config.ts
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string;
export const VISUAL_BASE_URL = import.meta.env.VITE_VISUAL_BASE_URL as string;

if (!API_BASE_URL) throw new Error('Missing VITE_API_BASE_URL');
if (!VISUAL_BASE_URL) throw new Error('Missing VITE_VISUAL_BASE_URL');
