/// <reference types="vite/client" />

// (optional) be explicit about the vars you expect:
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_VISUAL_BASE_URL: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
