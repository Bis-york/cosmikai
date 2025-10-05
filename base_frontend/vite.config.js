import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

function resolveAllowedHosts() {
  const envValue = process.env.VITE_ALLOWED_HOSTS
  if (!envValue) return undefined
  const hosts = envValue
    .split(',')
    .map((host) => host.trim())
    .filter(Boolean)
  return hosts.length ? hosts : undefined
}

// https://vite.dev/config/
export default defineConfig(() => {
  const allowedHosts = resolveAllowedHosts() ?? ['cosmikai.flyingwaffle.ca']
  const sharedServerConfig = {
    host: true,
    port: 5180,
    allowedHosts,
  }
  return {
    plugins: [react()],
    server: sharedServerConfig,
    preview: { ...sharedServerConfig },
  }
})
