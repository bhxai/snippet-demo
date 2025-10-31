import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react'; // Adjust for your framework

export default defineConfig({
  plugins: [react()],
  base: './', // Optional: Relative base for root deploys
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000', // Proxy API to backend
        changeOrigin: true,
      },
    },
  },
  build: {
    manifest: true, // For prod asset mapping
    // REMOVED: outDir: '../backend/static' — Use default 'dist/' instead
  },
});
