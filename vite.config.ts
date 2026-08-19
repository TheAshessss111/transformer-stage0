import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// GitHub Pages serves a project site under /<repo>/, but the dev server runs at /.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/transformer-stage0/' : '/',
  plugins: [react(), tailwindcss()],
}));
