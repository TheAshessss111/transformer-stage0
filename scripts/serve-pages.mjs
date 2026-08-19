/**
 * Serve the production build the way GitHub Pages will, so a deploy can be
 * checked before it is pushed.
 *
 * Reproduces the two things that differ from `vite preview`:
 *   1. the app lives under the /transformer-stage0/ base path
 *   2. unknown paths fall back to 404.html (which is a copy of index.html),
 *      which is what makes deep links like /step/0-3 survive a refresh
 *
 *   npm run build && npm run preview:pages
 */
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { join, normalize, extname } from 'node:path';

const BASE = '/transformer-stage0/';
const DIST = new URL('../dist/', import.meta.url).pathname;
const PORT = Number(process.env.PORT ?? 4321);

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
  '.woff2': 'font/woff2',
};

async function readIfFile(path) {
  try {
    const s = await stat(path);
    return s.isFile() ? await readFile(path) : null;
  } catch {
    return null;
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://localhost:${PORT}`);

  if (!url.pathname.startsWith(BASE)) {
    res.writeHead(302, { Location: BASE });
    res.end();
    return;
  }

  const rel = normalize(url.pathname.slice(BASE.length)).replace(/^(\.\.[/\\])+/, '');
  const direct = await readIfFile(join(DIST, rel));
  const index = direct ?? (await readIfFile(join(DIST, rel, 'index.html')));

  if (index) {
    const ext = direct ? extname(rel) : '.html';
    res.writeHead(200, { 'Content-Type': MIME[ext] ?? 'application/octet-stream' });
    res.end(index);
    return;
  }

  // GitHub Pages serves 404.html with a 404 status; the SPA boots and routes anyway.
  const fallback = await readIfFile(join(DIST, '404.html'));
  if (fallback) {
    res.writeHead(404, { 'Content-Type': 'text/html' });
    res.end(fallback);
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('not found (did you run `npm run build` first?)');
});

server.listen(PORT, () => {
  console.log(`GitHub Pages simulation: http://localhost:${PORT}${BASE}`);
  console.log(`deep-link check:        http://localhost:${PORT}${BASE}step/0-3`);
});
