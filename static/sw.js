const CACHE_NAME = 'tahoe-snow-v3';
const API_CACHE = 'tahoe-snow-api-v3';

self.addEventListener('install', event => {
  // Skip waiting so new SW activates immediately
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  // Delete ALL old caches on activation
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME && k !== API_CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Static assets (icons, manifest): cache-first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then(cached => cached || fetch(event.request))
    );
    return;
  }

  // Everything else (HTML, API): network-first, only cache successful responses
  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Only cache successful responses (not errors or redirects)
        if (response.ok) {
          const clone = response.clone();
          caches.open(API_CACHE).then(cache => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
