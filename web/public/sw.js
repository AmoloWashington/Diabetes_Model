// GlucoLab service worker: makes the app installable and lets the interface open offline.
// Calculations always need the server, so /api requests are never cached.
const VERSION = "glucolab-v1";
const SHELL = ["/", "/manifest.webmanifest", "/icons/icon-192.png", "/icons/icon-512.png", "/favicon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

// The first visit loads its scripts before this worker is active; the page then sends their URLs
// so they can be cached and the interface opens offline next time.
self.addEventListener("message", (event) => {
  const urls = event.data && event.data.type === "cache-assets" ? event.data.urls : null;
  if (!Array.isArray(urls)) return;
  const own = urls.filter((u) => { try { const x = new URL(u); return x.origin === self.location.origin && x.pathname.startsWith("/assets/"); } catch { return false; } });
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(own)).catch(() => {}));
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || url.pathname.startsWith("/api/")) return;

  // Pages: network first so a new deployment is picked up at once; cached shell when offline.
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req)
        .then((res) => { const copy = res.clone(); caches.open(VERSION).then((c) => c.put("/", copy)); return res; })
        .catch(() => caches.match("/")),
    );
    return;
  }
  // Built assets have content hashes in their names, so a cached copy never goes stale.
  if (url.pathname.startsWith("/assets/")) {
    event.respondWith(
      caches.match(req).then((hit) => hit || fetch(req).then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
        return res;
      })),
    );
    return;
  }
  event.respondWith(fetch(req).catch(() => caches.match(req)));
});
