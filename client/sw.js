self.addEventListener('install', event => {
  console.log("Service Worker installed");
  event.waitUntil(caches.open('v1').then(cache => {
    return cache.addAll(['/','/index.html','/manifest.json']);
  }));
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(resp => {
      return resp || fetch(event.request);
    })
  );
});
