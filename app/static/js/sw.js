/**
 * Service Worker for Burki Voice AI
 * Implements PWA capabilities, caching strategies, and offline functionality
 * Improves Core Web Vitals and SEO performance
 */

const CACHE_NAME = 'burki-v1.0.0';
const STATIC_CACHE = 'burki-static-v1';
const DYNAMIC_CACHE = 'burki-dynamic-v1';

// Resources to cache immediately
const STATIC_ASSETS = [
  '/',
  '/auth/login',
  '/auth/register',
  '/docs',
  '/static/css/tailwind.css',
  '/static/logo/dark.svg',
  '/static/logo/light.svg',
  '/static/logo/favicon.svg',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap'
];

// API endpoints that should be cached with network-first strategy
const API_CACHE_PATTERNS = [
  '/api/assistants',
  '/api/calls',
  '/health'
];

// Pages that should work offline
const OFFLINE_PAGES = [
  '/',
  '/docs',
  '/auth/login'
];

/**
 * Install Event - Cache static assets
 */
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('Service Worker: Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('Service Worker: Installation complete');
        return self.skipWaiting(); // Force activate immediately
      })
      .catch((error) => {
        console.error('Service Worker: Installation failed:', error);
      })
  );
});

/**
 * Activate Event - Clean up old caches
 */
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
              console.log('Service Worker: Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('Service Worker: Activation complete');
        return self.clients.claim(); // Take control of all clients
      })
  );
});

/**
 * Fetch Event - Implement caching strategies
 */
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Handle different types of requests with appropriate strategies
  if (isStaticAsset(url)) {
    event.respondWith(cacheFirstStrategy(request));
  } else if (isAPIRequest(url)) {
    event.respondWith(networkFirstStrategy(request));
  } else if (isPageRequest(url)) {
    event.respondWith(staleWhileRevalidateStrategy(request));
  } else {
    event.respondWith(networkOnlyStrategy(request));
  }
});

/**
 * Cache-First Strategy (for static assets)
 * Best for assets that rarely change
 */
async function cacheFirstStrategy(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    const cache = await caches.open(STATIC_CACHE);
    cache.put(request, networkResponse.clone());
    
    return networkResponse;
  } catch (error) {
    console.error('Cache-First Strategy failed:', error);
    return new Response('Offline', { status: 503 });
  }
}

/**
 * Network-First Strategy (for API requests)
 * Always try network first, fallback to cache
 */
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('Network failed, trying cache:', error);
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response(JSON.stringify({ error: 'Offline' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Stale-While-Revalidate Strategy (for pages)
 * Return cached version immediately, update cache in background
 */
async function staleWhileRevalidateStrategy(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  // Fetch fresh version in background
  const fetchPromise = fetch(request).then((networkResponse) => {
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  });
  
  // Return cached version immediately if available
  if (cachedResponse) {
    return cachedResponse;
  }
  
  // Otherwise wait for network
  try {
    return await fetchPromise;
  } catch (error) {
    console.error('Stale-While-Revalidate failed:', error);
    return getOfflinePage();
  }
}

/**
 * Network-Only Strategy (for uncached requests)
 */
async function networkOnlyStrategy(request) {
  try {
    return await fetch(request);
  } catch (error) {
    console.error('Network-Only Strategy failed:', error);
    return new Response('Network Error', { status: 503 });
  }
}

/**
 * Check if request is for static assets
 */
function isStaticAsset(url) {
  const staticExtensions = ['.css', '.js', '.png', '.jpg', '.jpeg', '.svg', '.woff', '.woff2'];
  const pathname = url.pathname;
  
  return staticExtensions.some(ext => pathname.endsWith(ext)) || 
         pathname.startsWith('/static/') ||
         url.hostname !== self.location.hostname;
}

/**
 * Check if request is for API endpoints
 */
function isAPIRequest(url) {
  return url.pathname.startsWith('/api/') || 
         API_CACHE_PATTERNS.some(pattern => url.pathname.includes(pattern));
}

/**
 * Check if request is for a page
 */
function isPageRequest(url) {
  return url.hostname === self.location.hostname &&
         !url.pathname.startsWith('/api/') &&
         !isStaticAsset(url);
}

/**
 * Get offline fallback page
 */
async function getOfflinePage() {
  const cache = await caches.open(STATIC_CACHE);
  const offlinePage = await cache.match('/');
  
  if (offlinePage) {
    return offlinePage;
  }
  
  return new Response(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Burki - Offline</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body { 
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          text-align: center; 
          padding: 50px;
          background: #111827;
          color: white;
        }
        .logo { max-width: 200px; margin-bottom: 20px; }
        .message { font-size: 18px; margin-bottom: 20px; }
        .retry { 
          background: #3b82f6; 
          color: white; 
          padding: 10px 20px; 
          border: none; 
          border-radius: 5px; 
          cursor: pointer;
        }
      </style>
    </head>
    <body>
      <div>
        <div class="message">You're offline</div>
        <p>Burki Voice AI is not available right now. Please check your connection.</p>
        <button class="retry" onclick="location.reload()">Try Again</button>
      </div>
    </body>
    </html>
  `, {
    headers: { 'Content-Type': 'text/html' }
  });
}

/**
 * Background Sync for failed requests
 */
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    event.waitUntil(doBackgroundSync());
  }
});

async function doBackgroundSync() {
  // Implement background sync logic here
  console.log('Background sync triggered');
}

/**
 * Push notification handling
 */
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'New update available',
    icon: '/static/logo/favicon.svg',
    badge: '/static/logo/favicon.svg',
    tag: 'burki-notification',
    actions: [
      {
        action: 'open',
        title: 'Open Burki',
        icon: '/static/logo/favicon.svg'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('Burki Voice AI', options)
  );
});

/**
 * Notification click handling
 */
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'open' || !event.action) {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

/**
 * Performance monitoring
 */
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'CACHE_PERFORMANCE') {
    // Log cache performance metrics
    console.log('Cache performance:', event.data.metrics);
  }
});

console.log('Burki Service Worker loaded successfully'); 