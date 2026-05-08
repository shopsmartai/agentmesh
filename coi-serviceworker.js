/*!
 * COOP/COEP Service Worker — adapted from coi-serviceworker (MIT, gzuidhof).
 * Injects Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
 * into every response so the page becomes cross-origin isolated, which enables
 * SharedArrayBuffer (required by ONNX Runtime WebGPU + threaded WASM).
 *
 * On hosts that can't set custom headers (GitHub Pages), this is the canonical
 * workaround. The first page load registers the SW and reloads once; subsequent
 * loads come up cross-origin isolated.
 */
let coepCredentialless = false;

if (typeof window === 'undefined') {
  // Service worker context.
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener('message', (ev) => {
    if (!ev.data) return;
    if (ev.data.type === 'deregister') {
      self.registration
        .unregister()
        .then(() => self.clients.matchAll())
        .then((clients) => clients.forEach((c) => c.navigate(c.url)));
    } else if (ev.data.type === 'coepCredentialless') {
      coepCredentialless = ev.data.value;
    }
  });

  self.addEventListener('fetch', (event) => {
    const r = event.request;
    if (r.cache === 'only-if-cached' && r.mode !== 'same-origin') return;

    const request = (coepCredentialless && r.mode === 'no-cors')
      ? new Request(r, { credentials: 'omit' })
      : r;

    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.status === 0) return response;
          const newHeaders = new Headers(response.headers);
          newHeaders.set(
            'Cross-Origin-Embedder-Policy',
            coepCredentialless ? 'credentialless' : 'require-corp'
          );
          if (!coepCredentialless) {
            newHeaders.set('Cross-Origin-Resource-Policy', 'cross-origin');
          }
          newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');
          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers: newHeaders,
          });
        })
        .catch((err) => console.error('[coi-sw] fetch failed', err))
    );
  });
} else {
  // Page context — register the service worker.
  (() => {
    const reloadedBySelf = window.sessionStorage.getItem('coiReloadedBySelf');
    window.sessionStorage.removeItem('coiReloadedBySelf');
    const coepDegrading = reloadedBySelf === 'coepdegrade';

    window.coi = {
      shouldRegister: () => !reloadedBySelf,
      shouldDeregister: () => false,
      coepCredentialless: () => true,
      coepDegrade: () => true,
      doReload: () => window.location.reload(),
      quiet: false,
      ...window.coi,
    };

    const n = navigator;
    if (n.serviceWorker && n.serviceWorker.controller) {
      n.serviceWorker.controller.postMessage({
        type: 'coepCredentialless',
        value: (coepDegrading || !window.crossOriginIsolated)
          ? false
          : window.coi.coepCredentialless(),
      });
      if (window.coi.shouldDeregister()) {
        n.serviceWorker.controller.postMessage({ type: 'deregister' });
      }
    }

    // If we're already cross-origin isolated, nothing to do.
    if (!window.crossOriginIsolated && !coepDegrading && window.coi.shouldRegister()) {
      if (!n.serviceWorker || !n.serviceWorker.register) {
        if (!window.coi.quiet) {
          console.error(
            '[coi-sw] Service Worker unavailable — page will not be cross-origin isolated.'
          );
        }
        return;
      }

      n.serviceWorker.register(window.document.currentScript.src).then(
        (registration) => {
          if (!window.coi.quiet) {
            console.log('[coi-sw] registered, scope:', registration.scope);
          }
          registration.addEventListener('updatefound', () => {
            window.sessionStorage.setItem('coiReloadedBySelf', 'updatedSW');
            window.coi.doReload();
          });
          if (registration.active && !n.serviceWorker.controller) {
            if (!window.coi.quiet) {
              console.log('[coi-sw] reloading to take control...');
            }
            window.sessionStorage.setItem('coiReloadedBySelf', 'notControlling');
            window.coi.doReload();
          }
        },
        (err) => {
          if (!window.coi.quiet) {
            console.error('[coi-sw] registration failed:', err);
          }
          if (window.coi.coepDegrade()) {
            window.sessionStorage.setItem('coiReloadedBySelf', 'coepdegrade');
            window.coi.doReload();
          }
        }
      );
    }
  })();
}
