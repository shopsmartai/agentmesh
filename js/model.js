// ============================================
// AGENTMESH // Main-thread proxy for the model worker
// ============================================
// Heavy work (Transformers.js, ORT WebGPU, Gemma 4 shader compilation) runs
// inside `model.worker.js`. This file exposes the same `model.load()` and
// `model.chat(messages, opts)` interface that agents.js depends on, but
// forwards every call across postMessage so the page stays responsive.
// ============================================

class ModelRuntime {
  constructor() {
    this.worker = null;
    this.ready = false;
    this.modelId = null;
    this.modelLabel = null;
    this.device = null;

    // Pending load promise + progress callback
    this.loadPromise = null;
    this.loadResolve = null;
    this.loadReject = null;
    this.loadOnProgress = null;

    // requestId -> { onToken, resolve, reject }
    this.chatPending = new Map();
    this.nextRequestId = 1;
  }

  spawnWorker() {
    if (this.worker) return this.worker;
    this.worker = new Worker(new URL('./model.worker.js', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', (ev) => this._onMessage(ev.data));
    this.worker.addEventListener('error', (ev) => {
      console.error('[model] worker error:', ev.message, ev);
      // If we're mid-load, reject the load promise so the UI can show it.
      if (this.loadReject) {
        this.loadReject(new Error(`Worker error: ${ev.message}`));
        this.loadResolve = this.loadReject = this.loadOnProgress = null;
        this.loadPromise = null;
      }
    });
    return this.worker;
  }

  /**
   * Load the model. Calls onProgress with {progress, status, file} updates.
   * Resolves to {id, label, family, device} when ready.
   */
  load(onProgress) {
    if (this.loadPromise) return this.loadPromise;

    this.spawnWorker();
    this.loadOnProgress = onProgress || (() => {});

    const params = new URLSearchParams(location.search);
    const choice = params.get('model');
    const forceCandidate = (choice === 'smollm' || choice === 'smollm2') ? 'smollm' : null;

    this.loadPromise = new Promise((resolve, reject) => {
      this.loadResolve = resolve;
      this.loadReject = reject;
      this.worker.postMessage({ type: 'load', forceCandidate });
    });
    return this.loadPromise;
  }

  /**
   * Streaming chat completion. Forwards messages + opts to the worker;
   * tokens come back over postMessage and are dispatched to opts.onToken.
   * Returns a Promise<string> with the full generated text.
   */
  async chat(messages, opts = {}) {
    if (!this.ready) throw new Error('Model not loaded');

    const requestId = this.nextRequestId++;
    const { maxTokens, temperature, onToken, signal } = opts;

    const promise = new Promise((resolve, reject) => {
      this.chatPending.set(requestId, { onToken, resolve, reject });
    });

    if (signal) {
      const onAbort = () => {
        this.worker.postMessage({ type: 'abort', requestId });
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener('abort', onAbort, { once: true });
    }

    this.worker.postMessage({
      type: 'chat',
      requestId,
      messages,
      opts: { maxTokens, temperature },
    });

    return promise;
  }

  abort() {
    // No-op convenience method. Per-request aborts go through opts.signal.
  }

  _onMessage(msg) {
    if (!msg || typeof msg !== 'object') return;
    switch (msg.type) {
      case 'progress':
        this.loadOnProgress?.({
          progress: msg.progress,
          status: msg.status,
          file: msg.file,
        });
        break;
      case 'loaded':
        this.ready = true;
        this.modelId = msg.info.id;
        this.modelLabel = msg.info.label;
        this.device = msg.info.device;
        this.loadResolve?.(msg.info);
        this.loadResolve = this.loadReject = this.loadOnProgress = null;
        break;
      case 'load-error':
        this.loadReject?.(new Error(msg.message));
        this.loadResolve = this.loadReject = this.loadOnProgress = null;
        this.loadPromise = null;
        break;
      case 'token': {
        const pending = this.chatPending.get(msg.requestId);
        if (pending?.onToken) {
          try { pending.onToken(msg.text); } catch (e) { console.warn('[model] onToken threw:', e); }
        }
        break;
      }
      case 'chat-done': {
        const pending = this.chatPending.get(msg.requestId);
        if (pending) {
          this.chatPending.delete(msg.requestId);
          pending.resolve(msg.fullText || '');
        }
        break;
      }
      case 'chat-error': {
        const pending = this.chatPending.get(msg.requestId);
        if (pending) {
          this.chatPending.delete(msg.requestId);
          pending.reject(new Error(msg.message));
        }
        break;
      }
      default:
        // ignore
    }
  }
}

export const model = new ModelRuntime();
