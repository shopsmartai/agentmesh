// ============================================
// AGENTMESH // Model runtime — local worker OR cloud Gemini API
// ============================================
// Two backends behind one interface. agents.js calls model.chat(messages,
// opts) and the runtime routes:
//   * Local mode  -> Web Worker running Transformers.js + Gemma 4 E2B
//   * Cloud mode  -> Google Gemini API using the user's own free key
//
// The cloud path is BYOK only. Keys live in the user's browser
// localStorage. We never embed a key in this code or transmit it
// anywhere except directly to Google.
// ============================================

import * as settings from './settings.js';

const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

class ModelRuntime {
  constructor() {
    this.worker = null;
    this.ready = false;
    this.modelId = null;
    this.modelLabel = null;
    this.device = null;

    this.loadPromise = null;
    this.loadResolve = null;
    this.loadReject = null;
    this.loadOnProgress = null;

    this.chatPending = new Map();
    this.nextRequestId = 1;

    // Cloud mode does not need a worker or a load step. Calling
    // setupCloudMode() flips this runtime into a "ready" state without
    // downloading anything. Local mode goes through load() as before.
    this.cloudReady = false;
  }

  // -----------------------------------
  // Mode selection
  // -----------------------------------

  /**
   * If the user has cloud mode enabled with a key set, mark the runtime
   * ready immediately and skip the 3 GB local model download. Returns the
   * info object the boot UI expects.
   */
  setupCloudMode() {
    if (settings.getMode() !== 'cloud' || !settings.getApiKey()) return null;
    const cloudModel = settings.getCloudModel();
    this.cloudReady = true;
    this.ready = true;
    this.modelId = cloudModel;
    this.modelLabel = `${cloudModel} (cloud)`;
    this.device = 'cloud';
    return {
      id: cloudModel,
      label: cloudModel,
      family: 'gemma',
      device: 'cloud',
      multimodal: false,
    };
  }

  // -----------------------------------
  // Local worker path
  // -----------------------------------

  spawnWorker() {
    if (this.worker) return this.worker;
    this.worker = new Worker(new URL('./model.worker.js', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', (ev) => this._onMessage(ev.data));
    this.worker.addEventListener('error', (ev) => {
      console.error('[model] worker error:', ev.message, ev);
      if (this.loadReject) {
        this.loadReject(new Error(`Worker error: ${ev.message}`));
        this.loadResolve = this.loadReject = this.loadOnProgress = null;
        this.loadPromise = null;
      }
    });
    return this.worker;
  }

  load(onProgress) {
    // Cloud mode short-circuits: we never download a model. The boot
    // sequence calls setupCloudMode() first; if that succeeded, load()
    // becomes a no-op.
    if (this.cloudReady) {
      onProgress?.({ progress: 100, status: `Ready · ${this.modelLabel}` });
      return Promise.resolve({
        id: this.modelId,
        label: settings.getCloudModel(),
        family: 'gemma',
        device: 'cloud',
      });
    }

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

  // -----------------------------------
  // Unified chat — routes by mode
  // -----------------------------------

  async chat(messages, opts = {}) {
    if (!this.ready) throw new Error('Model not loaded');

    if (settings.getMode() === 'cloud' && settings.getApiKey()) {
      return this._chatViaGemini(messages, opts);
    }
    return this._chatViaWorker(messages, opts);
  }

  _chatViaWorker(messages, opts) {
    const requestId = this.nextRequestId++;
    const { maxTokens, temperature, onToken, signal, image } = opts;

    const promise = new Promise((resolve, reject) => {
      this.chatPending.set(requestId, { onToken, resolve, reject });
    });

    if (signal) {
      const onAbort = () => {
        this.worker?.postMessage({ type: 'abort', requestId });
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener('abort', onAbort, { once: true });
    }

    this.worker.postMessage({
      type: 'chat',
      requestId,
      messages,
      opts: { maxTokens, temperature, image },
    });

    return promise;
  }

  /**
   * Stream a Gemini chat completion using the user's BYOK key. Converts our
   * standard {role, content} messages into Gemini's {systemInstruction,
   * contents[role/parts]} shape, opens an SSE stream, and emits tokens to
   * opts.onToken as they arrive.
   */
  async _chatViaGemini(messages, opts) {
    const { maxTokens = 512, temperature = 0, onToken, signal } = opts;

    const apiKey = settings.getApiKey();
    if (!apiKey) throw new Error('Cloud mode is on but no API key is set in Settings');

    // Convert messages: extract system prompt to systemInstruction, map
    // remaining roles. Gemini uses 'model' (not 'assistant') for replies.
    const systemMsgs = messages.filter((m) => m.role === 'system');
    const turns = messages.filter((m) => m.role !== 'system');
    const systemInstruction = systemMsgs.length > 0
      ? { parts: [{ text: systemMsgs.map((m) => m.content).join('\n\n') }] }
      : undefined;
    const contents = turns.map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: typeof m.content === 'string' ? m.content : String(m.content) }],
    }));

    const cloudModel = settings.getCloudModel();
    const url = `${GEMINI_BASE}/${encodeURIComponent(cloudModel)}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`;

    const body = JSON.stringify({
      systemInstruction,
      contents,
      generationConfig: {
        temperature,
        maxOutputTokens: maxTokens,
      },
    });

    let response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal,
      });
    } catch (err) {
      throw new Error(`Network error calling Gemini API: ${err.message}`);
    }

    if (!response.ok) {
      // Pull the error JSON if Google sent one — it usually has a useful
      // message like "API key not valid" or "model not found".
      let detail = `HTTP ${response.status}`;
      try {
        const errBody = await response.json();
        detail = errBody?.error?.message || detail;
      } catch { /* ignore */ }
      throw new Error(`Gemini API: ${detail}`);
    }

    return this._consumeGeminiStream(response, onToken);
  }

  async _consumeGeminiStream(response, onToken) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by blank lines. Each frame's lines start
      // with "data: ". Extract complete frames from the buffer and leave
      // any trailing partial frame for the next iteration.
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const dataLines = frame
          .split('\n')
          .filter((l) => l.startsWith('data: '))
          .map((l) => l.slice(6));
        if (dataLines.length === 0) continue;
        const payload = dataLines.join('\n');
        if (payload === '[DONE]') return fullText;
        try {
          const obj = JSON.parse(payload);
          const parts = obj?.candidates?.[0]?.content?.parts || [];
          for (const part of parts) {
            const text = part.text || '';
            if (text) {
              fullText += text;
              try { onToken?.(text); } catch (e) { console.warn('[model] onToken threw:', e); }
            }
          }
        } catch (e) {
          // Ignore malformed frames — Gemini sometimes sends keepalives or
          // partial JSON during slow connections.
        }
      }
    }

    // Flush any trailing buffered frame.
    if (buffer.trim()) {
      const dataLines = buffer
        .split('\n')
        .filter((l) => l.startsWith('data: '))
        .map((l) => l.slice(6));
      if (dataLines.length > 0) {
        try {
          const obj = JSON.parse(dataLines.join('\n'));
          const parts = obj?.candidates?.[0]?.content?.parts || [];
          for (const part of parts) {
            const text = part.text || '';
            if (text) {
              fullText += text;
              try { onToken?.(text); } catch { /* ignore */ }
            }
          }
        } catch { /* ignore */ }
      }
    }

    return fullText;
  }

  // -----------------------------------
  // Capabilities
  // -----------------------------------

  /**
   * True if the loaded model can accept image input. Both backends are
   * text-only in the current build. Multimodal Gemma 4 ships in
   * `model.worker.js` as documented future work.
   */
  get supportsImages() {
    return false;
  }

  abort() {
    // No-op convenience method. Per-request aborts go through opts.signal.
  }

  // -----------------------------------
  // Worker message dispatch
  // -----------------------------------

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
