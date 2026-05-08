// ============================================
// AGENTMESH // Model worker
// ============================================
// Runs Transformers.js + ORT WebGPU off the main thread so heavy work like
// Gemma 4 shader compilation doesn't freeze the page. The main thread sees
// a streaming-token interface via postMessage and continues to render the
// swarm UI, accept input, etc.
//
// MESSAGE PROTOCOL
//
// Main -> Worker:
//   { type: 'load', forceCandidate?: 'gemma'|'smollm'|null }
//   { type: 'chat', requestId, messages, opts: { maxTokens, temperature } }
//   { type: 'abort', requestId }
//
// Worker -> Main:
//   { type: 'progress', progress, status, file }
//   { type: 'loaded', info: { id, label, family, device } }
//   { type: 'load-error', message }
//   { type: 'token', requestId, text }
//   { type: 'chat-done', requestId, fullText }
//   { type: 'chat-error', requestId, message }
// ============================================

const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

const MODEL_GEMMA_4_E2B = {
  id: 'onnx-community/gemma-4-E2B-it-ONNX',
  label: 'gemma-4-E2B-it', family: 'gemma', size: '~3.1GB',
  attempts: [
    { device: 'webgpu', dtype: 'q4f16' },
    { device: 'webgpu', dtype: 'q4' },
  ],
};

const MODEL_SMOLLM2_360M = {
  id: 'HuggingFaceTB/SmolLM2-360M-Instruct',
  label: 'smollm2-360m', family: 'smollm', size: '~270MB',
  attempts: [
    { device: 'webgpu', dtype: 'q4' },
    { device: 'webgpu', dtype: 'q4f16' },
    { device: 'wasm', dtype: 'q4' },
  ],
};

const MODEL_SMOLLM2_135M = {
  id: 'HuggingFaceTB/SmolLM2-135M-Instruct',
  label: 'smollm2-135m', family: 'smollm', size: '~95MB',
  attempts: [
    { device: 'webgpu', dtype: 'q4' },
    { device: 'webgpu', dtype: 'q4f16' },
    { device: 'wasm', dtype: 'q4' },
  ],
};

function pickCandidates(forceCandidate) {
  if (forceCandidate === 'smollm' || forceCandidate === 'smollm2') {
    return [MODEL_SMOLLM2_360M, MODEL_SMOLLM2_135M];
  }
  return [MODEL_GEMMA_4_E2B, MODEL_SMOLLM2_360M, MODEL_SMOLLM2_135M];
}

let transformers = null;
let pipeline = null;
let tokenizer = null;
let modelInfo = null;
const aborted = new Set(); // requestIds the main thread asked to cancel

function post(msg) { self.postMessage(msg); }

// Throttled progress poster. Transformers.js fires progress callbacks for
// every fetch chunk during the 3 GB Gemma 4 download — that's hundreds of
// events per second. Forwarding each one to the main thread floods its
// message queue and freezes the page. Coalesce to one message per 100ms,
// but always flush 'initiate' / 'done' / 'ready' / final progress so the
// UI doesn't miss state transitions.
let lastProgressPostAt = 0;
const PROGRESS_THROTTLE_MS = 100;
function postProgress(payload, force = false) {
  const now = performance.now();
  if (force || now - lastProgressPostAt >= PROGRESS_THROTTLE_MS) {
    lastProgressPostAt = now;
    post(payload);
  }
}

function describeError(err) {
  if (err == null) return 'unknown';
  if (typeof err === 'number') {
    return `ORT runtime error code ${err} (likely missing WebGPU kernel for this dtype)`;
  }
  if (typeof err === 'string') return err;
  if (err.message) return err.message;
  if (err.name) return err.name;
  try { return JSON.stringify(err); } catch { return String(err); }
}

function fileLabel(file) {
  if (!file) return 'file';
  return file.split('/').pop();
}

function formatStatus(data) {
  switch (data.status) {
    case 'initiate': return `Fetching ${fileLabel(data.file)}...`;
    case 'download': return `Downloading ${fileLabel(data.file)}`;
    case 'progress':
      if (data.total) {
        const mb = (n) => (n / 1024 / 1024).toFixed(1);
        return `Streaming ${fileLabel(data.file)} · ${mb(data.loaded)}MB / ${mb(data.total)}MB`;
      }
      return `Streaming ${fileLabel(data.file)}`;
    case 'done': return `${fileLabel(data.file)} cached`;
    case 'ready': return 'Compiling shaders...';
    default: return data.status || 'Working...';
  }
}

async function hasWorkingWebGPU() {
  if (!('gpu' in self.navigator)) return false;
  try {
    const adapter = await self.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    return !!adapter;
  } catch { return false; }
}

async function loadModel(forceCandidate) {
  if (modelInfo) {
    post({ type: 'loaded', info: modelInfo });
    return;
  }

  try {
    post({ progress: 1, status: 'Initializing Transformers.js v4...', type: 'progress' });
    transformers = await import(TRANSFORMERS_CDN);
    transformers.env.allowLocalModels = false;
    transformers.env.useBrowserCache = true;

    post({ progress: 3, status: 'Detecting backends...', type: 'progress' });
    const hasWebGPU = await hasWorkingWebGPU();

    post({ progress: 5, status: 'Selecting model...', type: 'progress' });

    let lastDetail = 'unknown';
    const candidates = pickCandidates(forceCandidate);
    for (const candidate of candidates) {
      const attempts = candidate.attempts || [{ device: 'webgpu', dtype: 'q4' }];
      for (const attempt of attempts) {
        if (attempt.device === 'webgpu' && !hasWebGPU) continue;
        const { device, dtype } = attempt;
        try {
          post({
            type: 'progress', progress: 6,
            status: `Loading ${candidate.label} (${device}/${dtype})...`,
            file: candidate.id,
          });

          const { pipeline: makePipe } = transformers;
          const pipe = await makePipe('text-generation', candidate.id, {
            device, dtype,
            progress_callback: (data) => {
              const pct = data.progress
                ? Math.min(95, 10 + Math.round(data.progress * 0.85))
                : 10;
              // Force-flush state transitions; throttle 'progress' chunks.
              const force = data.status !== 'progress';
              postProgress({
                type: 'progress', progress: pct,
                status: formatStatus(data), file: data.file,
              }, force);
            },
          });

          pipeline = pipe;
          tokenizer = pipe.tokenizer;
          modelInfo = {
            id: candidate.id, label: candidate.label,
            family: candidate.family, device,
          };

          post({ type: 'progress', progress: 98, status: 'Warming up model...' });
          await pipe('Hello', { max_new_tokens: 4, do_sample: false });

          post({
            type: 'progress', progress: 100,
            status: `Ready · ${candidate.label} (${device})`,
          });
          post({ type: 'loaded', info: modelInfo });
          return;
        } catch (err) {
          const detail = describeError(err);
          console.warn(`[model.worker] ${candidate.id} ${device}/${dtype} failed:`, detail, err);
          lastDetail = `${candidate.label} ${device}/${dtype}: ${detail}`;
          post({
            type: 'progress', progress: 6,
            status: `${candidate.label} ${device}/${dtype} unavailable: ${String(detail).slice(0, 80)}`,
          });
        }
      }
    }
    throw new Error(`All model candidates failed. Last error: ${lastDetail}`);
  } catch (err) {
    post({ type: 'load-error', message: describeError(err) });
  }
}

async function runChat({ requestId, messages, opts }) {
  if (!pipeline) {
    post({ type: 'chat-error', requestId, message: 'Model not loaded' });
    return;
  }

  const { maxTokens = 512, temperature = 0.7 } = opts || {};
  let fullText = '';

  try {
    const { TextStreamer } = transformers;
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
        if (aborted.has(requestId)) {
          throw new Error('aborted');
        }
        fullText += text;
        post({ type: 'token', requestId, text });
      },
    });

    const out = await pipeline(messages, {
      max_new_tokens: maxTokens,
      do_sample: temperature > 0,
      temperature,
      streamer,
    });

    // Fallback: if streaming produced nothing (e.g. tokenizer didn't emit
    // intermediate text), recover the assistant content from the final
    // generated_text array.
    if (!fullText) {
      const last = out?.[0]?.generated_text;
      if (Array.isArray(last)) {
        const lastMsg = last[last.length - 1];
        fullText = lastMsg?.content || '';
      } else if (typeof last === 'string') {
        fullText = last;
      }
    }

    post({ type: 'chat-done', requestId, fullText });
  } catch (err) {
    post({ type: 'chat-error', requestId, message: describeError(err) });
  } finally {
    aborted.delete(requestId);
  }
}

self.addEventListener('message', (ev) => {
  const msg = ev.data || {};
  switch (msg.type) {
    case 'load':
      loadModel(msg.forceCandidate);
      break;
    case 'chat':
      runChat(msg);
      break;
    case 'abort':
      if (msg.requestId != null) aborted.add(msg.requestId);
      break;
    default:
      console.warn('[model.worker] unknown message:', msg.type);
  }
});
