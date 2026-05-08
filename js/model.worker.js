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
//   { type: 'chat', requestId, messages, opts: { maxTokens, temperature, image? } }
//   { type: 'abort', requestId }
//
// Worker -> Main:
//   { type: 'progress', progress, status, file }
//   { type: 'loaded', info: { id, label, family, device, multimodal } }
//   { type: 'load-error', message }
//   { type: 'token', requestId, text }
//   { type: 'chat-done', requestId, fullText }
//   { type: 'chat-error', requestId, message }
// ============================================

const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

// Gemma 4 needs per-component dtypes — audio/vision encoders don't have
// q4f16 WebGPU kernels for some Chrome drivers, so they ship fp16 while the
// decoder + embed_tokens stay quantized. This combo matches the official
// webml-community/Gemma-4-WebGPU demo and is verified-working on prod.
const GEMMA_4_DTYPE_OPTIMAL = {
  audio_encoder: 'fp16',
  vision_encoder: 'fp16',
  embed_tokens: 'q4f16',
  decoder_model_merged: 'q4f16',
};
// All-q4 fallback if the optimal mix fails (older drivers).
const GEMMA_4_DTYPE_FALLBACK = {
  audio_encoder: 'q4',
  vision_encoder: 'q4',
  embed_tokens: 'q4',
  decoder_model_merged: 'q4',
};

const MODEL_GEMMA_4_E2B = {
  id: 'onnx-community/gemma-4-E2B-it-ONNX',
  label: 'gemma-4-E2B-it', family: 'gemma', size: '~3.4GB',
  attempts: [
    { device: 'webgpu', dtype: GEMMA_4_DTYPE_OPTIMAL },
    { device: 'webgpu', dtype: GEMMA_4_DTYPE_FALLBACK },
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
// SmolLM2 path: high-level pipeline
let pipeline = null;
let pipelineTokenizer = null;
// Gemma 4 path: low-level processor + conditional generation model
// (required for image/audio inputs and exposes proper apply_chat_template).
let gemmaProcessor = null;
let gemmaModel = null;
let modelInfo = null;
const aborted = new Set();

function post(msg) { self.postMessage(msg); }

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
    case 'progress_total':
      // v4 fires this with cumulative-load progress as a percentage.
      return `Loading model · ${Math.round(data.progress || 0)}%`;
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

function makeProgressCb() {
  return (data) => {
    const pct = data.progress
      ? Math.min(95, 10 + Math.round(data.progress * 0.85))
      : 10;
    const force = data.status !== 'progress';
    postProgress({
      type: 'progress', progress: pct,
      status: formatStatus(data), file: data.file,
    }, force);
  };
}

async function loadGemma4(candidate, attempt) {
  const { device, dtype } = attempt;
  const { AutoProcessor, Gemma4ForConditionalGeneration } = transformers;

  const processor = await AutoProcessor.from_pretrained(candidate.id, {
    progress_callback: makeProgressCb(),
  });
  const model = await Gemma4ForConditionalGeneration.from_pretrained(candidate.id, {
    device, dtype,
    progress_callback: makeProgressCb(),
  });

  // Warmup with a tiny generation so first user query doesn't pay the
  // shader-compile cost on the critical path. Processor signature is
  // (prompt, image, audio, options) — pass null for both modalities here.
  post({ type: 'progress', progress: 98, status: 'Warming up model...' });
  const warmupPrompt = processor.apply_chat_template(
    [{ role: 'user', content: 'Hello' }],
    { add_generation_prompt: true, enable_thinking: false }
  );
  const warmupInputs = await processor(warmupPrompt, null, null, { add_special_tokens: false });
  await model.generate({ ...warmupInputs, max_new_tokens: 2, do_sample: false });

  gemmaProcessor = processor;
  gemmaModel = model;
  pipeline = null;
  pipelineTokenizer = null;
}

async function loadPipelineModel(candidate, attempt) {
  const { device, dtype } = attempt;
  const { pipeline: makePipe } = transformers;

  const pipe = await makePipe('text-generation', candidate.id, {
    device, dtype,
    progress_callback: makeProgressCb(),
  });

  post({ type: 'progress', progress: 98, status: 'Warming up model...' });
  await pipe('Hello', { max_new_tokens: 4, do_sample: false });

  pipeline = pipe;
  pipelineTokenizer = pipe.tokenizer;
  gemmaProcessor = null;
  gemmaModel = null;
}

async function loadModel(forceCandidate) {
  if (modelInfo) {
    post({ type: 'loaded', info: modelInfo });
    return;
  }

  try {
    post({ type: 'progress', progress: 1, status: 'Initializing Transformers.js v4...' });
    transformers = await import(TRANSFORMERS_CDN);
    transformers.env.allowLocalModels = false;
    transformers.env.useBrowserCache = true;

    post({ type: 'progress', progress: 3, status: 'Detecting backends...' });
    const hasWebGPU = await hasWorkingWebGPU();

    post({ type: 'progress', progress: 5, status: 'Selecting model...' });

    let lastDetail = 'unknown';
    const candidates = pickCandidates(forceCandidate);
    for (const candidate of candidates) {
      const attempts = candidate.attempts || [{ device: 'webgpu', dtype: 'q4' }];
      for (const attempt of attempts) {
        if (attempt.device === 'webgpu' && !hasWebGPU) continue;
        const { device, dtype } = attempt;
        const dtypeLabel = typeof dtype === 'string'
          ? dtype
          : (dtype.decoder_model_merged || 'mixed');
        try {
          post({
            type: 'progress', progress: 6,
            status: `Loading ${candidate.label} (${device}/${dtypeLabel})...`,
            file: candidate.id,
          });

          if (candidate.family === 'gemma') {
            await loadGemma4(candidate, attempt);
          } else {
            await loadPipelineModel(candidate, attempt);
          }

          modelInfo = {
            id: candidate.id, label: candidate.label,
            family: candidate.family, device,
            multimodal: candidate.family === 'gemma',
          };

          post({ type: 'progress', progress: 100, status: `Ready · ${candidate.label} (${device})` });
          post({ type: 'loaded', info: modelInfo });
          return;
        } catch (err) {
          const detail = describeError(err);
          console.warn(`[model.worker] ${candidate.id} ${device}/${dtypeLabel} failed:`, detail, err);
          lastDetail = `${candidate.label} ${device}/${dtypeLabel}: ${detail}`;
          post({
            type: 'progress', progress: 6,
            status: `${candidate.label} ${device}/${dtypeLabel} unavailable: ${String(detail).slice(0, 80)}`,
          });
        }
      }
    }
    throw new Error(`All model candidates failed. Last error: ${lastDetail}`);
  } catch (err) {
    post({ type: 'load-error', message: describeError(err) });
  }
}

async function chatViaPipeline({ requestId, messages, maxTokens, temperature }) {
  const { TextStreamer } = transformers;
  let fullText = '';

  const streamer = new TextStreamer(pipelineTokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      if (aborted.has(requestId)) throw new Error('aborted');
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

  if (!fullText) {
    const last = out?.[0]?.generated_text;
    if (Array.isArray(last)) {
      const lastMsg = last[last.length - 1];
      fullText = lastMsg?.content || '';
    } else if (typeof last === 'string') {
      fullText = last;
    }
  }

  return fullText;
}

async function chatViaGemma({ requestId, messages, maxTokens, temperature, image }) {
  const { TextStreamer, RawImage } = transformers;

  // Convert chat messages to Gemma 4's content-block format. If the caller
  // passed an `image` (Blob or URL), prepend it as the first block of the
  // first user message — Gemma 4 docs require image *before* text.
  const gemmaMessages = messages.map((m) => {
    if (typeof m.content === 'string') {
      return { role: m.role, content: [{ type: 'text', text: m.content }] };
    }
    return m; // already in block format
  });
  if (image && gemmaMessages.length > 0) {
    const firstUserIdx = gemmaMessages.findIndex((m) => m.role === 'user');
    if (firstUserIdx >= 0) {
      gemmaMessages[firstUserIdx] = {
        ...gemmaMessages[firstUserIdx],
        content: [{ type: 'image' }, ...gemmaMessages[firstUserIdx].content],
      };
    }
  }

  const prompt = gemmaProcessor.apply_chat_template(gemmaMessages, {
    add_generation_prompt: true,
    enable_thinking: false,
  });

  let imageInput = null;
  if (image) {
    if (image instanceof Blob) {
      imageInput = await RawImage.fromBlob(image);
    } else if (typeof image === 'string') {
      imageInput = await RawImage.read(image);
    }
  }

  // Processor signature: (prompt, image, audio, options). Audio support is
  // wired but we don't expose audio input from the UI yet, so always null.
  const inputs = await gemmaProcessor(prompt, imageInput, null, { add_special_tokens: false });

  let fullText = '';
  const streamer = new TextStreamer(gemmaProcessor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      if (aborted.has(requestId)) throw new Error('aborted');
      fullText += text;
      post({ type: 'token', requestId, text });
    },
  });

  const outputs = await gemmaModel.generate({
    ...inputs,
    max_new_tokens: maxTokens,
    do_sample: temperature > 0,
    temperature,
    top_p: 0.95,
    top_k: 64,
    streamer,
  });

  // Defensive fallback: if streaming yielded nothing, decode the new tokens.
  if (!fullText) {
    const promptLen = inputs.input_ids.dims.at(-1);
    const newTokens = outputs.slice(null, [promptLen, null]);
    const decoded = gemmaProcessor.batch_decode(newTokens, { skip_special_tokens: true });
    fullText = (decoded?.[0] || '').trim();
  }

  return fullText;
}

async function runChat({ requestId, messages, opts }) {
  if (!pipeline && !gemmaModel) {
    post({ type: 'chat-error', requestId, message: 'Model not loaded' });
    return;
  }

  const { maxTokens = 512, temperature = 0.7, image } = opts || {};

  try {
    let fullText;
    if (gemmaModel) {
      fullText = await chatViaGemma({ requestId, messages, maxTokens, temperature, image });
    } else {
      fullText = await chatViaPipeline({ requestId, messages, maxTokens, temperature });
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
