// ============================================
// AGENTMESH // Gemma 4 model loader (Transformers.js + WebGPU)
// ============================================

// Use ESM CDN for Transformers.js. Pinned to 3.7.6 because 3.5.1 surfaces
// raw numeric ORT errors that can't be diagnosed and prevent any model from
// loading on either WebGPU or WASM.
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';

// Model candidates in order of preference.
// `attempts` is the list of {device, dtype} combos to try for that candidate.
// We try WebGPU first (fast), then fall back to WASM (slow but always works
// when ORT WebGPU kernels are missing or buffer limits are exceeded).
const MODEL_CANDIDATES = [
  {
    id: 'onnx-community/gemma-3-1b-it-ONNX-GQA',
    label: 'gemma-3-1b-it', family: 'gemma', size: '~860MB',
    attempts: [
      { device: 'webgpu', dtype: 'q4f16' },
      { device: 'webgpu', dtype: 'q4' },
    ],
  },
  {
    id: 'HuggingFaceTB/SmolLM2-360M-Instruct',
    label: 'smollm2-360m', family: 'smollm', size: '~270MB',
    attempts: [
      { device: 'webgpu', dtype: 'q4' },
      { device: 'webgpu', dtype: 'q4f16' },
      { device: 'wasm', dtype: 'q4' },
    ],
  },
  {
    id: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    label: 'smollm2-135m', family: 'smollm', size: '~95MB',
    attempts: [
      { device: 'webgpu', dtype: 'q4' },
      { device: 'webgpu', dtype: 'q4f16' },
      { device: 'wasm', dtype: 'q4' },
    ],
  },
];

class ModelRuntime {
  constructor() {
    this.transformers = null;
    this.pipeline = null;
    this.tokenizer = null;
    this.modelId = null;
    this.modelLabel = null;
    this.ready = false;
  }

  async loadTransformersJS() {
    const tf = await import(TRANSFORMERS_CDN);
    // Don't use local model paths, fetch from HF Hub.
    // We disable the Transformers.js Cache Storage layer because partial
    // downloads from a failed candidate get cached and poison every retry
    // (ORT then throws unrecoverable numeric errors on the corrupted blob).
    // HTTP caching via HF Hub's Cache-Control headers still works for
    // normal page reloads, which is what users actually experience.
    tf.env.allowLocalModels = false;
    tf.env.useBrowserCache = false;
    return tf;
  }

  /**
   * Try each model in order. Returns first one that loads successfully.
   * @param {(p: {progress: number, status: string, file?: string}) => void} onProgress
   */
  async load(onProgress) {
    onProgress({ progress: 1, status: 'Initializing Transformers.js v3...' });
    this.transformers = await this.loadTransformersJS();

    onProgress({ progress: 3, status: 'Detecting backends...' });
    const hasWebGPU = await this.hasWorkingWebGPU();

    onProgress({ progress: 5, status: 'Selecting model...' });

    let lastError = null;
    let lastDetail = 'unknown';
    for (const candidate of MODEL_CANDIDATES) {
      const attempts = candidate.attempts || [{ device: 'webgpu', dtype: 'q4' }];
      for (const attempt of attempts) {
        if (attempt.device === 'webgpu' && !hasWebGPU) continue;
        const { device, dtype } = attempt;
        try {
          onProgress({
            progress: 6,
            status: `Loading ${candidate.label} (${device}/${dtype})...`,
            file: candidate.id,
          });

          const { pipeline } = this.transformers;
          const pipe = await pipeline('text-generation', candidate.id, {
            device,
            dtype,
            progress_callback: (data) => {
              const pct = data.progress
                ? Math.min(95, 10 + Math.round(data.progress * 0.85))
                : 10;
              onProgress({
                progress: pct,
                status: this.formatStatus(data),
                file: data.file,
              });
            },
          });

          this.pipeline = pipe;
          this.tokenizer = pipe.tokenizer;
          this.modelId = candidate.id;
          this.modelLabel = candidate.label;
          this.device = device;
          this.ready = true;

          onProgress({ progress: 98, status: 'Warming up model...' });
          await this.warmup();

          onProgress({ progress: 100, status: `Ready · ${candidate.label} (${device})` });
          return { id: candidate.id, label: candidate.label, family: candidate.family, device };
        } catch (err) {
          const detail = this.describeError(err);
          console.warn(`[model] ${candidate.id} ${device}/${dtype} failed:`, detail, err);
          lastError = err;
          lastDetail = `${candidate.label} ${device}/${dtype}: ${detail}`;
          onProgress({
            progress: 6,
            status: `${candidate.label} ${device}/${dtype} unavailable: ${String(detail).slice(0, 80)}`,
          });
        }
      }
    }
    throw new Error(`All model candidates failed. Last error: ${lastDetail}`);
  }

  formatStatus(data) {
    switch (data.status) {
      case 'initiate':
        return `Fetching ${this.fileLabel(data.file)}...`;
      case 'download':
        return `Downloading ${this.fileLabel(data.file)}`;
      case 'progress':
        if (data.total) {
          const mb = (n) => (n / 1024 / 1024).toFixed(1);
          return `Streaming ${this.fileLabel(data.file)} · ${mb(data.loaded)}MB / ${mb(data.total)}MB`;
        }
        return `Streaming ${this.fileLabel(data.file)}`;
      case 'done':
        return `${this.fileLabel(data.file)} cached`;
      case 'ready':
        return 'Compiling shaders...';
      default:
        return data.status || 'Working...';
    }
  }

  fileLabel(file) {
    if (!file) return 'file';
    return file.split('/').pop();
  }

  async hasWorkingWebGPU() {
    if (!('gpu' in navigator)) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      return !!adapter;
    } catch {
      return false;
    }
  }

  /**
   * Best-effort error description. ORT sometimes throws a raw number
   * (a pointer into WASM memory), so look at type and known fields.
   */
  describeError(err) {
    if (err == null) return 'unknown';
    if (typeof err === 'number') {
      return `ORT runtime error code ${err} (likely missing WebGPU kernel for this dtype)`;
    }
    if (typeof err === 'string') return err;
    if (err.message) return err.message;
    if (err.name) return err.name;
    try {
      return JSON.stringify(err);
    } catch {
      return String(err);
    }
  }

  async warmup() {
    if (!this.pipeline) return;
    await this.pipeline('Hello', { max_new_tokens: 4, do_sample: false });
  }

  /**
   * Run a chat completion with optional streaming.
   * @param {Array<{role: string, content: string}>} messages
   * @param {{maxTokens?: number, temperature?: number, onToken?: (t: string) => void, signal?: AbortSignal}} opts
   * @returns {Promise<string>}
   */
  async chat(messages, opts = {}) {
    if (!this.ready) throw new Error('Model not loaded');

    const {
      maxTokens = 512,
      temperature = 0.7,
      onToken,
      signal,
    } = opts;

    const { TextStreamer } = this.transformers;
    let fullText = '';

    const streamer = onToken
      ? new TextStreamer(this.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (text) => {
            fullText += text;
            onToken(text);
            if (signal?.aborted) {
              throw new Error('aborted');
            }
          },
        })
      : null;

    const out = await this.pipeline(messages, {
      max_new_tokens: maxTokens,
      do_sample: temperature > 0,
      temperature,
      streamer,
    });

    // Non-streaming path: extract assistant content from the result
    if (!streamer) {
      const last = out?.[0]?.generated_text;
      if (Array.isArray(last)) {
        const lastMsg = last[last.length - 1];
        return lastMsg?.content || '';
      }
      return typeof last === 'string' ? last : '';
    }

    return fullText;
  }

  abort() {
    // Pipeline doesn't expose mid-generation abort cleanly in Transformers.js v3;
    // streamer raises in the callback when signal is aborted, which the pipeline
    // surfaces as a thrown error. Caller handles cleanup.
  }
}

export const model = new ModelRuntime();
