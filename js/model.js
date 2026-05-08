// ============================================
// AGENTMESH // Gemma 4 model loader (Transformers.js + WebGPU)
// ============================================

// Use ESM CDN for Transformers.js v3 (WebGPU support)
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';

// Model candidates in order of preference.
// On production hosts we try Gemma first (qualifies for the Gemma challenge),
// then fall back to known-good small ONNX/WebGPU models so the app always runs.
//
// `dtypes` is the list of quantizations to try for that candidate, in order.
// q4f16 = 4-bit weights, fp16 activations -> best WebGPU compatibility
// q4    = pure int4 -> may lack WebGPU kernels for some models (e.g. Gemma)
// fp16  = full precision fp16 -> always works but largest
const MODEL_CANDIDATES = [
  { id: 'onnx-community/gemma-3-1b-it-ONNX-GQA', label: 'gemma-3-1b-it', family: 'gemma', size: '~860MB', dtypes: ['q4f16', 'q4'] },
  { id: 'HuggingFaceTB/SmolLM2-360M-Instruct', label: 'smollm2-360m', family: 'smollm', size: '~270MB', dtypes: ['q4f16', 'q4'] },
  { id: 'HuggingFaceTB/SmolLM2-135M-Instruct', label: 'smollm2-135m', family: 'smollm', size: '~95MB', dtypes: ['q4f16', 'q4'] },
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

    onProgress({ progress: 3, status: 'Verifying WebGPU adapter...' });
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not available');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');

    onProgress({ progress: 5, status: 'Selecting model...' });

    let lastError = null;
    let lastDetail = 'unknown';
    for (const candidate of MODEL_CANDIDATES) {
      const dtypes = candidate.dtypes || ['q4f16', 'q4'];
      for (const dtype of dtypes) {
        try {
          onProgress({
            progress: 6,
            status: `Loading ${candidate.label} (${dtype})...`,
            file: candidate.id,
          });

          const { pipeline } = this.transformers;
          const pipe = await pipeline('text-generation', candidate.id, {
            device: 'webgpu',
            dtype,
            progress_callback: (data) => {
              // data: { status, file, progress, loaded, total }
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
          this.ready = true;

          onProgress({ progress: 98, status: 'Warming up model...' });
          await this.warmup();

          onProgress({ progress: 100, status: `Ready · ${candidate.label}` });
          return { id: candidate.id, label: candidate.label, family: candidate.family };
        } catch (err) {
          // Surface as much info as possible — Transformers.js + ORT errors
          // are sometimes plain numbers, sometimes Error objects.
          const detail = this.describeError(err);
          console.warn(`[model] ${candidate.id} dtype=${dtype} failed:`, detail, err);
          lastError = err;
          lastDetail = `${candidate.label} (${dtype}): ${detail}`;
          onProgress({
            progress: 6,
            status: `${candidate.label} ${dtype} unavailable: ${String(detail).slice(0, 80)}`,
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
