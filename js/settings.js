// ============================================
// AGENTMESH // Settings (localStorage-backed)
// ============================================
// Inference mode + cloud API key state, persisted only in the user's
// browser. Nothing is sent to any server we control. The key never
// touches our infrastructure.
//
// Modes:
//   'local' — default. Runs Gemma 4 E2B in a Web Worker on the user's GPU.
//             Slow but private, free, offline-capable after first load.
//   'cloud' — uses the user's own Google AI Studio key to call the
//             Gemini API. Fast, higher quality, privacy goes through
//             Google. The key lives only in this browser's localStorage.
// ============================================

const KEYS = {
  mode: 'agentmesh.mode',
  apiKey: 'agentmesh.apiKey',
  cloudModel: 'agentmesh.cloudModel',
};

const VALID_MODES = new Set(['local', 'cloud']);

// Default cloud model. Gemma 4 was released April 2026 with two sizes
// hosted on Google AI Studio: 26B Mixture-of-Experts (fast, efficient)
// and 31B Dense (highest quality). We default to the MoE variant because
// it gives 31B-class quality at much lower latency and stays well within
// the free tier's per-minute quotas.
const DEFAULT_CLOUD_MODEL = 'gemma-4-26b-a4b-it';
// Fallback chain if the chosen model 5xx's. Both Gemma 4 variants are
// hosted on every AI Studio key (verified). gemma-3-27b-it was here
// previously but routinely 404s in v1beta.
export const CLOUD_MODEL_FALLBACKS = [
  'gemma-4-26b-a4b-it',
  'gemma-4-31b-it',
];
// Supported model IDs for cloud mode. Anything outside this list (e.g.
// stale gemma-3-27b-it values left in localStorage from an earlier
// session) gets reset to the default on next read.
const SUPPORTED_CLOUD_MODELS = new Set([
  'gemma-4-26b-a4b-it',
  'gemma-4-31b-it',
]);

const listeners = new Set();

function notify() {
  for (const cb of listeners) {
    try { cb(getSettings()); } catch (e) { console.warn('[settings] listener threw:', e); }
  }
}

export function onChange(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function getMode() {
  const m = localStorage.getItem(KEYS.mode);
  return VALID_MODES.has(m) ? m : 'local';
}

export function setMode(mode) {
  if (!VALID_MODES.has(mode)) return;
  localStorage.setItem(KEYS.mode, mode);
  notify();
}

export function getApiKey() {
  return localStorage.getItem(KEYS.apiKey) || '';
}

export function setApiKey(key) {
  const trimmed = (key || '').trim();
  if (trimmed) {
    localStorage.setItem(KEYS.apiKey, trimmed);
  } else {
    localStorage.removeItem(KEYS.apiKey);
  }
  notify();
}

export function getCloudModel() {
  const stored = localStorage.getItem(KEYS.cloudModel);
  if (stored && SUPPORTED_CLOUD_MODELS.has(stored)) return stored;
  // Stale value from an earlier session (e.g. gemma-3-27b-it). Reset it
  // so the next save round-trips a sane value instead of leaving the
  // bad one in storage where it'll keep coming back.
  if (stored) {
    try { localStorage.removeItem(KEYS.cloudModel); } catch {}
  }
  return DEFAULT_CLOUD_MODEL;
}

export function setCloudModel(modelId) {
  const trimmed = (modelId || '').trim();
  if (trimmed) {
    localStorage.setItem(KEYS.cloudModel, trimmed);
  } else {
    localStorage.removeItem(KEYS.cloudModel);
  }
  notify();
}

export function getSettings() {
  return {
    mode: getMode(),
    apiKey: getApiKey(),
    cloudModel: getCloudModel(),
  };
}

/**
 * True iff the app can run a query *right now*. Local mode requires the
 * worker to have loaded a model (caller checks separately). Cloud mode
 * requires the user to have pasted a key.
 */
export function isCloudReady() {
  return getMode() === 'cloud' && !!getApiKey();
}
