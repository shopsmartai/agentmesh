// ============================================
// AGENTMESH // Main entry point
// ============================================

import { model } from './model.js';
import { runSwarm } from './agents.js';
import * as settings from './settings.js';
import {
  showScreen,
  updateLoading,
  setAgentCount,
  resetStats,
  clearSwarm,
  renderPlannerCard,
  updatePlannerCard,
  renderWorkerCard,
  updateWorkerCard,
  renderSynthCard,
  updateSynthCard,
  showSynthesis,
} from './ui.js';
import { getViz, resetViz } from './viz.js';

const ASCII_LOGO = `
   ██████╗  ██████╗ ███████╗███╗   ██╗████████╗███╗   ███╗███████╗███████╗██╗  ██╗
   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝████╗ ████║██╔════╝██╔════╝██║  ██║
   ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██╔████╔██║█████╗  ███████╗███████║
   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║
   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║ ╚═╝ ██║███████╗███████║██║  ██║
   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
                  swarm intelligence · running locally · in your browser
`;

const BOOT_LINES = [
  { text: "BIOS v2.1 :: AgentMesh Bootloader", cls: "info", delay: 200 },
  { text: "Detecting runtime environment...", cls: "dim", delay: 400 },
  { text: "  ✓ navigator.gpu detected", cls: "ok", delay: 200, conditional: 'webgpu' },
  { text: "  ✗ WebGPU not available in this browser", cls: "err", delay: 200, conditional: 'no-webgpu' },
  { text: "  ✓ Web Workers supported", cls: "ok", delay: 150 },
  { text: "  ✓ Service Worker API present", cls: "ok", delay: 150 },
  { text: "  ✓ Cache Storage API present", cls: "ok", delay: 150 },
  { text: "Loading subsystems...", cls: "dim", delay: 300 },
  { text: "  [OK] tokenizer.runtime", cls: "ok", delay: 100 },
  { text: "  [OK] vector.cache", cls: "ok", delay: 100 },
  { text: "  [OK] tool.registry (wikipedia, hackernews, duckduckgo)", cls: "ok", delay: 100 },
  { text: "  [OK] swarm.orchestrator", cls: "ok", delay: 100 },
  { text: "", cls: "dim", delay: 200 },
  { text: "Network state: 0 outbound connections", cls: "info", delay: 200 },
  { text: "Privacy mode: enabled (all inference local)", cls: "ok", delay: 200 },
  { text: "", cls: "dim", delay: 200 },
  { text: "System ready. Awaiting operator.", cls: "info", delay: 300 },
];

const TIPS = [
  "First load downloads ~3.1 GB of Gemma 4 E2B weights. Cached after that — reload is instant.",
  "Append ?model=smollm for a 270 MB lite mode (SmolLM2-360M, faster but less capable).",
  "Your queries never leave this tab.",
  "Five agents share one Gemma 4 instance — they timeshare on the GPU.",
  "Gemma 4 E2B has a 128K context window, so agents can pass full notes between each other.",
  "Real local AI. No API key. No telemetry.",
  "WebGPU + Transformers.js v4 = no install required.",
];

// ============================================
// WEBGPU CHECK
// ============================================

async function checkWebGPU() {
  if (!('gpu' in navigator)) return { ok: false };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false };
    return { ok: true };
  } catch {
    return { ok: false };
  }
}

// ============================================
// BOOT SEQUENCE
// ============================================

async function runBootSequence() {
  document.getElementById('ascii-logo').textContent = ASCII_LOGO;
  const bootLog = document.getElementById('boot-log');
  const initBtn = document.getElementById('boot-init-btn');
  const bootActions = document.getElementById('boot-actions');
  const bootSettingsBtn = document.getElementById('boot-settings-btn');

  const gpu = await checkWebGPU();
  const hasWebgpu = gpu.ok;

  const lines = BOOT_LINES.filter(line => {
    if (!line.conditional) return true;
    if (line.conditional === 'webgpu') return hasWebgpu;
    if (line.conditional === 'no-webgpu') return !hasWebgpu;
    return true;
  });

  for (const line of lines) {
    await wait(line.delay);
    const span = document.createElement('span');
    span.className = `line ${line.cls}`;
    span.textContent = line.text || '\u00A0';
    bootLog.appendChild(span);
    bootLog.scrollTop = bootLog.scrollHeight;
  }

  await wait(400);

  // Cloud mode does not need WebGPU. If the user already has cloud + key
  // configured we let them through even without a GPU.
  if (!hasWebgpu && !settings.isCloudReady()) {
    setTimeout(() => showScreen('no-webgpu-screen'), 1500);
    return;
  }

  // Wire the boot settings button — opens the same panel used on the main
  // screen, so the user can switch to cloud mode before any 3 GB download.
  bootSettingsBtn?.addEventListener('click', openSettingsPanel);

  // Reflect the chosen mode in the init button label so the user knows
  // what they're about to trigger. Updates whenever settings change.
  const updateInitBtn = () => {
    if (settings.isCloudReady()) {
      initBtn.innerHTML = '[ START_CLOUD_MODE ] <kbd>⏎</kbd>';
    } else if (settings.getMode() === 'cloud') {
      initBtn.innerHTML = '[ ADD_API_KEY_IN_SETTINGS ]';
      initBtn.disabled = true;
      return;
    } else {
      initBtn.innerHTML = '[ INITIALIZE_SWARM.exe ]';
    }
    initBtn.disabled = false;
  };
  settings.onChange(updateInitBtn);
  updateInitBtn();

  if (bootActions) bootActions.style.display = 'flex';
  initBtn.addEventListener('click', () => {
    if (initBtn.disabled) return;
    startModelLoad();
  });
}

// ============================================
// MODEL LOAD
// ============================================

async function startModelLoad() {
  // Cloud mode short-circuit: if the user has chosen cloud + has a key,
  // we skip the 3 GB local download entirely. The runtime is ready
  // immediately and the user goes straight to the main screen.
  if (settings.isCloudReady()) {
    const info = model.setupCloudMode();
    const versionEl = document.querySelector('.header-version');
    if (versionEl && info?.label) {
      versionEl.textContent = `v1.0 · ${info.label} (cloud)`;
    }
    showMainScreen();
    return;
  }

  showScreen('loading-screen');

  // Rotate tips
  const tipText = document.getElementById('loading-tip-text');
  let tipIndex = 0;
  const tipInterval = setInterval(() => {
    tipIndex = (tipIndex + 1) % TIPS.length;
    tipText.textContent = TIPS[tipIndex];
  }, 3500);

  try {
    updateLoading({ progress: 1, status: 'Loading Transformers.js...' });
    const info = await model.load(updateLoading);

    const versionEl = document.querySelector('.header-version');
    if (versionEl && info.label) {
      versionEl.textContent = `v1.0 · ${info.label} (local)`;
    }

    clearInterval(tipInterval);
    await wait(500);
    showMainScreen();
  } catch (err) {
    clearInterval(tipInterval);
    console.error('[boot] model load failed:', err);
    document.getElementById('loading-status').textContent = `Failed: ${err.message}`;
    document.getElementById('loading-status').style.color = 'var(--neon-red)';
  }
}

// ============================================
// MAIN INTERFACE
// ============================================

let currentRun = null;

function showMainScreen() {
  showScreen('main-screen');
  setupQueryInterface();
  setupSettingsPanel();
  updateHeaderModeLabel();
}

// Update the header version label to reflect the active mode.
function updateHeaderModeLabel() {
  const versionEl = document.querySelector('.header-version');
  if (!versionEl) return;
  const mode = settings.getMode();
  if (mode === 'cloud') {
    const cloudModel = settings.getCloudModel();
    versionEl.textContent = `v1.0 · ${cloudModel} (cloud)`;
  } else {
    const label = model.modelLabel || 'gemma-4-E2B';
    versionEl.textContent = `v1.0 · ${label} (local)`;
  }
}

// ============================================
// SETTINGS PANEL
// ============================================

// Idempotent — safe to call multiple times. Each screen that mounts the
// settings panel calls this. We attach listeners only on first invocation.
let settingsWired = false;
function openSettingsPanel() {
  setupSettingsPanel();
  const panel = document.getElementById('settings-panel');
  if (panel) panel.hidden = false;
  // Force a re-render of the inputs from current settings state.
  document.dispatchEvent(new CustomEvent('agentmesh:settings-render'));
}

function setupSettingsPanel() {
  const panel = document.getElementById('settings-panel');
  if (!panel) return;
  const closeBtn = document.getElementById('settings-close');
  const backdrop = panel.querySelector('.settings-backdrop');
  const modeLocal = document.getElementById('mode-local');
  const modeCloud = document.getElementById('mode-cloud');
  const cloudGroup = document.getElementById('settings-cloud-group');
  const keyInput = document.getElementById('settings-key');
  const modelSel = document.getElementById('settings-model');
  const saveBtn = document.getElementById('settings-save');
  const clearBtn = document.getElementById('settings-clear');
  const status = document.getElementById('settings-status');
  const headerOpenBtn = document.getElementById('settings-btn');

  // Reflect current state into the panel inputs.
  const renderState = () => {
    const mode = settings.getMode();
    if (modeLocal) modeLocal.checked = mode === 'local';
    if (modeCloud) modeCloud.checked = mode === 'cloud';
    if (cloudGroup) cloudGroup.style.display = mode === 'cloud' ? '' : 'none';
    if (keyInput) keyInput.value = settings.getApiKey();
    if (modelSel) modelSel.value = settings.getCloudModel();
    if (status) {
      const ready = settings.isCloudReady();
      if (mode === 'cloud') {
        status.className = ready ? 'settings-status ok' : 'settings-status err';
        status.textContent = ready
          ? `Cloud mode ready. Using ${settings.getCloudModel()}.`
          : 'Paste your Google AI Studio API key above to enable cloud mode.';
      } else {
        status.className = 'settings-status';
        status.textContent = '';
      }
    }
  };

  if (settingsWired) {
    renderState();
    return;
  }
  settingsWired = true;

  document.addEventListener('agentmesh:settings-render', renderState);

  headerOpenBtn?.addEventListener('click', () => { panel.hidden = false; renderState(); });
  closeBtn?.addEventListener('click', () => { panel.hidden = true; });
  backdrop?.addEventListener('click', () => { panel.hidden = true; });

  modeLocal?.addEventListener('change', () => {
    if (modeLocal.checked) {
      settings.setMode('local');
      renderState();
      updateHeaderModeLabel();
    }
  });
  modeCloud?.addEventListener('change', () => {
    if (modeCloud.checked) {
      settings.setMode('cloud');
      renderState();
      updateHeaderModeLabel();
    }
  });

  modelSel?.addEventListener('change', () => {
    settings.setCloudModel(modelSel.value);
    renderState();
    updateHeaderModeLabel();
  });

  saveBtn?.addEventListener('click', () => {
    const v = (keyInput?.value || '').trim();
    if (!v) {
      if (status) {
        status.className = 'settings-status err';
        status.textContent = 'Empty key. Paste your Google AI Studio key first.';
      }
      return;
    }
    settings.setApiKey(v);
    if (settings.getMode() === 'cloud') {
      model.setupCloudMode();
      model.ready = true;
    }
    renderState();
    updateHeaderModeLabel();

    // Auto-close the panel after a successful save so the user does
    // not need a second click. If we're still sitting on the boot
    // screen and cloud is now fully configured, jump straight into
    // the main screen — same effect as clicking START_CLOUD_MODE.
    panel.hidden = true;
    const onBootScreen = document.getElementById('boot-screen')?.classList.contains('active');
    if (onBootScreen && settings.isCloudReady()) {
      showMainScreen();
    }
  });

  clearBtn?.addEventListener('click', () => {
    settings.setApiKey('');
    if (keyInput) keyInput.value = '';
    renderState();
    updateHeaderModeLabel();
  });

  renderState();
}

// Currently attached image (Blob) — only used when the loaded model is
// multimodal (Gemma 4). Cleared on swarm reset.
let attachedImage = null;
let attachedImageObjectUrl = null;

function clearAttachedImage() {
  if (attachedImageObjectUrl) {
    URL.revokeObjectURL(attachedImageObjectUrl);
    attachedImageObjectUrl = null;
  }
  attachedImage = null;
  const wrap = document.getElementById('image-preview');
  if (wrap) wrap.hidden = true;
  const input = document.getElementById('image-input');
  if (input) input.value = '';
}

function setAttachedImage(file) {
  if (!file || !file.type?.startsWith('image/')) return;
  if (attachedImageObjectUrl) URL.revokeObjectURL(attachedImageObjectUrl);
  attachedImage = file;
  attachedImageObjectUrl = URL.createObjectURL(file);
  const img = document.getElementById('image-preview-img');
  const name = document.getElementById('image-preview-name');
  const wrap = document.getElementById('image-preview');
  if (img) img.src = attachedImageObjectUrl;
  if (name) name.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  if (wrap) wrap.hidden = false;
}

function setupImageAttach() {
  // If the loaded model isn't multimodal, hide the attach UI entirely so users
  // aren't tempted to use a feature their lite-mode model can't fulfill.
  if (!model.supportsImages) {
    const wrap = document.getElementById('image-attach');
    if (wrap) wrap.style.display = 'none';
    return;
  }

  const fileInput = document.getElementById('image-input');
  const attachBtn = document.getElementById('attach-btn');
  const removeBtn = document.getElementById('image-remove-btn');
  const queryInput = document.getElementById('query-input');

  attachBtn?.addEventListener('click', () => fileInput?.click());
  fileInput?.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) setAttachedImage(file);
  });
  removeBtn?.addEventListener('click', clearAttachedImage);

  // Drag-and-drop onto the textarea — much nicer than picking a file.
  ['dragover', 'dragenter'].forEach((evt) =>
    queryInput?.addEventListener(evt, (e) => {
      e.preventDefault();
      queryInput.classList.add('drag-over');
    })
  );
  ['dragleave', 'dragend', 'drop'].forEach((evt) =>
    queryInput?.addEventListener(evt, () => queryInput.classList.remove('drag-over'))
  );
  queryInput?.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type?.startsWith('image/')) setAttachedImage(file);
  });

  // Allow paste-an-image (great for screenshots).
  queryInput?.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items || [];
    for (const item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) setAttachedImage(file);
      }
    }
  });
}

function setupQueryInterface() {
  const queryInput = document.getElementById('query-input');
  const runBtn = document.getElementById('run-btn');
  const exampleBtns = document.querySelectorAll('.example-btn');

  setupImageAttach();

  exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      queryInput.value = btn.dataset.query;
      queryInput.focus();
    });
  });

  runBtn.addEventListener('click', () => runQuery(queryInput.value));

  queryInput.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      runQuery(queryInput.value);
    }
  });

  document.getElementById('reset-btn')?.addEventListener('click', () => {
    queryInput.value = '';
    queryInput.focus();
    clearSwarm();
    resetStats();
    clearAttachedImage();
  });

  document.getElementById('copy-btn')?.addEventListener('click', () => {
    const text = document.getElementById('synthesis-body')?.innerText || '';
    navigator.clipboard?.writeText(text);
    const btn = document.getElementById('copy-btn');
    const original = btn.textContent;
    btn.textContent = '[ copied ✓ ]';
    setTimeout(() => { btn.textContent = original; }, 1500);
  });

  document.getElementById('share-btn')?.addEventListener('click', () => {
    const q = queryInput.value;
    const url = new URL(window.location.href);
    url.searchParams.set('q', q);
    navigator.clipboard?.writeText(url.toString());
    const btn = document.getElementById('share-btn');
    const original = btn.textContent;
    btn.textContent = '[ url copied ✓ ]';
    setTimeout(() => { btn.textContent = original; }, 1500);
  });

  // Pre-fill from URL ?q=
  const urlQ = new URL(window.location.href).searchParams.get('q');
  if (urlQ) {
    queryInput.value = urlQ;
  }
}

async function runQuery(query) {
  query = (query || '').trim();
  // Allow empty text query when an image is attached — the swarm can
  // research the image alone ("describe what's in this and look up
  // anything interesting").
  if (!query && !attachedImage) return;
  if (!query && attachedImage) query = 'Describe what is in this image and research the most interesting topics it raises.';
  if (currentRun) return; // simple lock

  const runBtn = document.getElementById('run-btn');
  runBtn.disabled = true;
  runBtn.textContent = '[ swarm running... ]';

  clearSwarm();
  resetStats();
  resetViz();
  document.getElementById('swarm-active').style.display = 'grid';
  document.getElementById('swarm-empty').style.display = 'none';

  // Render planner first
  renderPlannerCard(query);
  setAgentCount(1);

  // Initialize viz: register the planner node
  const viz = getViz();
  viz?.addNode('agent-planner', 'planner');

  // Capture the image for this run so a later remove doesn't yank it
  // mid-flight. The image is consumed by the planner once.
  const runImage = attachedImage;

  currentRun = (async () => {
    try {
      await runSwarm(model, query, {
        image: runImage,
        onPlanner: (u) => {
          updatePlannerCard(u);
          if (u.status === 'done' && viz && u.plan) {
            // Planner finished — visualize fan-out from planner to each worker
            // (the worker cards will be added a moment later)
            setTimeout(() => {
              for (let i = 0; i < u.plan.length; i++) {
                viz.setConnection('agent-planner', `agent-${i + 1}`, 1, 'cyan');
                viz.emitParticles('agent-planner', `agent-${i + 1}`, 6, 'cyan');
              }
            }, 50);
          }
        },
        onWorkerStart: (i, subQ, perspective) => {
          renderWorkerCard(i, subQ, perspective);
          viz?.addNode(`agent-${i + 1}`, 'worker');
          viz?.setConnection('agent-planner', `agent-${i + 1}`, 0.4, 'cyan');
        },
        onWorkerUpdate: (i, u) => updateWorkerCard(i, u),
        onWorkerDone: (i, result) => {
          updateWorkerCard(i, { status: 'done', text: result.answer, toolUsed: result.toolUsed, sources: result.sources });
          // Pulse the connection to indicate completion
          viz?.setConnection('agent-planner', `agent-${i + 1}`, 0.8, 'green');
          viz?.emitParticles(`agent-${i + 1}`, 'agent-synth', 5, 'magenta');
        },
        onSynthUpdate: (u) => {
          if (!document.getElementById('agent-synth')) {
            renderSynthCard();
            viz?.addNode('agent-synth', 'synth');
            // Connect every worker to the synth node
            for (let i = 1; i <= 5; i++) {
              if (document.getElementById(`agent-${i}`)) {
                viz?.setConnection(`agent-${i}`, 'agent-synth', 1, 'magenta');
              }
            }
          }
          updateSynthCard(u);
          if (u.status === 'done') {
            showSynthesis(u.text);
          }
        },
        onComplete: () => { /* done */ },
        onError: (err) => {
          console.error(err);
          alert(`Swarm error: ${err.message}\n\nTry refreshing or check the console.`);
        },
      });
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = '[ SPAWN_SWARM ] <kbd>⌘↵</kbd>';
      currentRun = null;
    }
  })();
}

// ============================================
// UTILITIES
// ============================================

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// BOOT
// ============================================

window.addEventListener('DOMContentLoaded', () => {
  showScreen('boot-screen');
  runBootSequence();
});
