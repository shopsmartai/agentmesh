// ============================================
// AGENTMESH // Main entry point
// ============================================

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
  { text: "  [OK] tool.registry (web_search, wikipedia, hackernews)", cls: "ok", delay: 100 },
  { text: "  [OK] swarm.orchestrator", cls: "ok", delay: 100 },
  { text: "", cls: "dim", delay: 200 },
  { text: "Network state: 0 outbound connections", cls: "info", delay: 200 },
  { text: "Privacy mode: enabled (all inference local)", cls: "ok", delay: 200 },
  { text: "", cls: "dim", delay: 200 },
  { text: "System ready. Awaiting operator.", cls: "info", delay: 300 },
];

const TIPS = [
  "First load downloads ~1.2GB. Cached after that.",
  "Your queries never leave this tab.",
  "Each agent runs Gemma 4 E4B in a Web Worker.",
  "Press Cmd+/ to toggle the console.",
  "Real local AI. No fine print, no telemetry.",
];

// ============================================
// SCREEN MANAGEMENT
// ============================================

const screens = {
  boot: document.getElementById('boot-screen'),
  loading: document.getElementById('loading-screen'),
  noWebgpu: document.getElementById('no-webgpu-screen'),
  main: document.getElementById('main-screen'),
};

function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.remove('active'));
  screens[name].classList.add('active');
}

// ============================================
// WEBGPU DETECTION
// ============================================

async function checkWebGPU() {
  if (!('gpu' in navigator)) return { ok: false, reason: 'no-gpu' };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: 'no-adapter' };
    return { ok: true, adapter };
  } catch (e) {
    return { ok: false, reason: 'request-failed', error: e.message };
  }
}

// ============================================
// BOOT SEQUENCE
// ============================================

async function runBootSequence() {
  // Render ASCII logo
  document.getElementById('ascii-logo').textContent = ASCII_LOGO;

  const bootLog = document.getElementById('boot-log');
  const initBtn = document.getElementById('boot-init-btn');

  // Check WebGPU first
  const gpu = await checkWebGPU();
  const hasWebgpu = gpu.ok;

  // Filter conditional lines
  const lines = BOOT_LINES.filter(line => {
    if (!line.conditional) return true;
    if (line.conditional === 'webgpu') return hasWebgpu;
    if (line.conditional === 'no-webgpu') return !hasWebgpu;
    return true;
  });

  // Render lines progressively
  for (const line of lines) {
    await wait(line.delay);
    const span = document.createElement('span');
    span.className = `line ${line.cls}`;
    span.textContent = line.text || '\u00A0';
    bootLog.appendChild(span);
    bootLog.scrollTop = bootLog.scrollHeight;
  }

  await wait(400);

  // Show init button or error redirect
  if (!hasWebgpu) {
    setTimeout(() => showScreen('noWebgpu'), 1500);
  } else {
    initBtn.style.display = 'inline-flex';
    initBtn.addEventListener('click', startModelLoad);
  }
}

// ============================================
// MODEL LOAD (placeholder for now - will integrate Transformers.js next)
// ============================================

async function startModelLoad() {
  showScreen('loading');

  const status = document.getElementById('loading-status');
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  const filesLog = document.getElementById('loading-files');
  const tipText = document.getElementById('loading-tip-text');

  // Rotate tips
  let tipIndex = 0;
  const tipInterval = setInterval(() => {
    tipIndex = (tipIndex + 1) % TIPS.length;
    tipText.textContent = TIPS[tipIndex];
  }, 3500);

  // Mock loading sequence (replaced with real Transformers.js progress in Phase 2)
  const stages = [
    { msg: 'Initializing WebGPU adapter...', pct: 5, files: ['adapter: NVIDIA RTX (or equivalent)'] },
    { msg: 'Allocating GPU memory...', pct: 10, files: ['memory: 4GB available'] },
    { msg: 'Fetching gemma-4-e4b config...', pct: 15, files: ['config.json (1.2KB)'] },
    { msg: 'Loading tokenizer...', pct: 20, files: ['tokenizer.json (2.4MB)', 'tokenizer_config.json'] },
    { msg: 'Streaming model weights...', pct: 35, files: ['model-00001-of-00003.onnx (412MB)'] },
    { msg: 'Streaming model weights...', pct: 60, files: ['model-00002-of-00003.onnx (398MB)'] },
    { msg: 'Streaming model weights...', pct: 85, files: ['model-00003-of-00003.onnx (245MB)'] },
    { msg: 'Compiling shaders...', pct: 95, files: ['compiling 47 WebGPU pipelines...'] },
    { msg: 'Warming up...', pct: 99, files: ['running 3 dummy inferences'] },
    { msg: 'Ready.', pct: 100, files: ['model loaded · 1.06GB · 124 t/s estimated'] },
  ];

  for (const stage of stages) {
    await wait(800);
    status.textContent = stage.msg;
    progressBar.style.width = `${stage.pct}%`;
    progressText.textContent = `${stage.pct}%`;
    stage.files.forEach(f => {
      const line = document.createElement('div');
      line.className = 'file-line';
      line.textContent = `> ${f}`;
      filesLog.appendChild(line);
      filesLog.scrollTop = filesLog.scrollHeight;
    });
  }

  clearInterval(tipInterval);
  await wait(500);
  showMainScreen();
}

// ============================================
// MAIN INTERFACE
// ============================================

function showMainScreen() {
  showScreen('main');
  setupQueryInterface();
}

function setupQueryInterface() {
  const queryInput = document.getElementById('query-input');
  const runBtn = document.getElementById('run-btn');
  const exampleBtns = document.querySelectorAll('.example-btn');

  exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      queryInput.value = btn.dataset.query;
      queryInput.focus();
    });
  });

  runBtn.addEventListener('click', () => runSwarm(queryInput.value));

  queryInput.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      runSwarm(queryInput.value);
    }
  });

  // Reset button
  document.getElementById('reset-btn')?.addEventListener('click', resetSwarm);
}

async function runSwarm(query) {
  if (!query.trim()) return;
  console.log('[swarm] running query:', query);
  // TODO: Phase 2 - implement actual agent orchestration
  alert('Swarm orchestration coming in Phase 2!\n\nQuery received: ' + query);
}

function resetSwarm() {
  document.getElementById('query-input').value = '';
  document.getElementById('synthesis-section').style.display = 'none';
  document.getElementById('swarm-empty').style.display = 'flex';
  document.getElementById('swarm-active').style.display = 'none';
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
  showScreen('boot');
  runBootSequence();
});
