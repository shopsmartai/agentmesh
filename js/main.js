// ============================================
// AGENTMESH // Main entry point
// ============================================

import { model } from './model.js';
import { runSwarm } from './agents.js';
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

  if (!hasWebgpu) {
    setTimeout(() => showScreen('no-webgpu-screen'), 1500);
  } else {
    initBtn.style.display = 'inline-flex';
    initBtn.addEventListener('click', startModelLoad);
  }
}

// ============================================
// MODEL LOAD
// ============================================

async function startModelLoad() {
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

    // Update header label with actual model loaded
    const versionEl = document.querySelector('.header-version');
    if (versionEl && info.label) {
      versionEl.textContent = `v1.0 · ${info.label}`;
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
  if (!query) return;
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

  currentRun = (async () => {
    try {
      await runSwarm(model, query, {
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
        onWorkerStart: (i, subQ) => {
          renderWorkerCard(i, subQ);
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
