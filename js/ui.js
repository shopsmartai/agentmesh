// ============================================
// AGENTMESH // UI rendering helpers
// ============================================

const $ = (sel) => document.querySelector(sel);

// ============================================
// SCREEN MANAGEMENT
// ============================================

const screens = ['boot-screen', 'loading-screen', 'no-webgpu-screen', 'main-screen'];

export function showScreen(name) {
  screens.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('active', id === name);
  });
}

// ============================================
// LOADING SCREEN
// ============================================

export function updateLoading({ progress, status, file }) {
  const bar = $('#progress-bar');
  const text = $('#progress-text');
  const statusEl = $('#loading-status');
  const filesEl = $('#loading-files');

  if (typeof progress === 'number') {
    bar.style.width = `${progress}%`;
    text.textContent = `${Math.round(progress)}%`;
  }
  if (status) statusEl.textContent = status;
  if (file) {
    const line = document.createElement('div');
    line.className = 'file-line';
    line.textContent = `> ${file}`;
    filesEl.appendChild(line);
    filesEl.scrollTop = filesEl.scrollHeight;
  }
}

// ============================================
// HEADER STATS
// ============================================

let agentCount = 0;
let tokenCount = 0;

export function setAgentCount(n) {
  agentCount = n;
  $('#agent-count-stat').textContent = `${n} agent${n === 1 ? '' : 's'}`;
}

export function incrementTokens(n) {
  tokenCount += n;
  $('#token-count-stat').textContent = `${formatNumber(tokenCount)} tokens`;
}

export function resetStats() {
  agentCount = 0;
  tokenCount = 0;
  setAgentCount(0);
  $('#token-count-stat').textContent = `0 tokens`;
}

function formatNumber(n) {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return (n / 1000).toFixed(1) + 'k';
  return (n / 1_000_000).toFixed(1) + 'M';
}

// ============================================
// AGENT CARDS
// ============================================

export function clearSwarm() {
  $('#swarm-active').innerHTML = '';
  $('#swarm-active').style.display = 'none';
  $('#swarm-empty').style.display = 'flex';
  $('#synthesis-section').style.display = 'none';
  $('#synthesis-body').innerHTML = '';
  synthPanelRevealed = false;
}

export function showSwarmGrid() {
  $('#swarm-empty').style.display = 'none';
  $('#swarm-active').style.display = 'grid';
}

export function renderPlannerCard(query) {
  showSwarmGrid();
  const grid = $('#swarm-active');
  const card = document.createElement('div');
  card.className = 'agent-card thinking';
  card.id = 'agent-planner';
  card.innerHTML = `
    <div class="agent-header">
      <span class="agent-id">PLANNER · agent[0]</span>
      <span class="agent-status thinking">decomposing</span>
    </div>
    <div class="agent-task">${escapeHtml(query)}</div>
    <div class="agent-output" id="agent-planner-output"><span class="cursor-blink"></span></div>
  `;
  grid.appendChild(card);
  return card;
}

export function updatePlannerCard({ status, text, plan }) {
  const output = $('#agent-planner-output');
  const card = $('#agent-planner');
  if (!output || !card) return;

  if (status === 'thinking') {
    output.innerHTML = `${escapeHtml(text)}<span class="cursor-blink"></span>`;
  } else if (status === 'done') {
    card.classList.remove('thinking');
    card.classList.add('done');
    card.querySelector('.agent-status').className = 'agent-status done';
    card.querySelector('.agent-status').textContent = 'done';
    if (plan) {
      output.innerHTML = plan.map((q, i) =>
        `<div style="margin-bottom: 6px;"><span style="color: var(--neon-magenta);">→ agent[${i + 1}]:</span> ${escapeHtml(q)}</div>`
      ).join('');
    }
  }
}

export function renderWorkerCard(index, subQuestion, perspective) {
  const grid = $('#swarm-active');
  const card = document.createElement('div');
  // Tag the card with the role so we can style each perspective distinctly.
  const role = perspective?.role || 'worker';
  const label = perspective?.label || `WORKER · agent[${index + 1}]`;
  card.className = `agent-card role-${role}`;
  card.id = `agent-${index + 1}`;
  card.innerHTML = `
    <div class="agent-header">
      <span class="agent-id">${escapeHtml(label)} · agent[${index + 1}]</span>
      <span class="agent-status">queued</span>
    </div>
    <div class="agent-task">${escapeHtml(subQuestion)}</div>
    <div class="agent-output" id="agent-${index + 1}-output">
      <span style="color: var(--text-dim);">// awaiting GPU...</span>
    </div>
  `;
  grid.appendChild(card);
  setAgentCount(index + 2); // +1 planner +1 (1-indexed)
}

export function updateWorkerCard(index, { status, text, toolUsed, sources }) {
  const card = $(`#agent-${index + 1}`);
  const output = $(`#agent-${index + 1}-output`);
  if (!card || !output) return;

  const statusEl = card.querySelector('.agent-status');

  if (status === 'tool') {
    card.classList.add('thinking');
    statusEl.className = 'agent-status thinking';
    statusEl.textContent = 'searching';
    output.innerHTML = `<span style="color: var(--neon-amber);">$ ${escapeHtml(text)}</span>`;
  } else if (status === 'thinking') {
    statusEl.textContent = 'reasoning';
    output.innerHTML = `${escapeHtml(text)}<span class="cursor-blink"></span>`;
  } else if (status === 'done') {
    card.classList.remove('thinking');
    card.classList.add('done');
    statusEl.className = 'agent-status done';
    statusEl.textContent = `done · ${toolUsed || ''}`;
    let html = escapeHtml(text);
    if (sources?.length) {
      const links = sources.slice(0, 3).map(s =>
        s.url
          ? `<a href="${s.url}" target="_blank" style="color: var(--neon-cyan); text-decoration: none; font-size: 11px;">[${escapeHtml((s.title || s.url).slice(0, 40))}]</a>`
          : ''
      ).filter(Boolean).join(' · ');
      if (links) html += `<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-dim); font-size: 11px;">sources: ${links}</div>`;
    }
    output.innerHTML = html;
  }
}

// The synthesizer no longer renders as a card in the swarm grid. The
// dedicated synthesis-section below the grid IS the synth surface — one
// bordered, scrollable box with a status badge in its header. This avoids
// the awkward "preview here, full body down there" split that confused
// the layout. We keep the function name + signature so main.js doesn't
// need to know.
//
// Note: we still register `agent-synth` as a virtual node in the viz
// graph from main.js (for the worker→synth particle effects), even
// though there's no DOM card with that ID. The viz layer is fine with
// that — node positions are computed independently of DOM elements.
export function renderSynthCard() {
  const section = $('#synthesis-section');
  const status = $('#synthesis-status');
  const body = $('#synthesis-body');
  if (!section) return;

  // Reveal the section if it's hidden, but DO NOT scroll — the user is
  // currently watching the worker cards finish. The first streaming
  // token will trigger the auto-scroll below.
  section.style.display = 'block';
  if (status) {
    status.textContent = 'synthesizing';
    status.className = 'synthesis-status thinking';
  }
  if (body) body.innerHTML = '<span class="cursor-blink"></span>';
  synthPanelRevealed = false;
}

// Tracks whether we've already auto-scrolled to the synth panel for the
// current run. We auto-scroll once on the first streaming token, not on
// every token (which would jitter as the body grows).
let synthPanelRevealed = false;

export function updateSynthCard({ status, text }) {
  if (status === 'thinking') {
    updateSynthesisStreaming(text);
  } else if (status === 'done') {
    const statusEl = $('#synthesis-status');
    if (statusEl) {
      statusEl.textContent = 'synthesized';
      statusEl.className = 'synthesis-status done';
    }
  }
}

// ============================================
// SYNTHESIS PANEL (final markdown output)
// ============================================

/**
 * Render the synthesis panel during streaming. Reveal the panel on the
 * first token (auto-scroll once), then keep updating its content as tokens
 * arrive. Uses lightweight markdown so headings + bullets render as the
 * user reads — no waiting for the full answer to appear.
 */
function updateSynthesisStreaming(text) {
  const section = $('#synthesis-section');
  const body = $('#synthesis-body');
  if (!section || !body) return;

  if (!synthPanelRevealed) {
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    synthPanelRevealed = true;
  }

  // Re-render the whole panel each tick. The text is small enough
  // (~1-2 KB max) that this is not a perf concern, and it keeps markdown
  // structure consistent as headings/bullets complete mid-stream.
  body.innerHTML = renderMarkdown(text) + '<span class="cursor-blink"></span>';
}

export function showSynthesis(markdownText) {
  const section = $('#synthesis-section');
  const body = $('#synthesis-body');
  body.innerHTML = renderMarkdown(markdownText);
  section.style.display = 'block';
  if (!synthPanelRevealed) {
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  synthPanelRevealed = false; // reset for next run
}

// ============================================
// MARKDOWN (lightweight, safe-ish renderer)
// ============================================

function renderMarkdown(md) {
  // Escape first, then carefully introduce safe tags.
  let html = escapeHtml(md);

  // Headings
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold / italic
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bullet lists
  html = html.replace(/^([\-\*])\s+(.+)$/gm, '<li>$2</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);

  // Numbered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

  // Paragraphs (double newline → paragraph break)
  html = html.replace(/\n\n+/g, '</p><p>');
  html = `<p>${html}</p>`;

  // Cleanup empty paragraphs around block elements
  html = html.replace(/<p>(\s*<(h[123]|ul|ol|li))/g, '$1');
  html = html.replace(/(<\/(h[123]|ul|ol|li)>\s*)<\/p>/g, '$1');

  return html;
}

// ============================================
// UTILITIES
// ============================================

function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
