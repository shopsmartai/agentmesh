// ============================================
// AGENTMESH // Network visualization layer
// ============================================
// Draws curved connections between agent cards and animates flowing particles
// along them. Pure Canvas 2D — sits on a transparent overlay above the grid.
// ============================================

const COLORS = {
  cyan: { r: 0, g: 240, b: 255 },
  magenta: { r: 255, g: 0, b: 212 },
  green: { r: 0, g: 255, b: 136 },
  amber: { r: 255, g: 170, b: 0 },
  dim: { r: 100, g: 120, b: 140 },
};

const c2css = ({ r, g, b }, a = 1) => `rgba(${r}, ${g}, ${b}, ${a})`;

class NetworkViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;

    /** @type {Array<{cardId: string, role: string}>} */
    this.nodes = [];
    /** @type {Array<{from: string, to: string, intensity: number, color: keyof typeof COLORS}>} */
    this.connections = [];
    /** @type {Array<{x: number, y: number, vx: number, vy: number, life: number, maxLife: number, color: keyof typeof COLORS, size: number, fromX: number, fromY: number, toX: number, toY: number, t: number, speed: number}>} */
    this.particles = [];

    this.running = false;
    this.lastTime = 0;

    this.resize = this.resize.bind(this);
    this.tick = this.tick.bind(this);

    window.addEventListener('resize', this.resize);
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const targetW = Math.round(rect.width * this.dpr);
    const targetH = Math.round(rect.height * this.dpr);
    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.lastTime = performance.now();
    requestAnimationFrame(this.tick);
  }

  stop() {
    this.running = false;
  }

  reset() {
    this.nodes = [];
    this.connections = [];
    this.particles = [];
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Register an agent card by ID and role. We resolve its on-screen position
   * lazily each frame from the DOM so reflows are tolerated.
   */
  addNode(cardId, role) {
    if (this.nodes.find(n => n.cardId === cardId)) return;
    this.nodes.push({ cardId, role });
  }

  /**
   * Add or update a connection from one node to another.
   * intensity: 0 = idle, 1 = active.
   */
  setConnection(fromId, toId, intensity = 1, color = 'cyan') {
    let conn = this.connections.find(c => c.from === fromId && c.to === toId);
    if (!conn) {
      conn = { from: fromId, to: toId, intensity: 0, color };
      this.connections.push(conn);
    }
    conn.intensity = intensity;
    conn.color = color;
  }

  /**
   * Emit N particles flowing from one node to another along the curve.
   */
  emitParticles(fromId, toId, count = 8, color = 'cyan') {
    const from = this.getNodePos(fromId);
    const to = this.getNodePos(toId);
    if (!from || !to) return;

    for (let i = 0; i < count; i++) {
      this.particles.push({
        fromX: from.x,
        fromY: from.y,
        toX: to.x,
        toY: to.y,
        x: from.x,
        y: from.y,
        t: 0,
        speed: 0.4 + Math.random() * 0.5, // progress per second
        size: 1.5 + Math.random() * 2,
        color,
        life: 1,
        maxLife: 1,
        // Stagger emission so they don't all start at the same instant
        delay: i * 0.08,
      });
    }
  }

  /**
   * Find live screen position of an agent card's center.
   */
  getNodePos(cardId) {
    const el = document.getElementById(cardId);
    if (!el) return null;
    const elRect = el.getBoundingClientRect();
    const canvasRect = this.canvas.getBoundingClientRect();
    return {
      x: elRect.left - canvasRect.left + elRect.width / 2,
      y: elRect.top - canvasRect.top + 18, // near top edge of card (where the glow strip is)
    };
  }

  /**
   * Bezier curve between two points, slight downward bow.
   */
  bezierAt(t, p0, p1, cp) {
    const u = 1 - t;
    return {
      x: u * u * p0.x + 2 * u * t * cp.x + t * t * p1.x,
      y: u * u * p0.y + 2 * u * t * cp.y + t * t * p1.y,
    };
  }

  controlPoint(p0, p1) {
    // Sag the curve slightly so it doesn't draw straight through cards
    const mx = (p0.x + p1.x) / 2;
    const my = (p0.y + p1.y) / 2;
    const dx = p1.x - p0.x;
    const dy = p1.y - p0.y;
    const len = Math.hypot(dx, dy);
    const sag = Math.min(80, len * 0.18);
    // Perpendicular offset (always downward in our layout)
    return { x: mx, y: my + sag };
  }

  drawConnection(p0, p1, intensity, color) {
    const cp = this.controlPoint(p0, p1);
    const ctx = this.ctx;
    const c = COLORS[color] || COLORS.cyan;

    // Base line — always visible so the topology is readable even at idle
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.quadraticCurveTo(cp.x, cp.y, p1.x, p1.y);
    ctx.strokeStyle = c2css(c, 0.35);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    if (intensity > 0) {
      // Active overlay — saturated and glowing
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.quadraticCurveTo(cp.x, cp.y, p1.x, p1.y);
      ctx.strokeStyle = c2css(c, 0.7 * intensity);
      ctx.lineWidth = 2.5;
      ctx.shadowColor = c2css(c, 1);
      ctx.shadowBlur = 16 * intensity;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }

  drawNode(p, role) {
    const ctx = this.ctx;
    const isPlanner = role === 'planner';
    const isSynth = role === 'synth';
    const color = isSynth ? COLORS.magenta : isPlanner ? COLORS.cyan : COLORS.cyan;

    // Outer glow ring
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = c2css(color, 0.15);
    ctx.fill();

    // Core dot
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = c2css(color, 1);
    ctx.shadowColor = c2css(color, 1);
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  drawParticle(p) {
    const ctx = this.ctx;
    const c = COLORS[p.color] || COLORS.cyan;

    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fillStyle = c2css(c, 1);
    ctx.shadowColor = c2css(c, 1);
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Small trailing tail
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size * 1.6, 0, Math.PI * 2);
    ctx.fillStyle = c2css(c, 0.2);
    ctx.fill();
  }

  tick(now) {
    if (!this.running) return;
    const dt = Math.min(0.05, (now - this.lastTime) / 1000);
    this.lastTime = now;

    // Re-check canvas size every frame — the swarm-section grows as cards are
    // added, so we can't rely on a one-time resize.
    this.resize();

    const ctx = this.ctx;
    const w = this.canvas.width / this.dpr;
    const h = this.canvas.height / this.dpr;
    ctx.clearRect(0, 0, w, h);

    // Draw connections
    for (const conn of this.connections) {
      const from = this.getNodePos(conn.from);
      const to = this.getNodePos(conn.to);
      if (!from || !to) continue;
      this.drawConnection(from, to, conn.intensity, conn.color);
    }

    // Draw nodes
    for (const node of this.nodes) {
      const pos = this.getNodePos(node.cardId);
      if (!pos) continue;
      this.drawNode(pos, node.role);
    }

    // Update + draw particles
    const remaining = [];
    for (const p of this.particles) {
      // Honor delay
      if (p.delay > 0) {
        p.delay -= dt;
        remaining.push(p);
        continue;
      }
      p.t += p.speed * dt;
      if (p.t >= 1) {
        // Done - skip drawing
        continue;
      }
      // Recompute current point on the bezier (positions may have shifted)
      const p0 = { x: p.fromX, y: p.fromY };
      const p1 = { x: p.toX, y: p.toY };
      const cp = this.controlPoint(p0, p1);
      const pt = this.bezierAt(p.t, p0, p1, cp);
      p.x = pt.x;
      p.y = pt.y;
      this.drawParticle(p);
      remaining.push(p);
    }
    this.particles = remaining;

    // Decay connections that are still drawing but not actively used
    for (const conn of this.connections) {
      if (conn.intensity > 0.2) conn.intensity = Math.max(0.2, conn.intensity - 0.5 * dt);
    }

    requestAnimationFrame(this.tick);
  }
}

// Singleton
let _viz = null;
export function getViz() {
  if (_viz) return _viz;
  const canvas = document.getElementById('viz-canvas');
  if (!canvas) return null;
  _viz = new NetworkViz(canvas);
  _viz.start();
  return _viz;
}

export function resetViz() {
  if (_viz) _viz.reset();
}
