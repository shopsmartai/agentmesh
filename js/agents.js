// ============================================
// AGENTMESH // Agent orchestration
// ============================================
// 1. Planner: decomposes user query into 3-5 sub-questions
// 2. Workers: each takes one sub-question, runs tools, summarizes
// 3. Synthesizer: combines worker outputs into a final answer
// ============================================

import { runTool, formatResultsForModel, TOOLS } from './tools.js';

// ============================================
// PROMPTS
// ============================================

const PLANNER_SYSTEM = `You are the Planner agent in a swarm. Your job is to break a research question into 3-5 specific sub-questions that other agents will research in parallel.

Rules:
- Each sub-question should be focused, atomic, and answerable independently.
- Avoid sub-questions that just rephrase the original.
- Pick angles that together produce a complete answer.
- Output ONLY a JSON array of strings, nothing else.

Example:
Question: "What are the trade-offs between WebGPU and WebGL for ML inference?"
Output: ["What is WebGPU and how does it differ from WebGL technically?", "What ML frameworks support WebGPU in 2026?", "What are real-world performance benchmarks of WebGPU vs WebGL for inference?", "What are the browser compatibility limitations of WebGPU today?"]`;

const WORKER_SYSTEM = `You are a Worker agent in a research swarm. You have been assigned ONE specific sub-question.

You have access to one tool result (research notes from Wikipedia or Hacker News). Read it carefully and write a concise, factual answer to your sub-question in 3-5 sentences.

Rules:
- Stay focused on YOUR sub-question only.
- Cite specific facts from the research notes.
- If the notes don't answer your question, say so honestly.
- Plain prose, no bullet points.
- 100 words maximum.`;

const SYNTHESIZER_SYSTEM = `You are the Synthesizer agent. The other agents in the swarm have each researched a sub-question. Combine their findings into a final answer to the user's original question.

Rules:
- Write a clear, well-organized response in markdown.
- Use the agents' findings as your evidence base.
- If findings conflict, acknowledge it.
- Use ## headings for major sections, bullet points where useful.
- End with a brief "Bottom line:" sentence.
- Do not invent facts not present in the agents' findings.`;

// ============================================
// HELPERS
// ============================================

function safeParseJSON(text) {
  // Models often wrap JSON in markdown fences; strip them.
  const cleaned = text
    .replace(/```json\s*/gi, '')
    .replace(/```\s*$/g, '')
    .replace(/^[^\[\{]*/, '') // anything before first [ or {
    .trim();

  // Find the first valid JSON array
  const match = cleaned.match(/\[[\s\S]*\]/);
  if (!match) return null;
  try {
    return JSON.parse(match[0]);
  } catch {
    return null;
  }
}

function pickToolForQuestion(question) {
  // Lightweight heuristic: HN for tech-news-y questions, Wikipedia otherwise.
  const lc = question.toLowerCase();
  const hnSignals = ['recent', 'launch', 'release', 'startup', 'developer', 'opinion', 'discussion', 'reception', 'ship', '2026', '2025', 'news'];
  if (hnSignals.some(s => lc.includes(s))) return 'hackernews';
  return 'wikipedia';
}

// ============================================
// PLANNER
// ============================================

export async function planQuery(model, query, { onUpdate } = {}) {
  onUpdate?.({ status: 'thinking', text: '' });

  const messages = [
    { role: 'system', content: PLANNER_SYSTEM },
    { role: 'user', content: `Question: "${query}"\n\nOutput the JSON array of sub-questions now.` },
  ];

  let streamed = '';
  const raw = await model.chat(messages, {
    maxTokens: 400,
    temperature: 0.4,
    onToken: (t) => {
      streamed += t;
      onUpdate?.({ status: 'thinking', text: streamed });
    },
  });

  const text = raw || streamed;
  let plan = safeParseJSON(text);

  // Fallback if model fails to produce valid JSON
  if (!Array.isArray(plan) || plan.length === 0) {
    plan = [
      `What is the core concept behind: ${query}?`,
      `What are key facts and context for: ${query}?`,
      `What are common opinions or trade-offs about: ${query}?`,
    ];
  }

  // Cap to 4 sub-questions to keep latency reasonable
  plan = plan.slice(0, 4).map(s => String(s).trim()).filter(Boolean);

  onUpdate?.({ status: 'done', text: `Decomposed into ${plan.length} sub-questions`, plan });
  return plan;
}

// ============================================
// WORKER
// ============================================

export async function runWorker(model, subQuestion, agentId, { onUpdate } = {}) {
  // Step 1: pick a tool and run it
  const toolName = pickToolForQuestion(subQuestion);
  onUpdate?.({ status: 'tool', text: `${toolName}.search("${subQuestion}")` });

  const toolResult = await runTool(toolName, subQuestion);
  const notes = formatResultsForModel(toolResult);

  // Step 2: ask model to synthesize a focused answer
  onUpdate?.({ status: 'thinking', text: '' });

  const messages = [
    { role: 'system', content: WORKER_SYSTEM },
    {
      role: 'user',
      content: `Sub-question: ${subQuestion}\n\nResearch notes:\n${notes}\n\nWrite the answer now.`,
    },
  ];

  let streamed = '';
  const raw = await model.chat(messages, {
    maxTokens: 256,
    temperature: 0.6,
    onToken: (t) => {
      streamed += t;
      onUpdate?.({ status: 'thinking', text: streamed });
    },
  });

  const answer = (raw || streamed).trim();
  onUpdate?.({ status: 'done', text: answer, toolUsed: toolName, sources: toolResult.results || [] });

  return { agentId, subQuestion, answer, toolUsed: toolName, sources: toolResult.results || [] };
}

// ============================================
// SYNTHESIZER
// ============================================

export async function synthesize(model, query, workerResults, { onUpdate } = {}) {
  onUpdate?.({ status: 'thinking', text: '' });

  const findingsText = workerResults
    .map((w, i) => `### Agent ${i + 1} (${w.toolUsed}) on "${w.subQuestion}":\n${w.answer}`)
    .join('\n\n');

  const messages = [
    { role: 'system', content: SYNTHESIZER_SYSTEM },
    {
      role: 'user',
      content: `Original question: ${query}\n\nAgent findings:\n${findingsText}\n\nWrite the final synthesized answer in markdown now.`,
    },
  ];

  let streamed = '';
  const raw = await model.chat(messages, {
    maxTokens: 700,
    temperature: 0.5,
    onToken: (t) => {
      streamed += t;
      onUpdate?.({ status: 'thinking', text: streamed });
    },
  });

  const finalAnswer = (raw || streamed).trim();
  onUpdate?.({ status: 'done', text: finalAnswer });

  return finalAnswer;
}

// ============================================
// FULL ORCHESTRATION
// ============================================

/**
 * Run the full swarm: planner → workers (parallel) → synthesizer.
 * Each phase emits events so the UI can render live progress.
 */
export async function runSwarm(model, query, callbacks = {}) {
  const {
    onPlanner = () => {},
    onWorkerStart = () => {},
    onWorkerUpdate = () => {},
    onWorkerDone = () => {},
    onSynthUpdate = () => {},
    onComplete = () => {},
    onError = () => {},
  } = callbacks;

  try {
    // Phase 1: planning
    const plan = await planQuery(model, query, { onUpdate: onPlanner });

    // Phase 2: workers in parallel
    plan.forEach((subQ, i) => onWorkerStart(i, subQ));

    const workerPromises = plan.map((subQ, i) =>
      runWorker(model, subQ, i, {
        onUpdate: (u) => onWorkerUpdate(i, u),
      }).then((result) => {
        onWorkerDone(i, result);
        return result;
      })
    );

    // NOTE: Despite calling them "parallel" the underlying model is single-stream,
    // so workers actually serialize on the GPU. That's expected for one E4B
    // model on one GPU. The architecture supports true parallelism if/when we
    // host multiple model instances or use multi-tenant inference.
    const workerResults = await sequentialAwait(workerPromises);

    // Phase 3: synthesis
    const finalAnswer = await synthesize(model, query, workerResults, {
      onUpdate: onSynthUpdate,
    });

    onComplete({ plan, workerResults, finalAnswer });
    return { plan, workerResults, finalAnswer };
  } catch (err) {
    console.error('[swarm] error:', err);
    onError(err);
    throw err;
  }
}

// Helper: await an array of promises sequentially (so the UI shows progress
// rather than all workers racing). For true parallel rendering we'd need
// independent model instances.
async function sequentialAwait(promises) {
  const results = [];
  for (const p of promises) {
    results.push(await p);
  }
  return results;
}
