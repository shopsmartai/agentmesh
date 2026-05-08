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

const PLANNER_SYSTEM = `You produce exactly 3 short search phrases about the user's topic.

Each phrase is 4 to 8 words. No question marks. No filler words like "fundamental", "core", "current", "various". Just the topic plus one angle.

Use these 3 angles in order:
1. <topic> definition
2. <topic> examples
3. <topic> limitations

Output exactly 3 lines, numbered "1.", "2.", "3.". The topic word from the user's question MUST appear in each line. No preamble. No JSON. No commentary.`;

const WORKER_SYSTEM = `You answer ONE sub-question using ONLY the research notes provided.

Rules:
- Read the notes carefully. Quote or paraphrase specific facts from them.
- If the notes do not contain enough information, say so directly. Do not invent facts.
- Write 2-4 short sentences in plain prose. No bullets, no headings.
- Stay strictly on the sub-question — do not drift to related topics.
- Maximum 80 words.`;

const SYNTHESIZER_SYSTEM = `You combine findings from several research agents into one clear answer.

Rules:
- Write a complete, well-structured response to the original question in markdown.
- Use TWO or THREE \`##\` section headings that map to the major themes in the findings.
- Inside each section, use short paragraphs or bullet points (whichever reads better).
- Only use facts present in the agents' findings — never invent.
- If findings disagree or are thin on a topic, acknowledge it briefly.
- End with a single line starting with "**Bottom line:**" that gives a one-sentence takeaway.
- No preamble like "Here is the answer". Start directly with content.`;

// ============================================
// HELPERS
// ============================================

/**
 * Extract sub-questions from planner output. Handles multiple formats:
 *   1. "1. foo\n2. bar\n3. baz"  (preferred numbered format)
 *   2. "- foo\n- bar"              (bullets)
 *   3. JSON array                   (legacy fallback)
 * Returns array of cleaned question strings, or empty if nothing found.
 */
function extractSubQuestions(text) {
  if (!text) return [];

  // Try JSON first (legacy compatibility — array of strings or objects)
  const jsonMatch = text.match(/\[[\s\S]*\]/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const items = parsed
          .map((item) => {
            if (typeof item === 'string') return item.trim();
            // Common object shapes: {title}, {question}, {text}, {q}
            if (item && typeof item === 'object') {
              return String(item.title || item.question || item.text || item.q || '').trim();
            }
            return '';
          })
          .filter((q) => isValidSubQuestion(q));
        if (items.length > 0) return items;
      }
    } catch { /* fallthrough */ }
  }

  // Try to extract any "title": "..." or "question": "..." patterns from
  // malformed JSON-ish output (common with small models)
  const titleMatches = [...text.matchAll(/["'](?:title|question|q|text)["']\s*:\s*["']([^"']{8,200})["']/gi)];
  if (titleMatches.length > 0) {
    const items = titleMatches
      .map((m) => m[1].trim())
      .filter((q) => isValidSubQuestion(q));
    if (items.length > 0) return items;
  }

  // Numbered or bulleted list
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  const items = [];
  for (const line of lines) {
    // Match "1." "1)" "-" "*" "•" prefixes
    const m = line.match(/^(?:\d+[\.\)]|[\-\*\u2022])\s+(.+)$/);
    if (m && m[1]) {
      let q = m[1].trim();
      // Strip wrapping quotes/backticks
      q = q.replace(/^["'`]+|["'`]+$/g, '').trim();
      if (isValidSubQuestion(q)) items.push(q);
    }
  }
  return items;
}

/**
 * Reject obvious garbage output from small models.
 */
function isValidSubQuestion(q) {
  if (!q || typeof q !== 'string') return false;
  if (q.length < 8 || q.length > 240) return false;
  // Common garbage patterns
  if (/\[object\s+Object\]/i.test(q)) return false;
  if (/^(yes|no|ok|true|false|null|undefined)\.?$/i.test(q)) return false;
  if (/^\.\.\.$/.test(q)) return false;
  // Must contain at least 3 alpha-rich words
  const words = q.split(/\s+/).filter(w => /[a-z]{2,}/i.test(w));
  if (words.length < 3) return false;
  return true;
}

/**
 * Returns true if the planned sub-questions share substantive vocabulary with
 * the original query (signals that the model didn't just copy an example).
 */
function planSharesTopicWith(plan, query) {
  const stopwords = new Set(['the','a','an','of','to','for','and','or','is','are','what','how','why','do','does','in','on','at','about','between','with','from']);
  const tokens = (s) => s.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 3 && !stopwords.has(w));

  const queryWords = new Set(tokens(query));
  if (queryWords.size === 0) return true; // can't check, assume ok

  for (const q of plan) {
    const qWords = tokens(q);
    if (qWords.some(w => queryWords.has(w))) return true;
  }
  return false;
}

function pickToolForQuestion(question) {
  // 3-way heuristic across wikipedia / hackernews / arxiv. The cascading
  // fallback below catches misclassifications by retrying with a different
  // source when the picked tool returns nothing.
  const lc = question.toLowerCase();
  const arxivSignals = ['paper', 'preprint', 'study', 'research', 'arxiv', 'algorithm', 'mechanism', 'theorem', 'proof', 'neural network', 'gradient', 'optimization', 'embedding', 'transformer', 'training'];
  if (arxivSignals.some((s) => lc.includes(s))) return 'arxiv';

  const hnSignals = ['recent', 'launch', 'release', 'startup', 'developer', 'opinion', 'discussion', 'reception', 'ship', '2026', '2025', '2024', 'news', 'open source', 'open-source'];
  if (hnSignals.some((s) => lc.includes(s))) return 'hackernews';

  return 'wikipedia';
}

// Cascading fallback order. If the first tool returns 0 results, try the
// next ones in turn. Order chosen so generic-knowledge sources back up
// specialty sources and vice versa.
const TOOL_FALLBACKS = {
  wikipedia: ['duckduckgo', 'arxiv'],
  hackernews: ['duckduckgo', 'wikipedia'],
  arxiv: ['wikipedia', 'duckduckgo'],
  duckduckgo: ['wikipedia', 'arxiv'],
};

async function runToolWithFallback(primary, query, { onUpdate } = {}) {
  const order = [primary, ...(TOOL_FALLBACKS[primary] || [])];
  let last = null;
  for (const tool of order) {
    onUpdate?.({ status: 'tool', text: `${tool}.search("${query}")` });
    last = await runTool(tool, query);
    if (last.ok && Array.isArray(last.results) && last.results.length > 0) {
      return { toolUsed: tool, toolResult: last };
    }
  }
  // All sources empty — return whatever we last got (so the worker can be
  // honest about the lack of findings instead of hallucinating).
  return { toolUsed: primary, toolResult: last || { ok: false, tool: primary, error: 'no result', results: [] } };
}

// ============================================
// PLANNER
// ============================================

export async function planQuery(model, query, { onUpdate, image } = {}) {
  onUpdate?.({ status: 'thinking', text: '' });

  // When an image is attached, ask the planner to anchor on the image
  // content as well as any text query. Image appears as a content block so
  // multimodal Gemma 4 sees it directly.
  const userText = image
    ? `Image attached above.\nTopic from user prompt: "${query}"\n\nOutput exactly 3 short search phrases about THIS image now.`
    : `User's topic: "${query}"\n\nOutput exactly 3 short search phrases now.`;

  const messages = [
    { role: 'system', content: PLANNER_SYSTEM },
    { role: 'user', content: userText },
  ];

  let streamed = '';
  // Greedy decoding (temperature 0) and a tight token cap. Planner outputs
  // are short and structured; sampling adds latency without quality.
  const raw = await model.chat(messages, {
    maxTokens: 150,
    temperature: 0,
    image,
    onToken: (t) => {
      streamed += t;
      onUpdate?.({ status: 'thinking', text: streamed });
    },
  });

  const text = raw || streamed;
  let plan = extractSubQuestions(text);

  // Sanity check: at least one sub-question must share a keyword with the
  // original query, otherwise the small model copied the example prompt.
  if (plan.length > 0 && !planSharesTopicWith(plan, query)) {
    plan = [];
  }

  // Fallback if model fails to produce a parseable list. We pick three
  // distinct lenses on the topic to ensure the workers don't all duplicate.
  if (plan.length === 0) {
    plan = [
      `Background and definition: ${query}`,
      `Real-world examples or use cases related to: ${query}`,
      `Trade-offs, limitations, or criticisms of: ${query}`,
    ];
  }

  // Cap to 3 sub-questions for predictable latency on small models
  plan = plan.slice(0, 3);

  onUpdate?.({ status: 'done', text: `Decomposed into ${plan.length} sub-questions`, plan });
  return plan;
}

// ============================================
// WORKER
// ============================================

export async function runWorker(model, subQuestion, agentId, { onUpdate } = {}) {
  // Step 1: pick a primary tool, run it, cascade through fallbacks if empty
  const primary = pickToolForQuestion(subQuestion);
  const { toolUsed: toolName, toolResult } = await runToolWithFallback(primary, subQuestion, { onUpdate });
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
  // Greedy decoding + tighter cap. Workers should be 2-4 sentences max;
  // longer outputs are usually padding.
  const raw = await model.chat(messages, {
    maxTokens: 160,
    temperature: 0,
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
  // Greedy decoding for the synthesizer too. The structure is rigid
  // (## sections + Bottom line) so determinism helps. Token cap 500 is
  // enough for 2-3 sections plus the takeaway.
  const raw = await model.chat(messages, {
    maxTokens: 500,
    temperature: 0,
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
    image = null,
    onPlanner = () => {},
    onWorkerStart = () => {},
    onWorkerUpdate = () => {},
    onWorkerDone = () => {},
    onSynthUpdate = () => {},
    onComplete = () => {},
    onError = () => {},
  } = callbacks;

  try {
    // Phase 1: planning. Image (if any) is consumed here and not propagated
    // to workers — they research the planner's sub-questions in text.
    const plan = await planQuery(model, query, { onUpdate: onPlanner, image });

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
