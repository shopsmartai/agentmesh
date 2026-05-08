<!--
Cover image: assets/og-cover.png
Tags: devchallenge, gemmachallenge, gemma, webgpu, javascript
Category: "Build With Gemma 4"
Title (under 70 chars): "I Ran a Five-Agent Research Swarm in a Browser Tab on Gemma 4"
-->

# I Ran a Five-Agent Research Swarm in a Browser Tab on Gemma 4

> No server. No API key. No install. Just Chrome, ~3 GB of weights, and seven engineering problems I had to solve before the demo would render its first token.

**Live demo:** https://shopsmartai.github.io/agentmesh/  *(Chrome 113+ with WebGPU; first run downloads ~3.1 GB, cached after)*

**Repo (MIT):** https://github.com/shopsmartai/agentmesh

I asked it *"What is photosynthesis?"* and 210 seconds later got this back:

```markdown
## Definition and Core Process

Photosynthesis is a biological process utilized by organisms such as plants,
algae, and cyanobacteria. This process involves converting light energy,
typically sunlight, into chemical energy, which is subsequently used to fuel
the organisms' metabolism.

## Organisms and Ecosystem Roles

  Participants: Photosynthesis is carried out by photopigment-bearing
  autotrophic organisms, including plants, algae, and cyanobacteria.

  Ecosystem Function: This process is a fundamental component of ecosystems,
  driving energy flows and nutrient cycles through the interaction of these
  organisms with their environment.

## Energy Conversion and Limitations

  Efficiency: The efficiency of photosynthesis is measured by the fraction of
  light energy successfully converted into chemical energy within the process
  in green plants and algae.

  Light Spectrum Limitation: Photosynthetic organisms are restricted ...

**Bottom line:** Photosynthesis converts solar energy into chemical fuel for
autotrophic organisms, and its efficiency is bounded by light absorption
characteristics.
```

That output came from five Gemma 4 E2B agents running in sequence on my GPU. It's grounded in three Wikipedia articles (Photosynthesis, Photopigment, Photosynthetic efficiency) — not training-data recall.

This post is about the engineering. What it took to get here, what broke along the way, and what I'd do differently.

---

## What it actually does

```
            ┌──────────┐
            │ PLANNER  │   "decompose this query into 3 angles"
            └────┬─────┘
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   ┌──────┐  ┌──────┐  ┌──────┐
   │  W1  │  │  W2  │  │  W3  │   each picks a tool
   │      │  │      │  │      │   (wiki / HN / arxiv / DDG),
   └──┬───┘  └──┬───┘  └──┬───┘   reads notes, drafts an answer
      └─────────┼─────────┘
                ▼
           ┌─────────────┐
           │ SYNTHESIZER │   combines into structured markdown
           └─────────────┘
```

User types a question → Planner emits three sub-questions → three Workers each search Wikipedia (or Hacker News, DuckDuckGo, arXiv) → Synthesizer combines into `## sections` with a `**Bottom line:**` closer.

Everything runs in `model.worker.js`, isolated from the UI thread. Tokens stream back over `postMessage` so the page never freezes — not even during ORT WebGPU shader compilation, which takes ~30–60 seconds the first time.

Source code is ~1,500 lines of vanilla JS. No framework, no build step, no `npm install`. Open `index.html` directly with `npx http-server .` and it runs.

---

## Why Gemma 4 E2B specifically

Multi-agent systems collapse without three properties. Gemma 4 E2B is the smallest open model that ships all three for browser deployment.

### 1. Strong instruction-following

The planner has to emit three numbered sub-questions, no preamble, in the user's domain. The synthesizer has to follow a specific markdown shape. Smaller models drift, hallucinate, or copy the example prompt verbatim.

Here's Gemma 4 E2B's planner output for *"How does WebGPU enable in-browser AI inference?"*:

```
1. What is WebGPU and how does it facilitate in-browser AI inference?
2. How is WebGPU utilized in practical examples for deploying AI inference
   within web applications?
3. What are the limitations and trade-offs when using WebGPU for in-browser
   AI inference?
```

Three distinct angles (definition / examples / limitations), all topic-anchored, perfectly formatted. SmolLM2-360M (the lite-mode fallback) for the same prompt produced unstructured JSON-shaped output that my parser had to recover from with regex.

### 2. Hallucination resistance — the actual differentiator

Workers are told explicitly: *"Do not invent facts."*

Gemma 4 E2B obeys. When Wikipedia returns articles whose content doesn't address the sub-question, the worker says so:

> "The provided research notes do not contain information about how WebGPU and WebGL are implemented or utilized in practical examples for deploying ML inference on the web."

SmolLM2 in the same situation produces confident-sounding text that's confabulated from training data. The fact that I can put this in a research demo without it lying back at users is the *actual* product property worth shipping.

### 3. Browser footprint

Gemma 4 E2B q4f16 is ~3.1 GB of weights — Chrome can hold that in a Web Worker alongside ORT WebGPU, the COI service worker, and a streaming UI. Gemma 4 E4B (~6 GB) and 31B (server-class) can't.

E2B is also *the only* Gemma 4 variant where 128 K context fits browser memory. That matters because in the swarm, agents could share full notes between phases (currently we summarize because the prompts are still small, but the headroom is there).

---

## The seven engineering problems

I expected this to be a weekend. It became a week. Each problem stalled the demo for hours; surfacing them is the actual point of this post.

### 1. The page wasn't cross-origin isolated

GitHub Pages doesn't let you set custom HTTP headers. Without `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy`, the page isn't *cross-origin isolated*, which means `SharedArrayBuffer` is unavailable. ORT WebGPU and threaded WASM both need SAB. Without it, ORT throws raw numeric pointer errors (literally `11514632`, `12077640` — pointers into the WASM heap) for every model + backend combination.

**Fix:** ship a service worker (`coi-serviceworker.js`, MIT, adapted from [gzuidhof/coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)) that intercepts every response and injects the headers. First load registers + reloads; second load is isolated.

```js
event.respondWith(
  fetch(request).then((response) => {
    const newHeaders = new Headers(response.headers);
    newHeaders.set('Cross-Origin-Embedder-Policy', 'credentialless');
    newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');
    newHeaders.set('Cross-Origin-Resource-Policy', 'cross-origin');
    return new Response(response.body, {
      status: response.status,
      headers: newHeaders,
    });
  })
);
```

Verify in DevTools: `self.crossOriginIsolated` must be `true`. If it's not, nothing else matters.

### 2. WebGPU compile froze the tab for 60 seconds

ORT compiles WebGPU shaders for Gemma 4 E2B for ~30–60 seconds on first run. On the main thread, that means a frozen tab. Chrome eventually offers to kill the page.

**Fix:** isolate the entire model lifetime in a Web Worker. Main thread keeps a thin proxy:

```js
this.worker = new Worker(
  new URL('./model.worker.js', import.meta.url),
  { type: 'module' }
);
this.worker.postMessage({ type: 'load', forceCandidate });
```

Tokens flow back over `postMessage` and re-emit through the existing `chat(messages, opts)` interface. `agents.js` and `main.js` don't change.

### 3. The worker flooded the main thread

Transformers.js fires `progress_callback` for every fetch chunk. Streaming a 1.5 GB file = hundreds of events per second. Forwarding each one via `postMessage` floods the main thread's message queue. The page freezes despite the actual work being in a worker.

**Fix:** throttle to one message per 100 ms. Force-flush state transitions:

```js
let lastProgressPostAt = 0;
function postProgress(payload, force = false) {
  const now = performance.now();
  if (force || now - lastProgressPostAt >= 100) {
    lastProgressPostAt = now;
    post(payload);
  }
}
```

### 4. The Cache API kept poisoning itself

When a Gemma 4 attempt failed mid-download, Transformers.js cached the partial blob. The next load fetched the corrupt blob from cache and ORT threw a numeric error on it. Reloading didn't help; the cache was the source of corruption.

**Fix in v3:** disable the Cache API entirely (`tf.env.useBrowserCache = false`) and rely on HTTP `Cache-Control` headers from HF Hub. **Fix in v4 (current):** the bug doesn't exist — partial downloads aren't committed.

### 5. Transformers.js v3.5.1 was broken

Every model + every backend (`webgpu/q4`, `webgpu/q4f16`, `wasm/q4`) failed with raw numeric ORT errors. SmolLM2 — which I knew worked elsewhere — failed the same way.

**Fix:** pin `@huggingface/transformers@4.2.0`. The diagnostic was process: pin to a version the reference impl ([nico-martin/gemma4-browser-extension](https://github.com/nico-martin/gemma4-browser-extension)) uses, retest. Worked first try.

### 6. Wikipedia's `opensearch` returned nothing for verbose questions

Pressure-testing on the live site exposed that even *"What is photosynthesis?"* was getting 2 of 3 workers reporting "research notes are empty." The planner generates verbose sub-questions like *"What is the fundamental definition of photosynthesis and its core chemical processes?"* — Wikipedia's `opensearch` API does title-prefix matching, and that string doesn't title-match.

**Fix part A:** switch to `srsearch` (full-text):

```js
const url = `https://en.wikipedia.org/w/api.php?action=query&list=search`
          + `&format=json&origin=*&srlimit=3`
          + `&srsearch=${encodeURIComponent(condensed)}`;
```

**Fix part B:** strip filler from the query so TF-IDF ranks correctly. A 60-word stopword list removes question-shape filler ("fundamental", "core", "exemplified", "associated", "current", "environmental"). Verified result: `"fundamental photosynthesis chemical processes"` now returns `Photosynthesis` as result #1.

**Fix part C:** add a relevance gate. If no Wikipedia result *title* contains any distinctive word from the query, treat as empty so the cascade falls through to DuckDuckGo and arXiv. This catches the "loosely-related noise" case where Wikipedia returns three articles that share *some* terms but don't address the question.

### 7. Multimodal didn't ship — and the failure is its own story

Gemma 4 E2B is multimodal in theory. The recipe in the model card is `AutoProcessor` + `Gemma4ForConditionalGeneration`, with a per-component `dtype` config:

```js
const dtype = {
  audio_encoder: 'fp16',     // q4f16 lacks WebGPU kernels
  vision_encoder: 'fp16',
  embed_tokens: 'q4f16',
  decoder_model_merged: 'q4f16',
};
```

I implemented it. The load path failed silently in the worker — falling through the candidate fallback chain to SmolLM2. Two bugs in that path:

1. The `processor()` signature is `(prompt, image, audio, options)` — four args. My calls were three, so the options object was being interpreted as audio input.
2. Even after fixing the signature, q4f16 audio encoder weights threw ORT errors on some Chrome drivers.

I rolled back. The multimodal code paths stay in `model.worker.js` (`loadGemma4`, `chatViaGemma`) as documented future work. The current production demo runs text-only.

This is the kind of thing that doesn't land in a hackathon time-box but lands cleanly in a follow-up. Better than shipping it broken.

---

## What I didn't build (and why)

A few things worth naming explicitly so you don't infer them from the demo:

- **The agents are not parallel.** WebGPU LLM inference is single-stream per session. The five-agent pipeline visualization implies parallelism; in reality the workers timeshare on one GPU. *Logical* parallelism, not physical. Real parallelism would require multiple ORT sessions or a multi-tenant inference server, which kills the "browser-only" pitch.

- **The synth is good, not great.** Multi-agent works best when workers research *different* sources and bring back genuinely distinct evidence. Mine often all land on Wikipedia and read variants of the same article — so the synth ends up restating the workers, not transcending them. Fixable with native Gemma 4 tool-calling (which I have wired but didn't ship), proper specialization per worker, and a wider tool catalog.

- **Niche cross-domain queries return "no findings."** *"WebGPU vs WebGL for ML inference"* — Wikipedia doesn't have an article for that. arXiv has tangentially-relevant papers. The relevance gate fires, the cascade exhausts, and Gemma 4 honestly reports the gap. That's a *correct* failure mode, not a bug — it's the alternative to fabrication. But it's a UX limit users will hit.

- **3 GB is a wall.** No way around it for first-time visitors. We gate behind an explicit click and persist via the browser Cache API so reload is free. There's a `?model=smollm` lite mode (~270 MB) that loads SmolLM2-360M for visitors who want to see the swarm without the commitment. Quality is noticeably lower; the architecture is identical.

---

## Try it

**Default (Gemma 4 E2B, ~3.1 GB first run):** [https://shopsmartai.github.io/agentmesh/](https://shopsmartai.github.io/agentmesh/)

**Lite mode (SmolLM2-360M, ~270 MB):** [https://shopsmartai.github.io/agentmesh/?model=smollm](https://shopsmartai.github.io/agentmesh/?model=smollm)

**Source:** [github.com/shopsmartai/agentmesh](https://github.com/shopsmartai/agentmesh) — MIT, ~1,500 LOC, no build step.

**Suggested first queries** (these have good Wikipedia coverage and produce clean outputs):

- *"What is photosynthesis?"*
- *"What is the difference between Mixture of Experts and Dense models?"*
- *"How do enzymes work?"*

Niche cross-domain queries (e.g. *"latest 2026 transformer papers"*) will hit the honest "no findings" path. That's intentional.

---

## What's next

If I keep working on this past the challenge:

1. **Native Gemma 4 tool-calling.** Replace the heuristic tool router with `apply_chat_template({ tools: [...] })`. Gemma 4's chat template supports structured tool calls natively. Adds another rubric line to the "specifically Gemma 4" argument.

2. **Multimodal, properly.** Resume the `Gemma4ForConditionalGeneration` work. Drop an image into the query, the planner sees it, the swarm researches what's in it. The code paths exist; the bugs are in the per-component dtype + processor signature space and are debuggable.

3. **Worker specialization.** Right now all three workers use the same prompt template. Differentiating them ("you focus on definitions", "you focus on counter-evidence") would actually use the multi-agent shape.

4. **Persistent local research notebook.** OPFS-backed history so the model amortizes its 3 GB across many uses, and earlier queries inform later ones. This is the path that makes the project useful for real work, not just demos.

I'm not committing to any of these yet — depends on response to this post. If a hospital CTO emails about clinical-literature review, I'll go faster on (4). If five engineers fork the repo, I'll go faster on (1) and (2).

---

## Acknowledgements

- [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) — Google DeepMind
- [Transformers.js](https://huggingface.co/docs/transformers.js) — Hugging Face, version 4.2.0
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) — gzuidhof, MIT
- [nico-martin/gemma4-browser-extension](https://github.com/nico-martin/gemma4-browser-extension) — the reference impl that proved the v4 API at version 4.2.0 with this exact model
- [webml-community/Gemma-4-WebGPU](https://huggingface.co/spaces/webml-community/Gemma-4-WebGPU) — the official multimodal demo whose bundle taught me the per-component dtype trick

---

*MIT-licensed. Fork it, modify it, ship something better. The challenge submission is the floor, not the ceiling.*
