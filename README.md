# AgentMesh

> A multi-agent research swarm running entirely in your browser tab on Gemma 4 E2B. No server, no API key, no install.

**[→ Try it live](https://shopsmartai.github.io/agentmesh/)** (Chrome 113+ with WebGPU; first run downloads ~3.1 GB, cached after — reload is instant)

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE)
[![Gemma 4 Challenge](https://img.shields.io/badge/Built%20with-Gemma%204-blueviolet.svg)](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)
[![WebGPU](https://img.shields.io/badge/Runtime-WebGPU-orange.svg)](https://www.w3.org/TR/webgpu/)

Submission for the [DEV Gemma 4 Challenge](https://dev.to/challenges/google-gemma-2026-05-06).

---

## What it does

You ask a research question. Five AI agents collaborate to answer it — entirely on your GPU, with no network calls except to fetch open public data (Wikipedia, Hacker News, arXiv).

```
            ┌──────────┐
            │ PLANNER  │   "decompose this query into 3 angles"
            └────┬─────┘
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   ┌──────┐  ┌──────┐  ┌──────┐
   │  W1  │  │  W2  │  │  W3  │   each picks a tool (wiki / HN / arxiv),
   │      │  │      │  │      │   reads notes, drafts a focused answer
   └──┬───┘  └──┬───┘  └──┬───┘
      └─────────┼─────────┘
                ▼
           ┌──────────┐
           │ SYNTHESIZER │   combines findings into structured markdown
           └──────────┘
```

The cyberpunk visualization renders curved cyan/magenta connections between agents with flowing particles when one completes its work.

## Why Gemma 4 E2B specifically

Multi-agent systems collapse without three properties — and Gemma 4 E2B is the smallest open model that delivers all three in a browser:

1. **Strong instruction-following.** The planner has to emit a numbered list of 3 sub-questions, no preamble, in the user's domain. The synthesizer has to follow a strict markdown format (`## sections`, `**Bottom line:**`). Smaller models drift, hallucinate, or copy the example prompt verbatim. Gemma 4 E2B doesn't.

2. **Hallucination resistance.** Workers are explicitly told to refuse to invent facts when the research notes are thin. Gemma 4 E2B *actually obeys* — when the tool returns nothing useful, the agent says "the research findings do not contain that information" rather than confabulating. SmolLM2-360M (our lite-mode fallback) confabulates freely on the same prompts.

3. **Browser footprint.** Gemma 4 E2B q4f16 is ~3.1 GB of weights — the only frontier-quality open model small enough to fit in browser memory alongside the COI service worker, ORT WebGPU runtime, and a streaming UI. Gemma 4 E4B (~6 GB) and the 31B/26B variants are server-class.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Browser tab                                                    │
│                                                                │
│  ┌──────────────┐    ┌─────────────────────────────────────┐   │
│  │  Main thread │    │  Web Worker (model.worker.js)       │   │
│  │              │    │                                     │   │
│  │  agents.js  ─┼─→  │  Transformers.js v4.2.0             │   │
│  │  (planner,  │ pm │  pipeline('text-generation', ...)    │   │
│  │   workers,  │ ←──┤                                     │   │
│  │   synth)    │    │  ORT WebGPU                          │   │
│  │              │    │                                     │   │
│  │  ui.js       │    │  Gemma 4 E2B (q4f16)                │   │
│  │  viz.js      │    │   ↑ loads from HF Hub on first run  │   │
│  └──────┬───────┘    └─────────────────────────────────────┘   │
│         │                                                      │
│         ▼  (no servers contacted for inference)                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Tools (run via fetch, no API keys)                     │   │
│  │  - Wikipedia  - Hacker News  - DuckDuckGo  - arXiv      │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Why a Web Worker?

ORT compiles WebGPU shaders for Gemma 4 E2B for ~30–60 seconds on first run. On the main thread that means a frozen tab. We isolate everything (model load, generation, streaming) in `model.worker.js` and message-pass tokens back so the UI stays interactive throughout. Progress messages are throttled to 100 ms intervals to prevent main-thread message-queue floods during the 3 GB download.

### Why a service worker

GitHub Pages doesn't allow setting custom HTTP headers, so by default the page is **not** cross-origin isolated and `SharedArrayBuffer` is unavailable. ORT WebGPU + ORT WASM threaded both require SAB. `coi-serviceworker.js` (adapted from [gzuidhof/coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker), MIT) injects `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless` headers into every response, making the page cross-origin isolated on the second load. After this lands, SAB becomes available and ORT can actually run.

### Why a cascading tool fallback

Wikipedia's `opensearch` API does title-prefix matching, not semantic search. Verbose sub-questions like *"How does WebGPU enable in-browser AI inference?"* return zero matches — at which point the worker has no notes to summarize and Gemma 4 honestly reports nothing. We cascade through `[primary] → DuckDuckGo → arXiv` so most queries land actual content even when the heuristic-picked first tool comes up dry.

## Files of interest

| File | What it does |
|---|---|
| `js/main.js` | Boot sequence, screen orchestration, query input, swarm visualization wiring |
| `js/model.js` | Main-thread proxy for the worker — exposes `model.load(onProgress)` and `model.chat(messages, opts)` |
| `js/model.worker.js` | The actual model lifetime: load, warmup, streaming chat, throttled progress |
| `js/agents.js` | Planner / worker / synthesizer prompts + parsing + cascading tool calls |
| `js/tools.js` | Wikipedia, Hacker News, DuckDuckGo, arXiv — all CORS-friendly, no keys |
| `js/viz.js` | Canvas 2D network visualization (curved connections, flowing particles) |
| `js/ui.js` | Agent card rendering, status badges, markdown |
| `coi-serviceworker.js` | COOP/COEP header injector |

## Run locally

```bash
git clone https://github.com/shopsmartai/agentmesh.git
cd agentmesh
npx http-server . -p 4570 -c-1 --cors
open http://localhost:4570/
```

That's it — no build step, no `npm install`. Vanilla ESM straight to the browser.

## Lite mode (fast preview, no 3 GB download)

`https://shopsmartai.github.io/agentmesh/?model=smollm` loads SmolLM2-360M instead (~270 MB). Useful for previewing the swarm UX without committing to the full Gemma 4 download. Output quality is noticeably lower — SmolLM2 confabulates on thin notes; Gemma 4 doesn't — but the architecture is identical.

## Known limitations / future work

- **Multimodal image input.** Code paths for `Gemma4ForConditionalGeneration` + `AutoProcessor` are wired in `model.worker.js` but not yet enabled as the default — initial attempts hit per-component dtype + processor signature issues that need more debug iteration. Once stable, you'll be able to drop a screenshot or photo into the query and the planner will see it.
- **Sequential, not parallel.** WebGPU LLM inference is single-stream per session, so the three workers actually timeshare on one GPU. The visualization's "parallel" framing is logical, not physical.
- **Wikipedia search.** The `opensearch` API is title-prefix only; verbose sub-questions miss. We mitigate via the cascade, but switching to `srsearch` (full-text) would be cleaner.
- **First-run UX.** 3 GB is a wall. We gate the download behind an explicit click and persist via the browser Cache API so reload is free, but there's no resume-on-network-failure yet.

## Credits

- [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) — Google DeepMind
- [Transformers.js](https://huggingface.co/docs/transformers.js) — Hugging Face
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) — gzuidhof, MIT
- ONNX runtime, ONNX-community model conversions, the open WebGPU stack

## License

[MIT](LICENSE) — code only. The Gemma 4 model itself is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms); we link to it from HF Hub at runtime.
