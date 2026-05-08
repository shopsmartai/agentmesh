# AgentMesh

> A thinking partner that argues with itself. Three Gemma 4 agents take different stances on your question, search the public web for evidence, and a synthesizer shows you where they disagree.
>
> Runs entirely in your browser. No server, no API key, no telemetry.

**[Try it live →](https://shopsmartai.github.io/agentmesh/)** *(Chrome 113+ with WebGPU; first run downloads ~3.1 GB, cached after, reload is instant)*

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE)
[![Built with Gemma 4](https://img.shields.io/badge/Built%20with-Gemma%204-blueviolet.svg)](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)
[![WebGPU](https://img.shields.io/badge/Runtime-WebGPU-orange.svg)](https://www.w3.org/TR/webgpu/)

Submission for the [DEV Gemma 4 Challenge](https://dev.to/challenges/google-gemma-2026-05-06).

---

## Why this exists

Single-shot LLMs collapse to one voice. Ask ChatGPT or Gemini "Is remote work better than in-office work?" and you get a balanced-sounding answer that hides the actual disagreement. The model is averaging perspectives behind the scenes.

AgentMesh does the opposite. It runs three agents on your question, each holding a fixed stance:

- **The skeptic** finds the strongest counter-arguments and real failure modes.
- **The advocate** makes the strongest case in favor.
- **The pragmatist** describes how it actually plays out in practice.

Each agent searches public sources (Wikipedia, Hacker News, arXiv, DuckDuckGo) for evidence supporting its stance. Then a synthesizer shows you where they agreed, where they disagreed, and reconciles. You see the *argument*, not just the conclusion.

This is something single-call models cannot do well, even when you ask them to. The architecture forces the disagreement to actually happen.

## What it looks like

```
                    [ user question ]
                           │
                           ▼
                  ┌────────────────┐
                  │   PLANNER      │  extracts topic, builds 3 search queries
                  └────────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │  SKEPTIC   │ │  ADVOCATE  │ │ PRAGMATIST │
     │ ┄┄┄┄┄┄┄┄   │ │ ┄┄┄┄┄┄┄┄   │ │ ┄┄┄┄┄┄┄┄   │
     │ searches   │ │ searches   │ │ searches   │
     │ for risks  │ │ for upside │ │ for usage  │
     └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                  ┌──────────────┐
                  │ SYNTHESIZER  │  agreed / disagreed / reconciliation
                  └──────────────┘
                          ▼
                   [ final answer ]
```

The page renders this as a live network visualization. Curved cyan and magenta connections appear between the cards as evidence flows from agents to the synthesizer. Each agent card streams its tokens in real time so you can watch the model think.

## Why Gemma 4 E2B specifically

Multi-perspective agents only work if the model has three properties at once. Gemma 4 E2B is the smallest open model that ships all three for browser deployment.

**Strong instruction following.** The skeptic must actually argue against the topic. The advocate must actually argue for it. Smaller models drift, hedge, or write balanced summaries that ignore their assigned stance. Gemma 4 stays in role.

**Refuses to invent facts.** Every worker is told: use only the research notes, do not invent. When the public sources do not support a strong claim, the agent says so directly. SmolLM2-360M (our lite-mode fallback) confabulates plausible content in the same situation. The hallucination resistance is what makes the disagreement *trustworthy* rather than theatrical.

**Fits in a browser tab.** Gemma 4 E2B at q4f16 is ~3.1 GB of weights. Chrome can hold that in a Web Worker alongside the ONNX runtime and a streaming UI. The next size up (E4B at ~6 GB) starts hitting memory pressure. The 26B and 31B variants are server-class. E2B is the only one that runs in a browser tab today.

E2B also supports a 128K context window. The synthesizer in this build does not need it, but the headroom means the architecture can grow.

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
│  │   3 perspe- │ ←──┤                                     │   │
│  │   ctives,   │    │  ORT WebGPU                          │   │
│  │   synth)    │    │                                     │   │
│  │              │    │  Gemma 4 E2B (q4f16)                │   │
│  │  ui.js       │    │   ↑ loaded from HF Hub on first run │   │
│  │  viz.js      │    │     and cached in browser storage   │   │
│  └──────┬───────┘    └─────────────────────────────────────┘   │
│         │                                                      │
│         ▼  (no servers contacted for inference)                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Tools (run via fetch, no API keys)                     │   │
│  │  - Wikipedia  - Hacker News  - DuckDuckGo  - arXiv      │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

**Why a Web Worker.** ORT compiles WebGPU shaders for Gemma 4 for ~30 to 60 seconds on first run. Done on the main thread, that means a frozen tab. We isolate the entire model life cycle in `model.worker.js` and message-pass tokens back so the UI stays interactive throughout.

**Why a service worker.** GitHub Pages does not let you set custom HTTP headers, so by default the page is not cross-origin isolated and `SharedArrayBuffer` is unavailable. ORT WebGPU needs SAB. `coi-serviceworker.js` (adapted from [gzuidhof/coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker), MIT) injects the right headers into every response, making the page cross-origin isolated on the second load.

**Why a deterministic planner.** Earlier versions used the model itself for planning. It took ~30 seconds and routinely failed the topic-extraction job (it would echo the user's full question into the search template). We replaced it with a small JavaScript function that strips question words and pulls out the topic. The planner card still appears; it just shows the result instantly.

**Why a cascading tool fallback.** If a worker's primary source returns nothing relevant, we cascade through DuckDuckGo and arXiv before giving up. We also gate Wikipedia results on title relevance, so loosely-related articles do not get treated as findings. The combined effect: workers find real evidence for most topical queries, and honestly report the gap when none exists.

## Files of interest

| File | What it does |
|---|---|
| `js/main.js` | Boot, screen orchestration, query input, swarm visualization wiring |
| `js/model.js` | Main-thread proxy for the worker. Exposes `model.load(onProgress)` and `model.chat(messages, opts)` |
| `js/model.worker.js` | The actual model life cycle: load, warmup, streaming chat, throttled progress |
| `js/agents.js` | Planner / perspective workers (skeptic, advocate, pragmatist) / synthesizer + cascading tool calls |
| `js/tools.js` | Wikipedia, Hacker News, DuckDuckGo, arXiv. All CORS-friendly, no keys |
| `js/viz.js` | Canvas 2D network visualization (curved connections, flowing particles) |
| `js/ui.js` | Agent card rendering, status badges, live markdown |
| `coi-serviceworker.js` | COOP/COEP header injector |

## Run locally

```bash
git clone https://github.com/shopsmartai/agentmesh.git
cd agentmesh
npx http-server . -p 4570 -c-1 --cors
open http://localhost:4570/
```

No build step. No `npm install`. Vanilla ES modules straight to the browser.

## Lite mode (fast preview, no 3 GB download)

Append `?model=smollm` to the URL to load SmolLM2-360M instead (~270 MB). Useful for previewing the swarm UX without committing to the full Gemma 4 download. Output quality is noticeably lower because SmolLM2 confabulates on thin notes; Gemma 4 does not. The architecture is identical.

## Try these questions

The architecture is built for opinionated, multi-stance questions. These work well:

- *Is remote work better than in-office work?*
- *Should I learn Rust in 2026?*
- *Is intermittent fasting actually beneficial?*
- *Is buying a Tesla a good investment in 2026?*
- *Does drinking coffee improve productivity?*

For purely factual lookups (*"What is photosynthesis?"*), the perspective framing does not add much. The skeptic does not really argue against photosynthesis. For those queries, you would be better served by reading the Wikipedia article directly. AgentMesh shines on questions that have actual disagreement.

## Known limitations

- **Sequential, not parallel.** WebGPU LLM inference is single-stream per session. The three perspective agents take turns on one GPU. The visualization with the connecting lines makes it look concurrent; logically it is, physically it is not.
- **Niche cross-domain queries return "no findings."** When the public sources do not have a strong opinion to mine, the agents honestly report the gap. This is correct behavior but a real UX limit.
- **3 GB is a wall.** No way around it for first-time visitors. We gate the download behind an explicit click and persist via the browser cache so reload is free. Lite mode (`?model=smollm`) is the escape hatch.
- **Multimodal disabled.** Gemma 4 E2B is multimodal. The code paths exist in `model.worker.js` for image input, but per-component dtype configuration on diverse Chrome drivers needs more debugging than the challenge time-box allowed. Documented future work.

## Future work

- **Native Gemma 4 tool-calling.** Replace the heuristic tool router with `apply_chat_template({ tools: [...] })`. Gemma 4 supports structured tool calls natively.
- **Multimodal, properly.** Resume the `Gemma4ForConditionalGeneration` work. Drop an image into the prompt; the planner sees it; the perspective agents argue about what is in it.
- **More than three perspectives.** A "historical" or "futurist" agent. A "user-of-the-thing" agent. The architecture trivially extends.
- **Persistent local notebook.** OPFS-backed query history so the model amortizes its 3 GB across many uses, and earlier perspective debates inform later ones.

## Credits

- [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) by Google DeepMind
- [Transformers.js](https://huggingface.co/docs/transformers.js) by Hugging Face, version 4.2.0
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) by gzuidhof, MIT
- [nico-martin/gemma4-browser-extension](https://github.com/nico-martin/gemma4-browser-extension), the reference implementation that pinned us to the working library version
- ONNX Runtime, the WebGPU spec, and everyone who keeps the open browser-ML stack moving

## License

[MIT](LICENSE), code only. The Gemma 4 model itself is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms); we link to it from HF Hub at runtime and never redistribute weights.
