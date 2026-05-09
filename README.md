# AgentMesh

> A thinking partner that argues with itself. Three Gemma 4 agents take different stances on your question, search public sources for evidence, and a synthesizer shows you where they disagree.

**[Try it live →](https://shopsmartai.github.io/agentmesh/)**

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE)
[![Built with Gemma 4](https://img.shields.io/badge/Built%20with-Gemma%204-blueviolet.svg)](https://ai.google.dev/gemma/docs/core/model_card_4)
[![WebGPU](https://img.shields.io/badge/Runtime-WebGPU-orange.svg)](https://www.w3.org/TR/webgpu/)

Submission for the [DEV Gemma 4 Challenge](https://dev.to/challenges/google-gemma-2026-05-06).

---

## Why this exists

Single-shot LLMs collapse to one voice. Ask ChatGPT or Gemini "Is the AI bubble about to burst?" and you get a balanced-sounding answer that hides the actual disagreement. The model is averaging perspectives behind the scenes.

AgentMesh does the opposite. It runs three agents on your question, each holding a fixed stance:

- **The skeptic** finds the strongest counter-arguments and real failure modes.
- **The advocate** makes the strongest case in favor.
- **The pragmatist** describes how it actually plays out in practice.

Each agent searches public sources (Wikipedia, Hacker News comments, arXiv, DuckDuckGo) for evidence. Then a synthesizer shows you where they agreed, where they disagreed, and reconciles. You see the *argument*, not just the conclusion.

This is something single-call models cannot do well, even when you ask them to. The architecture forces the disagreement to actually surface.

## Two modes

AgentMesh ships with two ways to run Gemma 4. Switch between them in the Settings panel.

| | Local mode | Cloud mode (BYOK) |
|---|---|---|
| Where the model runs | Your GPU, in a Web Worker | Google's servers, via your API key |
| Model | Gemma 4 E2B (q4f16, ~3.1 GB) | Gemma 4 26B-A4B MoE or 31B Dense |
| First-load cost | 3.1 GB download, cached after | Zero, instant |
| Per-query latency | 3 to 5 minutes on consumer GPU | 20 to 40 seconds |
| Privacy story | Nothing leaves your tab. Total. | Queries go through Google. Your key stays in your browser only. |
| API key needed | No | Yes, free, no credit card. Get one at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| Browser support | Chrome 113+ with WebGPU | Any modern browser |
| Default | Yes | Opt-in |

The four research tools (Wikipedia, HN comments, arXiv, DuckDuckGo) are called from the browser in both modes. They are public APIs that need no auth.

## What it looks like

```
                    [ user question ]
                           │
                           ▼
                  ┌────────────────┐
                  │   PLANNER      │  extracts topic deterministically
                  └────────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │  SKEPTIC   │ │  ADVOCATE  │ │ PRAGMATIST │
     │ ┄┄┄┄┄┄┄┄   │ │ ┄┄┄┄┄┄┄┄   │ │ ┄┄┄┄┄┄┄┄   │
     │ wiki + HN  │ │ wiki + HN  │ │ wiki + HN  │
     │ comments   │ │ comments   │ │ comments   │
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

The page renders this as a live network visualization. Curved cyan and magenta connections appear between the cards as evidence flows from agents to the synthesizer. Each agent card is color-coded by role (red skeptic, green advocate, amber pragmatist) and streams its tokens in real time.

## Why Gemma 4 specifically

Multi-perspective agents only work if the model has three properties at once. Gemma 4 ships all three, and the family covers both browser and server scales, which is why we can offer one architecture in two modes.

**Strong instruction following.** The skeptic must actually argue against the topic. The advocate must actually argue for it. Smaller models drift, hedge, or write balanced summaries that ignore their assigned stance. Gemma 4 stays in role.

**Refuses to invent facts when notes are thin.** Workers anchor in research notes when they support the stance, and use general training knowledge to fill gaps. They are honest about what is well-evidenced versus widely-held belief. The hallucination resistance is what makes the disagreement *trustworthy* rather than theatrical.

**Browser to server, one architecture.** Gemma 4 E2B at q4f16 is ~3.1 GB and fits in a browser tab. Gemma 4 26B-A4B and 31B are server-class and Google hosts them on AI Studio. Same model family, two scales, so the local and cloud paths produce comparable shape of answers.

E2B and the larger variants both expose a 128K context window, leaving room for richer multi-document grounding in future work.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ Browser tab                                                            │
│                                                                        │
│  ┌──────────────┐                                                      │
│  │  Main thread │   model.chat(messages, opts)                         │
│  │              │           ▼                                          │
│  │  agents.js  ─┼──→  ┌─────────────┐                                  │
│  │              │     │  router     │  picks Local or Cloud per call   │
│  │  ui.js       │     └──┬───────┬──┘                                  │
│  │  viz.js      │        │       │                                     │
│  └──────────────┘        ▼       ▼                                     │
│                     ┌────────┐  ┌──────────────────────────────┐       │
│                     │ Worker │  │ fetch -> Gemini API           │       │
│                     │ ORT GPU│  │ (your key, your browser only) │       │
│                     │ Gemma  │  └──────────────────────────────┘       │
│                     │ 4 E2B  │           ▲                             │
│                     └────────┘           │ HTTPS                       │
│                          ▲               ▼                             │
│         (no servers contacted)  generativelanguage.googleapis.com      │
│                                                                        │
│  Tools (run via fetch, no API keys, both modes):                       │
│  - Wikipedia  - Hacker News comments  - arXiv  - DuckDuckGo            │
└────────────────────────────────────────────────────────────────────────┘
```

**Why a Web Worker (local mode).** ORT compiles WebGPU shaders for Gemma 4 for ~30 to 60 seconds on first run. On the main thread that means a frozen tab. We isolate the entire model life cycle in `model.worker.js` and message-pass tokens back so the UI stays interactive throughout.

**Why a service worker.** GitHub Pages does not let you set custom HTTP headers, so by default the page is not cross-origin isolated and `SharedArrayBuffer` is unavailable. ORT WebGPU needs SAB. `coi-serviceworker.js` (adapted from [gzuidhof/coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker), MIT) injects the right headers into every response, making the page cross-origin isolated on the second load.

**Why a deterministic planner.** Earlier versions used the model itself for planning. It took ~30 seconds and routinely failed the topic-extraction job (it would echo the user's full question into the search template). Replaced with a small JavaScript function that strips question words and pulls out the topic. The planner card still appears; it just shows the result instantly.

**Why multi-source gather.** Workers fetch from Wikipedia AND Hacker News comments in parallel for every search. Wikipedia gives encyclopedic facts, HN comments give substantive user opinions. Workers see both as one notes block and decide what to anchor on.

**Why client-side stripping for cloud mode.** Gemma 4 on the Gemini API ignores `thinkingConfig.thinkingBudget=0` and dumps internal reasoning into the response. We strip drafts and refinement notes by finding the LAST occurrence of the synth's first heading and trimming everything before it.

## How to use cloud mode

1. Open the live site at [shopsmartai.github.io/agentmesh](https://shopsmartai.github.io/agentmesh/).
2. Click `[ settings ]` next to the boot button.
3. Switch the radio to **Cloud**.
4. Click the link in the panel to get a free Google AI Studio key (no credit card required).
5. Paste the key into the field. Click `[ save ]`. The panel closes and you go straight to the main screen.
6. The header now reads `v1.0 · gemma-4-26b-a4b-it (cloud)` (or whichever model you picked).
7. Pick a suggested question or type your own. ~30 seconds later you have a three-perspective answer.

Your key stays in your browser's localStorage. It is never sent to anyone except Google's API endpoint. You can clear it any time via `[ clear ]` in the same settings panel. The public code does not contain any API key, verified with `git grep` before each push.

## Files of interest

| File | What it does |
|---|---|
| `js/main.js` | Boot, screen orchestration, settings panel wiring, query input, swarm visualization |
| `js/model.js` | Routes `model.chat()` to the local Web Worker or the Gemini API. Cloud adapter with timeout, fallback chain, thinking-mode stripping. |
| `js/model.worker.js` | The local model life cycle: load, warmup, streaming chat, throttled progress |
| `js/settings.js` | localStorage-backed mode + cloud model + API key state. Notify-on-change so the UI stays in sync. |
| `js/agents.js` | Planner / perspective workers (skeptic, advocate, pragmatist) / synthesizer + multi-source gather |
| `js/tools.js` | Wikipedia, Hacker News stories + comments, DuckDuckGo, arXiv. All CORS-friendly, no keys. |
| `js/viz.js` | Canvas 2D network visualization (curved connections, flowing particles) |
| `js/ui.js` | Agent card rendering, status badges, live markdown for the synth panel |
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

Append `?model=smollm` to the URL to load SmolLM2-360M instead (~270 MB) for the local path. Useful for previewing the swarm UX without committing to the full Gemma 4 download. Output quality is noticeably lower because SmolLM2 confabulates on thin notes; Gemma 4 does not. The architecture is identical.

## Try these questions

The architecture is built for opinionated, multi-stance questions:

- *Is buying a Tesla a good investment in 2026?*
- *Is the AI bubble about to burst?*
- *Is buying a house better than renting in 2026?*
- *Should I learn Rust in 2026?*
- *Is intermittent fasting actually beneficial?*

Encyclopedic lookups (*"What is photosynthesis?"*) work too but the perspective framing does not add much value, since the skeptic does not really argue against photosynthesis. AgentMesh shines on questions that have actual disagreement to mine.

## Known limitations

- **Sequential, not parallel.** WebGPU LLM inference is single-stream per session in local mode. The three perspective agents take turns on one GPU. The visualization makes it look concurrent; logically it is, physically it is not. Cloud mode actually fires three parallel Gemini calls but the synthesizer still waits for all three before it runs.
- **Niche cross-domain queries return weak findings.** When neither Wikipedia nor Hacker News has substantive content on a topic, the agents lean on training knowledge alone. They are honest about which claims have evidence vs widely-held belief.
- **3 GB local-mode wall.** No way around it for local first-time visitors. Gated behind an explicit click and persists via the browser cache so reload is free. Cloud mode is the alternative.
- **Multimodal not enabled.** Gemma 4 E2B is multimodal. The code paths exist in `model.worker.js` for image input, but per-component dtype configuration on diverse Chrome drivers needs more debugging than the challenge time-box allowed. Documented future work.

## Future work

- **Native Gemma 4 tool-calling.** Replace the heuristic tool router with `apply_chat_template({ tools: [...] })`. Gemma 4 supports structured tool calls natively.
- **Multimodal, properly.** Resume the `Gemma4ForConditionalGeneration` work. Drop an image into the prompt; the planner sees it; the perspective agents argue about what is in it.
- **More perspectives.** A "historical" or "futurist" agent. A "user-of-the-thing" agent. The architecture trivially extends.
- **More sources.** Reddit JSON, Brave Search (with second BYOK key), Stack Exchange. Each broadens the grounding for opinionated queries.
- **Persistent local notebook.** OPFS-backed query history so past arguments inform later ones.

## Credits

- [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) by Google DeepMind
- [Transformers.js](https://huggingface.co/docs/transformers.js) by Hugging Face, version 4.2.0
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) by gzuidhof, MIT
- [nico-martin/gemma4-browser-extension](https://github.com/nico-martin/gemma4-browser-extension), the reference implementation that pinned us to the working library version
- ONNX Runtime, the WebGPU spec, and everyone who keeps the open browser-ML stack moving

## License

[MIT](LICENSE), code only. The Gemma 4 model itself is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms); we link to it at runtime and never redistribute weights.
