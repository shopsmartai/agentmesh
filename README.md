# AgentMesh

> A swarm of AI agents running entirely in your browser tab. No server, no API key, no install.

A research agent built for the [DEV Gemma 4 Challenge](https://dev.to/challenges/google-gemma-2026-05-06).

## How it works

1. You ask a research question
2. A **Planner** agent decomposes it into 3 focused sub-questions
3. **Worker** agents each take one sub-question, search Wikipedia or HackerNews, and write a focused answer
4. A **Synthesizer** combines all findings into a final markdown response

All inference runs locally via [Transformers.js](https://huggingface.co/docs/transformers.js) + WebGPU. Your queries never leave the tab.

## Tech

- Vanilla JS, HTML, CSS (no framework)
- Transformers.js v3 + WebGPU for in-browser inference
- Wikipedia REST API + HackerNews Algolia API + DuckDuckGo Instant Answers (no API keys)
- Pure Canvas 2D for the network visualization

## Try it

[Live demo →](https://shopsmartai.github.io/agentmesh/)

Requires a WebGPU-enabled browser (Chrome 113+, Edge 113+, Brave, Arc).
