<!--
Cover image: assets/og-cover.png
Tags: devchallenge, gemmachallenge, gemma, webgpu
Category: "Build With Gemma 4"
Title (under 70 chars): "I Built an AI That Argues With Itself, Running in a Browser Tab"
-->

# I Built an AI That Argues With Itself, Running in a Browser Tab

Live demo: https://shopsmartai.github.io/agentmesh/

Open in Chrome 113 or newer with a real GPU. The first time you load it, it downloads about 3.1 GB of Gemma 4 weights into your browser cache. After that, reload is instant. The whole thing is open source under MIT: https://github.com/shopsmartai/agentmesh

I asked it: "Is remote work better than in-office work?"

What came back was not a single balanced answer. It was three agents arguing. A skeptic listed real failure modes of remote work. An advocate listed concrete benefits. A pragmatist described how it actually plays out for different kinds of jobs. Then a synthesizer showed me where the three agreed, where they disagreed, and reconciled.

That is the whole point of this project. Single-shot LLMs collapse to one voice. They average perspectives behind the scenes and hide the disagreement. AgentMesh forces the disagreement to actually happen, with three agents that each have a fixed stance, and a synthesizer that calls out where they parted ways.

This post is about why I built it that way and the engineering it took to get there.

## Why three agents instead of one

Try this in your head. Ask any chat model "Is remote work better than in-office work?" You will get a balanced essay. The model will give you one paragraph for each side and a polite ending. It is fine. It is also boring. It is hiding what the model actually thinks each side's strongest argument is.

A multi-agent system can do something a single call cannot. You give each agent a fixed role and a separate context. The skeptic does not get to soften its view by also writing the advocate's paragraph. The advocate does not get to hedge. They have to commit. Only then do you reconcile.

This is the only architecture I know of where the multi-agent shape gives a concrete win that single-call inference cannot match. It is not "five agents read the same Wikipedia article and produce variants." It is three agents holding three positions and a synthesizer showing the gap.

The architecture also makes the answer better in a small way that matters. Each agent searches the public web for evidence supporting its stance. The skeptic searches for criticism, drawbacks, problems. The advocate searches for benefits, advantages, success. The pragmatist searches for real-world usage. Each one finds different sources. The synthesizer reconciles real evidence, not vibes.

## Why Gemma 4 E2B specifically

This architecture only works if the model has three properties at once. Gemma 4 E2B is the smallest open model I found that has all three.

**It actually stays in role.** Smaller models drift. You tell SmolLM2-360M "you are the skeptic, find counter-arguments" and it writes a balanced paragraph anyway because the underlying instinct is to be helpful. Gemma 4 stays sharp. The skeptic produces criticisms. The advocate produces upside. The roles do not blur.

**It refuses to invent facts.** Every agent is told: use only the research notes provided, do not make things up. Gemma 4 obeys. When the search returns articles that do not address the question, the agent says so directly. SmolLM2 in the same situation produces confident-sounding content from training data. The disagreement only matters if it is grounded; Gemma 4 keeps it grounded.

**It fits in a browser tab.** Gemma 4 E2B at q4f16 is about 3.1 GB of weights. Chrome can hold that in a Web Worker alongside the ONNX runtime and a streaming UI. The next size up, Gemma 4 E4B, is around 6 GB and starts hitting browser memory limits. The 26B and 31B variants are server class. E2B is the only one that runs in a normal browser tab.

E2B also has a 128K context window. I am not using all of it yet, but the headroom means the architecture can grow. A future version could feed the agents long documents instead of short search snippets.

## What the demo actually looks like

You type a question. You watch five cards appear:

```
                    [ user question ]
                           │
                           ▼
                  ┌────────────────┐
                  │   PLANNER      │
                  └────────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │  SKEPTIC   │ │  ADVOCATE  │ │ PRAGMATIST │
     └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                  ┌──────────────┐
                  │ SYNTHESIZER  │
                  └──────────────┘
```

The planner extracts the topic from your question. Then three perspective agents fire, each with a different system prompt and a different web search. They stream their views as live tokens. The synthesizer reads all three and writes a structured response: where they agreed, where they disagreed, and a reconciliation. Each card is color-coded by role (red for skeptic, green for advocate, amber for pragmatist) so you can see who said what.

Everything runs locally on your GPU. The only network calls are to Hugging Face Hub for the model weights on first load, and to the public search APIs (Wikipedia, Hacker News, arXiv, DuckDuckGo) for evidence. No inference goes through any server I run.

## The engineering problems I had to solve

I expected this to take a weekend. It took a week. Naming the problems is the actual point of this post because most of them are reusable knowledge for anyone trying to ship browser-LLM apps.

### The page was not cross-origin isolated

GitHub Pages does not let you set custom HTTP headers. Without two specific headers (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless`), the browser will not enable `SharedArrayBuffer`. The ONNX runtime needs `SharedArrayBuffer` to run on WebGPU. Without it, every model load failed with raw numeric errors that pointed into the WebAssembly heap (literal numbers like `11514632`).

Fix: ship a service worker (adapted from coi-serviceworker, MIT) that intercepts every response and adds the missing headers. First load registers the worker and reloads. After that, the page is cross-origin isolated and `SharedArrayBuffer` becomes available.

If you build anything like this on GitHub Pages, you will hit this. Plan for it.

### WebGPU compile froze the tab for a minute

The first time the ONNX runtime loads a model on WebGPU, it has to compile shaders for every operation in the network. For Gemma 4 E2B that takes anywhere from thirty seconds to over a minute. Done on the main thread, the page becomes unresponsive. Chrome eventually offers to kill the tab.

Fix: put the entire model life cycle in a Web Worker. The page now sees a thin proxy that exposes `model.load(onProgress)` and `model.chat(messages, opts)`. The proxy talks to the worker over `postMessage`. The worker does the heavy work; the page stays interactive.

### The worker flooded the main thread

After moving the model into the worker, the page froze again for a different reason. Transformers.js fires a progress callback for every chunk of the download. With a 1.5 GB file, that is hundreds of events per second. I was forwarding each one to the main thread via `postMessage`. The main thread message queue saturated.

Fix: throttle progress messages to one every 100 milliseconds. Always send state changes immediately (started, finished, ready) but coalesce the actual progress percentages.

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

After this, the page stayed smooth for the entire 3 GB download.

### The browser cache poisoned itself

When a model load failed mid-download, Transformers.js cached the partial blob. The next page load fetched the corrupt blob from cache and the runtime threw on it. Reloading did not help. The cache was the source of the corruption and it persisted across sessions.

Fix in v3 of the library: disable the cache layer entirely. Fix in v4: the bug does not exist, partial downloads do not get committed.

### Transformers.js 3.5.1 was just broken

Every model and every backend combination failed with raw numeric errors. SmolLM2, which works fine elsewhere, failed the same way as Gemma 4. I lost a day thinking my code was wrong.

Fix: pin to 4.2.0. The diagnostic process was simple. Find a known-working public project that uses Gemma 4 (the gemma4-browser-extension by nico-martin). Copy their version pin. Worked first try.

### Wikipedia search returned nothing for verbose questions

Pressure-testing the live site exposed a fundamental issue. Even simple queries like "What is photosynthesis?" were producing answers like "research notes are empty." Two of three workers were giving up.

The bug was upstream. The model-driven planner was generating verbose sub-questions like "What is the fundamental definition of photosynthesis and its core chemical processes?". Wikipedia's classic search API does title-prefix matching. That string is not the prefix of any Wikipedia article title, so it returned nothing.

Three changes fixed this:

1. Switch to Wikipedia's full-text search API instead of the title-prefix one.
2. Strip filler words from the search query before sending it. A list of about 60 question-shape filler words ("fundamental", "core", "exemplified", "current", "associated") gets removed so the topic word ranks highest.
3. Add a relevance gate. If the search returns three articles but none of their titles contain a meaningful word from the query, treat that as zero results and cascade to a different source (DuckDuckGo, then arXiv).

After these changes, "fundamental photosynthesis chemical processes" reliably ranks the actual Photosynthesis article first.

### The planner was spending 30 seconds on something a regex could do

The original planner used the model itself to extract the topic and generate three search queries. It took about 30 seconds on Gemma 4. Then I noticed the model was just echoing the user's full question into the template. It was not actually doing extraction. It was a slow templating engine.

Fix: replace the planner's model call with a small JavaScript function. Strip leading question words and trailing punctuation. Pull out the topic. Build three search queries. Done in milliseconds.

The planner card still appears in the swarm visualization. It just shows the topic immediately rather than streaming through 30 seconds of model generation. Net result: 30 seconds saved, plus better-quality search queries downstream.

### Multimodal did not ship

Gemma 4 E2B is multimodal. You can drop an image into the prompt and the model will describe what it sees. I tried to wire this up. It did not work in the time I had.

The recipe in the model card uses a different API than the one I had been using. Instead of the simple `pipeline('text-generation', ...)` call, you have to instantiate `Gemma4ForConditionalGeneration` directly along with `AutoProcessor`, and configure different precision settings for each component (the audio encoder and vision encoder need fp16, while the decoder and embeddings can stay at q4f16).

I implemented all of that. The load kept falling through to SmolLM2 instead. Two bugs in the path. The processor signature has four arguments and I was passing three (so the options object was being interpreted as audio input). And the q4f16 audio encoder weights threw on some Chrome drivers.

Both are fixable. I did not get them fixed in time. The multimodal code is still in the repository under `loadGemma4` and `chatViaGemma` in the worker file. The current production demo is text only. I will pick this up after the challenge.

## What I want to be honest about

A few things people might assume from the demo that are not actually true.

**The agents are not running in parallel.** WebGPU LLM inference is single-stream per session. The three agents take turns on one GPU. The visualization with connecting lines makes it look concurrent. Logically it is, physically it is not.

**Niche cross-domain queries return "no findings."** When the public sources do not have an opinion to mine, the agents honestly report the gap. This is correct behavior but a real UX limit. Asking "What is the fastest way to optimize my custom internal tool?" will not find evidence anywhere on the open web.

**The 3 GB download is a wall.** There is no way around it for first-time visitors. I gate the download behind an explicit click and persist via the browser cache so reload is free. There is also a lite mode at `?model=smollm` that loads SmolLM2-360M (about 270 MB). Output quality is noticeably lower in lite mode because SmolLM2 confabulates on thin notes. The architecture is identical.

**Factual lookup queries do not benefit from this architecture.** If you ask "What is photosynthesis?" the skeptic does not really argue against photosynthesis. Three perspective agents on a factual lookup just produces three slightly different summaries of the same article. AgentMesh shines on questions that have actual disagreement.

## Try it

Default with Gemma 4 (3.1 GB first load): https://shopsmartai.github.io/agentmesh/

Lite mode with SmolLM2 (270 MB): https://shopsmartai.github.io/agentmesh/?model=smollm

Source: https://github.com/shopsmartai/agentmesh

These questions work well because they have real disagreement to mine:

- *Is remote work better than in-office work?*
- *Should I learn Rust in 2026?*
- *Is intermittent fasting actually beneficial?*
- *Is buying a Tesla a good investment?*
- *Does drinking coffee improve productivity?*

## What I would build next

If I keep working on this past the challenge, four things in priority order.

**Native Gemma 4 tool calling.** The model has a structured tool-call format built into its chat template. Right now I use a simple heuristic to pick a tool. Switching to the native format would make tool selection cleaner and add another concrete reason the project specifically uses Gemma 4.

**Multimodal, properly.** Resume the work I rolled back. Drop an image into the prompt. The skeptic argues against what is in it. The advocate argues for. The pragmatist describes how it gets used.

**More than three perspectives.** A "historical" agent (how did this work in the past?). A "futurist" agent (where is this going?). A "user-of-the-thing" agent. The architecture trivially extends.

**A persistent local research notebook.** Save past arguments. Build context across sessions. Make the 3 GB download amortize across many uses. This is the path that turns the project from a demo into something useful for real work.

I am not committing to any of these yet. What I do next depends on what happens with this post. If a hospital CTO emails me about clinical literature review with locally-bound data, I will go faster on the notebook. If five engineers fork the repo, I will go faster on the multimodal and tool-calling pieces. If neither, this stays a portfolio piece I am happy with.

## Thanks

To Google DeepMind for releasing Gemma 4 with weights small enough to actually fit in a browser tab. To Hugging Face for Transformers.js, especially the 4.2.0 release that finally worked end to end. To gzuidhof for the coi-serviceworker library, which solved the cross-origin isolation problem in about five minutes once I knew it existed. To nico-martin for publishing the gemma4-browser-extension code, which was my reference for which library version actually loads this model. To everyone who keeps the open browser-ML stack moving forward.

The code is MIT. Fork it. Rip out the parts you want. If you build something better with it, I would love to see what.
