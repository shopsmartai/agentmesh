<!--
Cover image: assets/og-cover.png
Tags: devchallenge, gemmachallenge, gemma, webgpu
Category: "Build With Gemma 4"
Title (under 70 chars): "I Ran a Five Agent Research Swarm in a Browser Tab on Gemma 4"
-->

# I Ran a Five Agent Research Swarm in a Browser Tab on Gemma 4

Live demo: https://shopsmartai.github.io/agentmesh/

You'll need Chrome 113 or newer with a real GPU. The first time you load the page, it downloads about 3.1 GB of model weights. After that the model lives in your browser cache and reload is instant. The whole thing is open source under MIT: https://github.com/shopsmartai/agentmesh

I asked it "What is photosynthesis?" and waited 210 seconds. Five AI agents running on my laptop GPU passed work to each other and produced this:

```
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

Bottom line: Photosynthesis converts solar energy into chemical fuel for
autotrophic organisms, and its efficiency is bounded by light absorption.
```

That answer is grounded in three Wikipedia articles. It is not the model recalling something from training. It is the model reading notes and writing about them.

I want to talk about how it actually works, why I picked Gemma 4 for this, and the seven things that broke along the way before the demo would render its first token.

## What it actually does

There are five agents in the pipeline. They look like this:

```
            Planner
              |
       /------+------\
       |      |      |
   Worker1 Worker2 Worker3
       |      |      |
       \------+------/
              |
         Synthesizer
```

The Planner reads your question and writes three sub-questions. Each Worker takes one sub-question, picks a research tool (Wikipedia, Hacker News, arXiv, or DuckDuckGo), reads the results, and writes a focused answer using only what it found. The Synthesizer reads all three Worker outputs and combines them into a structured markdown response with a one line takeaway at the end.

Everything runs inside one Web Worker. The page itself stays interactive even while the model is loading or generating. Tokens stream back to the page over postMessage and render live.

The codebase is roughly 1500 lines of plain JavaScript. No React. No build step. No npm install. You can clone it, run `npx http-server .`, and it works.

## Why I picked Gemma 4 E2B specifically

I spent some time trying smaller models. SmolLM2 360M loads in under a minute. It would have made the demo much faster. I went with Gemma 4 E2B anyway. Three reasons.

**Reason one. It follows instructions.** The Planner has to write exactly three numbered sub-questions, with no preamble, in the topic the user asked about. SmolLM2 either copied the example from my prompt verbatim, or wrote JSON I had to recover with regex, or drifted off topic. Gemma 4 just does what you ask. Here is its actual Planner output for "How does WebGPU enable in browser AI inference?":

```
1. What is WebGPU and how does it facilitate in-browser AI inference?
2. How is WebGPU utilized in practical examples for deploying AI inference
   within web applications?
3. What are the limitations and trade-offs when using WebGPU for in-browser
   AI inference?
```

Three sub-questions. Three different angles. All anchored to the topic. I did not have to argue with the model.

**Reason two. It refuses to make things up.** This is the property I actually care about most. I tell the Workers in the prompt: do not invent facts. If the research notes do not answer the sub-question, say so directly.

Gemma 4 obeys this. When Wikipedia returns articles that do not address the question, the Worker writes something like "the provided research notes do not contain information about this." It does not paper over the gap with confident sounding nonsense from training data.

SmolLM2 does paper over the gap. It writes plausible content that sounds right and is wrong. For a research demo, that is the difference between something I can show people and something I cannot.

**Reason three. It actually fits in a browser.** Gemma 4 E2B at q4f16 is about 3.1 GB of weights. Chrome can hold that in a Web Worker alongside the ONNX runtime and a streaming UI. The next size up, Gemma 4 E4B, is about 6 GB and starts running into memory pressure. The bigger Gemma 4 variants (26B and 31B) are server class. E2B is the only one that ships in a browser tab today.

E2B also has a 128K context window. I am not using all of it yet, but the headroom is there.

## The seven things that broke

I expected this to take a weekend. It took a week. I want to walk through each problem because the engineering is the actually interesting part of this post.

### 1. The page was not cross origin isolated

GitHub Pages does not let you set custom HTTP headers. Without two specific headers (Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy), the browser will not enable SharedArrayBuffer. The ONNX runtime needs SharedArrayBuffer to run on WebGPU. Without it, every model load failed with raw numeric errors that pointed into the WebAssembly heap (literally numbers like `11514632`) and gave me nothing to debug.

I shipped a service worker (adapted from a small open source library called coi-serviceworker) that intercepts every response and adds the missing headers. The first time a visitor loads the page, the service worker installs and the page reloads itself. After that, the page is cross origin isolated and SharedArrayBuffer becomes available.

If you build something like this on GitHub Pages, you will hit this. Plan for it.

### 2. The tab froze for sixty seconds during model load

The first time the ONNX runtime loads a model on WebGPU, it has to compile shaders for every operation in the network. For Gemma 4 E2B that takes anywhere from thirty seconds to over a minute. If you do this on the main thread, the page becomes unresponsive. Chrome eventually offers to kill the page.

I moved the entire model life cycle into a Web Worker. The page now sees a thin proxy that exposes `model.load(onProgress)` and `model.chat(messages, opts)`. The proxy talks to the worker over postMessage. The worker downloads weights, compiles shaders, runs inference. The page stays responsive throughout.

### 3. The worker flooded the main thread

After moving the model into the worker, the page froze again. Different reason this time. Transformers.js fires a progress callback for every chunk of the download. With a 1.5 GB file, that is hundreds of events per second. I was forwarding each one to the main thread via postMessage. The main thread message queue saturated and the page stopped painting.

The fix is one variable and one comparison. Throttle the progress messages to one every 100 milliseconds. Always send state changes immediately (started, finished, ready) but coalesce the actual progress percentages.

```
let lastProgressPostAt = 0;
function postProgress(payload, force = false) {
  const now = performance.now();
  if (force || now - lastProgressPostAt >= 100) {
    lastProgressPostAt = now;
    post(payload);
  }
}
```

After this change, the page stayed smooth for the entire 3 GB download.

### 4. The browser cache was poisoning itself

When a model load failed mid download (which happened a lot during early debugging), Transformers.js was caching the partial blob. The next page load fetched the corrupt blob from cache and the runtime threw on it. Reloading did not help. Disabling JavaScript caching did not help. The cache was the source of the corruption and it persisted across sessions.

Two fixes. Disable the cache layer entirely on older versions, then upgrade to Transformers.js 4.x where partial downloads do not get committed to the cache in the first place. The 4.x behavior is what I want and what shipped.

### 5. Transformers.js 3.5.1 was just broken

Every model and every backend combination failed with raw numeric errors. SmolLM2 (which works fine elsewhere) failed the same way as Gemma 4. I spent a day thinking my code was wrong before I noticed it was the library version.

Pinned to 4.2.0. Same code, same models, all loaded first try. The diagnostic process was simple: find a known working public project that uses Gemma 4 and copy their version pin. nico-martin's gemma4-browser-extension uses 4.2.0, so I tried that.

### 6. Wikipedia search returned nothing for verbose questions

I was pressure testing the live site after deploying. Even simple queries like "What is photosynthesis?" were producing answers like "research notes are empty." Two of three Workers were giving up.

The bug was upstream. The Planner generates verbose sub-questions like "What is the fundamental definition of photosynthesis and its core chemical processes?" Wikipedia's classic search API does title prefix matching only. That string is not the prefix of any Wikipedia article title, so it returned zero results.

Three changes fixed this:

First, switch to the full text search API instead of the title prefix one. This made every realistic question return some Wikipedia article.

Second, strip filler words from the search query before sending it. Words like "fundamental", "core", "exemplified", "current", "associated" are common across all questions and they hurt search relevance. I built a stopword list of about 60 of these, leaving only the topic words. After this, "photosynthesis chemical processes" reliably ranks the actual Photosynthesis article first.

Third, add a relevance gate. If the search returns three articles but none of their titles contain a meaningful word from the query, treat that as zero results and cascade to a different source. This was the case for niche cross domain queries like "WebGPU vs WebGL for ML inference," where Wikipedia returned articles like "Predictive coding" and "Groq" that share some terminology but do not address the question. The cascade then tries DuckDuckGo, then arXiv.

### 7. Multimodal did not ship

This is the honest one. Gemma 4 E2B is multimodal. You can drop an image into the prompt and the model will describe what it sees. I tried to wire this up. It did not work in the time I had.

The recipe in the model card uses a different API than what I had been using. Instead of the simple `pipeline('text-generation', ...)` call, you have to instantiate `Gemma4ForConditionalGeneration` directly along with `AutoProcessor`, and configure different precision settings for each component (the audio encoder and vision encoder need fp16, the decoder and embeddings can stay at q4f16).

I implemented all of this. The load kept falling through to SmolLM2 instead. There were two bugs in the path. The processor signature has four arguments and I was passing three (so the options object got interpreted as audio input). And the q4f16 audio encoder weights were throwing on some Chrome drivers.

Both are fixable. I did not get them fixed in time. The multimodal code is still in the repository under `loadGemma4` and `chatViaGemma` in the worker file. The current production demo is text only. I will pick this up after the challenge.

## What I want to be honest about

A few things people might infer from the demo that are not actually true:

The agents are not running in parallel. WebGPU LLM inference is single stream per session. The five agents take turns on one GPU. The visualization with the connecting lines and flowing particles makes it look parallel but logically. The actual GPU work serializes.

The synthesis is good but not amazing. Multi agent works best when the workers research genuinely different sources and bring back distinct evidence. Mine often all land on Wikipedia and read variants of the same article. So the synthesis ends up being a nicely structured restatement of the workers, not something that transcends them. The way to fix this is worker specialization (different prompts for different roles) and Gemma 4's native tool calling (which I have wired but did not ship). Future work.

Niche cross domain queries return "no findings." This is the relevance gate doing its job. If you ask about a topic that does not have a Wikipedia article and is not in arXiv either, all the workers honestly report the gap. That is correct behavior for a demo about hallucination resistance, but it is also a real UX limit.

The 3 GB download is a wall. There is no way around it for first time visitors. I gate the download behind an explicit click and persist via the browser cache so reload is free. There is also a lite mode at `?model=smollm` which loads SmolLM2 360M (about 270 MB) for visitors who want to see the swarm UX without the commitment. The output quality is noticeably lower in lite mode, because SmolLM2 confabulates. The architecture is identical.

## Try it

Default with Gemma 4 (3.1 GB first load): https://shopsmartai.github.io/agentmesh/

Lite mode with SmolLM2 (270 MB): https://shopsmartai.github.io/agentmesh/?model=smollm

Source code: https://github.com/shopsmartai/agentmesh

These are good first queries because Wikipedia covers them well:

- What is photosynthesis?
- What is the difference between Mixture of Experts and Dense models?
- How do enzymes work?

If you ask something niche like "latest 2026 transformer papers" you will hit the honest "no findings" path. That is intentional, not a bug.

## What I would build next

Four things, in priority order, if I keep working on this past the challenge.

Native Gemma 4 tool calling. The model has a structured tool call format built into its chat template. Right now I use a simple heuristic to pick a tool. Switching to the native format would make tool selection more accurate and add another concrete reason the project specifically uses Gemma 4 (other models do not have this).

Multimodal, properly. Resume the work I rolled back. Drop an image into the query. The Planner sees it. The Workers research what is in it. The Synthesizer ties it together with text findings. The code is in the repo and the failure modes are known.

Worker specialization. Right now all three workers use the same prompt template. Differentiating them ("you focus on definitions", "you find counter evidence", "you look for examples") would actually use the multi agent shape. Today, the swarm is more theater than function for many queries.

A persistent local research notebook. Save past queries. Build context across sessions. Make the 3 GB download amortize across many uses instead of being a one shot cost. This is the path that turns the project from a demo into something useful for real research work.

I am not committing to any of these. What I do next depends on the response to this post. If a hospital CTO emails me about clinical literature review, I will go faster on the notebook. If five engineers fork the repo, I will go faster on the tool calling and multimodal pieces. If neither, this stays a portfolio piece I am happy with.

## Thanks

To Google DeepMind for releasing Gemma 4 with weights small enough to actually fit in a browser tab. To Hugging Face for Transformers.js, especially the 4.2.0 release that finally worked end to end. To gzuidhof for the coi-serviceworker library which solved the cross origin isolation problem in about five minutes. To nico-martin for publishing the gemma4-browser-extension code, which was my reference for which library version actually loads this model. To the webml-community Gemma 4 demo whose minified bundle taught me the per component dtype trick I tried to ship and rolled back.

The code is MIT. Fork it. Rip out the parts you want. If you build something better with it, I would love to see what.
