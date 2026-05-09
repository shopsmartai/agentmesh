<!--
Cover image: assets/og-cover.png
Tags: devchallenge, gemmachallenge, gemma, webgpu
Category: "Build With Gemma 4"
Title (under 70 chars): "I Built an AI That Argues With Itself About Anything"
-->

# I Built an AI That Argues With Itself About Anything

Live demo: https://shopsmartai.github.io/agentmesh/

Open in any modern browser. There are two ways to run it. Either let it download Gemma 4 E2B onto your GPU and run everything locally (3 GB first load, then cached), or paste in a free Google AI Studio key and run on the bigger Gemma 4 26B-A4B in the cloud (~30 seconds per query, no download). Same architecture, two different shapes of trade-off. The whole thing is open source under MIT: https://github.com/shopsmartai/agentmesh

I asked it: "Is the AI bubble about to burst?"

What came back was not a single balanced answer. It was three agents arguing. A skeptic listed real failure modes: compute economics, valuations detached from revenue, parallels to the 2000 dot-com crash. An advocate listed concrete benefits: productivity gains in coding and writing, durable infrastructure demand, real categories that will outlast the hype. A pragmatist described how it actually plays out: some companies will pop, others will thrive, and which side you land on depends on whether the company is application or infrastructure. Then a synthesizer showed me where the three agreed, where they disagreed, and reconciled.

That is the whole point of this project. Single-shot LLMs collapse to one voice. They average perspectives behind the scenes and hide the disagreement. AgentMesh forces the disagreement to actually surface, with three agents that each have a fixed stance, and a synthesizer that calls out where they parted ways.

This post is about why I built it that way and the engineering it took to get there. There is more of the engineering than I expected.

## Why three agents instead of one

Try this in your head. Ask any chat model "Is buying a Tesla a good investment in 2026?" You will get a balanced essay. The model will give you one paragraph for each side and a polite ending. It is fine. It is also boring. It is hiding what the model actually thinks each side's strongest argument is.

A multi-agent system can do something a single call cannot. You give each agent a fixed role and a separate context. The skeptic does not get to soften its view by also writing the advocate's paragraph. The advocate does not get to hedge. They have to commit. Only then do you reconcile.

This is the only architecture I know of where the multi-agent shape gives a concrete win that single-call inference cannot match. It is not "five agents read the same Wikipedia article and produce variants." It is three agents holding three positions and a synthesizer showing the gap.

The architecture also makes the answer better in a small way that matters. Each agent searches Wikipedia AND Hacker News comments in parallel for evidence supporting its stance. Wikipedia gives encyclopedic facts. HN comments give substantive user opinions and lived experience. Workers see both, anchor in the parts that support their stance, and ground their argument in real material rather than hallucinating.

## Why Gemma 4 specifically

This architecture only works if the model has three properties at once. Gemma 4 ships all three across two scales: E2B for browser deployment, 26B-A4B and 31B for server-class inference. That is why the same architecture can run locally or in the cloud.

**It actually stays in role.** Smaller models drift. You tell SmolLM2-360M "you are the skeptic, find counter-arguments" and it writes a balanced paragraph anyway because the underlying instinct is to be helpful. Gemma 4 stays sharp. The skeptic produces criticisms. The advocate produces upside. The roles do not blur.

**It is honest about its evidence.** Every agent is told: anchor in research notes when they support the stance, but use general knowledge to construct a stronger argument when notes are thin. Be honest about which claims have evidence vs which are widely-held belief. Gemma 4 obeys this. The disagreement only matters if it is grounded; Gemma 4 keeps it grounded.

**It comes in browser size and server size.** Gemma 4 E2B at q4f16 is about 3.1 GB of weights. Chrome can hold that in a Web Worker alongside the ONNX runtime and a streaming UI. The next sizes up (26B-A4B Mixture-of-Experts and 31B Dense) are server-class and Google hosts them on AI Studio. So the local and cloud paths produce comparable shape of answers without rearchitecting.

E2B and the larger variants both expose a 128K context window. I am not using all of it yet, but the headroom means future versions can feed full documents to the perspective agents instead of search snippets.

## The two modes

Local mode runs Gemma 4 E2B on your GPU. First page load downloads about 3.1 GB of weights into the browser cache. After that, reload is instant. Inference takes 3 to 5 minutes per query on a consumer GPU because WebGPU LLM inference is single-stream and we have five sequential model calls per swarm. Privacy is total: nothing leaves the tab once weights are cached.

Cloud mode runs Gemma 4 26B-A4B (or 31B if you pick it) on Google's servers via the Gemini API. You bring your own free key from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). The key lives only in your browser's localStorage; it is never embedded in the public code, never transmitted to anyone except Google. Inference takes 20 to 40 seconds per query. The privacy story shifts: queries go through Google.

You pick. The Settings panel has a radio toggle and a key input. The header tells you which mode is active.

## The eight engineering problems

I expected this to take a weekend. It took two weeks. Naming the problems is the actual point of this post because most of them are reusable knowledge for anyone trying to ship browser-LLM apps.

### 1. The page was not cross-origin isolated

GitHub Pages does not let you set custom HTTP headers. Without two specific headers (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless`), the browser will not enable `SharedArrayBuffer`. The ONNX runtime needs `SharedArrayBuffer` to run on WebGPU. Without it, every model load failed with raw numeric errors that pointed into the WebAssembly heap (literal numbers like `11514632`).

Fix: ship a service worker (adapted from coi-serviceworker, MIT) that intercepts every response and adds the missing headers. First load registers the worker and reloads. After that, the page is cross-origin isolated and `SharedArrayBuffer` becomes available.

### 2. WebGPU compile froze the tab for a minute

The first time the ONNX runtime loads a model on WebGPU, it has to compile shaders for every operation in the network. For Gemma 4 E2B that takes anywhere from thirty seconds to over a minute. Done on the main thread, the page becomes unresponsive.

Fix: put the entire model life cycle in a Web Worker. The page sees a thin proxy that exposes `model.load(onProgress)` and `model.chat(messages, opts)`. The proxy talks to the worker over `postMessage`. Heavy work happens off the main thread; the page stays interactive.

### 3. The worker flooded the main thread anyway

After moving the model into the worker, the page froze again for a different reason. Transformers.js fires a progress callback for every chunk of the download. With a 1.5 GB file, that is hundreds of events per second. Each one became a `postMessage` to the main thread. The main thread message queue saturated.

Fix: throttle progress messages to one every 100 milliseconds. Always send state changes immediately (started, finished, ready) but coalesce the actual progress percentages.

### 4. Wikipedia search returned nothing for verbose questions

Pressure-testing the live site exposed a fundamental issue. Even simple queries like "What is photosynthesis?" were producing answers like "research notes are empty." Two of three workers were giving up.

The bug was upstream. The model-driven planner was generating verbose sub-questions like "What is the fundamental definition of photosynthesis and its core chemical processes?" Wikipedia's classic search API does title-prefix matching, and that string is not the prefix of any Wikipedia article title.

Fix in three parts. Switch to Wikipedia's full-text search API. Strip filler words from the search query before sending it (a list of about 60 question-shape filler words). Add a relevance gate: if no result title contains a meaningful word from the query, treat it as zero results and cascade to a different source.

### 5. The planner was spending 30 seconds on something a regex could do

The original planner used the model itself to extract the topic and generate three search queries. It took about 30 seconds on Gemma 4. Then I noticed the model was just echoing the user's full question into the template. It was not actually doing extraction. It was a slow templating engine.

Fix: replace the planner's model call with a small JavaScript function. Strip leading question words and trailing punctuation. Strip "X better than Y" comparisons to keep just X. Cap at 3 words. Done in milliseconds. The planner card still appears in the swarm visualization; it just shows the topic immediately rather than streaming through 30 seconds of model generation.

### 6. Wikipedia gave facts but no opinions

For opinion questions ("Is remote work better than in-office work?"), Wikipedia gave the encyclopedic article on remote work, which is descriptive, not opinionated. Workers correctly noticed they could not construct a real critique from descriptive material and refused.

Fix: search multiple sources in parallel. For every worker query, fetch Wikipedia AND Hacker News comments at the same time. Wikipedia provides encyclopedic grounding. HN comments provide actual user opinions and lived experience, with paragraphs of real takes that no encyclopedia has. Workers see both and decide what to anchor on. The "0 servers contacted" line in the footer lost its third significant digit but the architecture finally produces grounded opinion content.

### 7. Cloud mode was the next thing the project needed

Local Gemma 4 E2B at 3 to 5 minutes per query is real and unavoidable on consumer GPUs. The architecture only worked at "single user with patience" volume. Adding a cloud option mattered because it lets the same architecture run at 30 seconds per query on bigger Gemma 4 variants.

The challenge was doing it without compromising the privacy story for users who care about it. Solution: BYOK. The Settings panel has a key input that writes only to localStorage. The public code contains no key (verified with `git grep` for `AIza` patterns before every push). The cloud path makes one fetch per agent call directly to `generativelanguage.googleapis.com` from the user's browser. We never see the key, never proxy the requests, never log anything.

The cloud adapter has a 60-second per-call timeout (we observed one worker hang indefinitely while the other two completed) and a fallback chain across three Gemma 4 variants so a transient 5xx on one does not kill the swarm.

### 8. Gemma 4 dumps its thinking into the response

Even with `thinkingConfig.thinkingBudget=0` (which the API rejects for Gemma 4 with a 400, by the way), Gemma 4 on the Gemini API outputs its internal reasoning verbatim: drafts, refinement passes, "Word count check" notes, before the actual final answer. The user's first cloud query produced about 600 words of thinking-out-loud before the clean structured answer.

Fix: client-side stripping. The synthesizer's output always starts with `## Where they agreed`. Find the LAST occurrence of that heading, trim everything before it. Strip leading "Final Polish:" / "Refined version:" labels. The model's thinking goes in the bin; the user sees only the polished output.

## What I want to be honest about

A few things people might assume from the demo that are not actually true.

**Local mode is sequential, not parallel.** WebGPU LLM inference is single-stream per session. The three perspective agents take turns on one GPU. The visualization makes it look concurrent. Logically it is, physically it is not. Cloud mode actually fires three parallel Gemini calls but the synthesizer still waits for all three.

**Cloud mode is private to Google, not to me.** Your queries go to Google's servers. Your key goes from your browser directly to Google. I never see either. But this is different from the local mode promise of "nothing leaves your tab." Both are valid trade-offs; you pick.

**Niche cross-domain queries return weak findings.** When the public sources do not have an opinion to mine, the agents lean on training knowledge alone. They are honest about which claims have evidence versus widely-held belief.

**The 3 GB local-mode download is real.** No way around it for first-time local visitors. We gate the download behind an explicit click and persist via the browser cache so reload is free. Cloud mode is the alternative for visitors who do not want to commit.

**Factual lookup queries do not benefit from this architecture.** If you ask "What is photosynthesis?" the skeptic does not really argue against photosynthesis. Three perspective agents on a factual lookup just produce three slightly different summaries of the same article. AgentMesh shines on questions with actual disagreement.

## Try it

Live: https://shopsmartai.github.io/agentmesh/

Source: https://github.com/shopsmartai/agentmesh

The site has a Settings panel reachable from both the boot screen and the main screen. Pick local or cloud. For cloud, get a free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and paste it in.

Three suggested questions are wired to suggestion buttons on the page:

- *Is buying a Tesla a good investment in 2026?*
- *Is the AI bubble about to burst?*
- *Is buying a house better than renting in 2026?*

Or type your own. Anything with real disagreement to mine works well.

## What I would build next

If I keep working on this past the challenge, four things in priority order.

**Native Gemma 4 tool calling.** The model has a structured tool-call format built into its chat template. Right now I use a deterministic dispatch (Wikipedia + HN comments in parallel). Switching to the native format would make tool selection cleaner and add another concrete reason the project specifically uses Gemma 4.

**Multimodal, properly.** Gemma 4 E2B is multimodal. The code paths for image input exist in `model.worker.js` but I rolled them back because per-component dtype configuration on diverse Chrome drivers needed more debugging than the challenge time-box allowed. Drop an image into the prompt. The skeptic argues against what is in it. The advocate argues for. The pragmatist describes how it gets used.

**More than three perspectives.** A historical agent (how did this work in the past?). A futurist agent (where is this going?). A user-of-the-thing agent. The architecture trivially extends.

**More sources.** Reddit JSON, Brave Search (would need a second BYOK key), Stack Exchange. Each broadens the grounding for opinionated queries.

I am not committing to any of these yet. What I do next depends on what happens with this post. If a hospital CTO emails me about clinical literature review with locally-bound data, I will go faster on something like a persistent OPFS notebook. If five engineers fork the repo, I will go faster on multimodal and tool-calling. If neither, this stays a portfolio piece I am happy with.

## Thanks

To Google DeepMind for releasing Gemma 4 with weights small enough to actually fit in a browser tab AND making the bigger variants available on AI Studio for free, so the same project can run at two scales. To Hugging Face for Transformers.js, especially the 4.2.0 release that finally worked end to end. To gzuidhof for the coi-serviceworker library, which solved the cross-origin isolation problem in about five minutes once I knew it existed. To nico-martin for publishing the gemma4-browser-extension code, which was my reference for which library version actually loads this model. To everyone who keeps the open browser-ML stack moving forward.

The code is MIT. Fork it. Rip out the parts you want. If you build something better with it, I would love to see what.
