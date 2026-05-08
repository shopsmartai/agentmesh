// ============================================
// AGENTMESH // Tool registry
// All endpoints are CORS-friendly and require no API keys.
// ============================================

const FETCH_TIMEOUT = 8000;

async function fetchWithTimeout(url, opts = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT);
  try {
    const res = await fetch(url, { ...opts, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(timer);
  }
}

// ============================================
// WIKIPEDIA
// ============================================

// Strip stopwords + filler so a verbose sub-question becomes a
// search-friendly phrase. Wikipedia's srsearch ranks via TF-IDF, so
// generic question-shaped filler ("fundamental", "core", "current",
// "various") dilutes the topic signal and pulls off-topic articles to
// the top. Aggressive filtering makes srsearch return the topic article
// reliably.
const WIKI_STOPWORDS = new Set([
  // grammar
  'what','is','are','was','were','the','a','an','of','to','for','and','or','in','on',
  'at','by','with','from','about','between','do','does','did','how','why','when','where',
  'which','who','whom','whose','this','that','these','those','can','should','would','could',
  'may','might','will','shall','have','has','had','its','it','as','than','then','some',
  'any','all','most','more','less','few','many','their','your','my','our','us','you','i',
  'me','they','them','if','so','also','such',
  // generic question-shape filler that question-style sub-questions love
  'define','definition','describe','description','explain','explanation','provide','provided',
  'tell','give','given','core','main','key','primary','secondary','fundamental','basic',
  'basics','simple','complex','various','different','common','typical','general','specific',
  'particular','associated','related','involved','involving','utilized','utilization','usage',
  'used','using','use','employed','exemplified','exemplifies','example','examples','exemplary',
  'instance','instances','case','cases','context','contexts','contextual','aspect','aspects',
  'feature','features','element','elements','process','processes','processing','procedure',
  'procedures','step','steps','part','parts','component','components',
  // generic "limitation/tradeoff" filler
  'limit','limits','limitation','limitations','tradeoff','tradeoffs','trade-offs','trade-off',
  'constraint','constraints','consideration','considerations','factor','factors','requirement',
  'requirements','condition','conditions','environmental','environment','impact','impacts',
  'effect','effects','consequence','consequences',
  // generic adjective-y filler
  'current','recent','modern','advanced','enhanced','improved','enable','enables','enabled',
  'enabling','support','supports','supported','supporting','contains','containing','include',
  'includes','included',
]);

function condenseWikiQuery(q) {
  if (!q) return '';
  const words = q
    .toLowerCase()
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !WIKI_STOPWORDS.has(w));
  if (words.length === 0) return '';

  // Wikipedia's srsearch ranks by TF-IDF: rare words dominate. Long words
  // tend to be the topical nouns (e.g. "photosynthesis", "transformer",
  // "WebGPU"); short common words ("core", "process") dilute the signal
  // and pull off-topic articles to the top.
  // Strategy: keep the 4 longest content words while preserving order.
  const sortedByLength = [...words].sort((a, b) => b.length - a.length);
  const keep = new Set(sortedByLength.slice(0, 4));
  return words.filter((w) => keep.has(w)).join(' ');
}

export async function searchWikipedia(query, { limit = 3 } = {}) {
  // Step 1: full-text search via the action API. srsearch matches article
  // *content*, not just titles, so verbose sub-questions actually return
  // hits. opensearch (the prior implementation) was title-prefix only.
  const condensed = condenseWikiQuery(query) || query;
  const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&origin=*&srlimit=${limit}&srsearch=${encodeURIComponent(condensed)}`;
  const searchRes = await fetchWithTimeout(searchUrl);
  if (!searchRes.ok) throw new Error('wikipedia search failed');
  const data = await searchRes.json();
  const hits = data?.query?.search || [];

  if (!hits.length) {
    return { source: 'wikipedia', query, results: [] };
  }

  // Step 2: fetch readable summaries for the top matches via the REST
  // summary endpoint (cleaner extract than srsearch's snippet).
  const summaries = await Promise.all(
    hits.slice(0, limit).map(async (hit) => {
      const title = hit.title;
      try {
        const sumUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
        const sumRes = await fetchWithTimeout(sumUrl);
        if (!sumRes.ok) throw new Error('summary fetch failed');
        const sum = await sumRes.json();
        return {
          title: sum.title || title,
          // Strip srsearch HTML highlight tags from the snippet fallback.
          extract: sum.extract || (hit.snippet || '').replace(/<[^>]+>/g, ''),
          url: sum.content_urls?.desktop?.page || `https://en.wikipedia.org/wiki/${encodeURIComponent(title)}`,
        };
      } catch {
        return {
          title,
          extract: (hit.snippet || '').replace(/<[^>]+>/g, ''),
          url: `https://en.wikipedia.org/wiki/${encodeURIComponent(title)}`,
        };
      }
    })
  );

  return { source: 'wikipedia', query, results: summaries };
}

// ============================================
// HACKER NEWS (Algolia)
// ============================================

export async function searchHackerNews(query, { limit = 5 } = {}) {
  const url = `https://hn.algolia.com/api/v1/search?query=${encodeURIComponent(query)}&hitsPerPage=${limit}&tags=story`;
  const res = await fetchWithTimeout(url);
  if (!res.ok) throw new Error('hackernews search failed');
  const data = await res.json();

  return {
    source: 'hackernews',
    query,
    results: (data.hits || []).map((hit) => ({
      title: hit.title || '',
      url: hit.url || `https://news.ycombinator.com/item?id=${hit.objectID}`,
      points: hit.points || 0,
      author: hit.author || '',
      comments: hit.num_comments || 0,
      date: hit.created_at || '',
    })),
  };
}

// ============================================
// DUCKDUCKGO Instant Answer
// (Free, CORS-friendly — limited but useful for entity lookups)
// ============================================

export async function searchDuckDuckGo(query) {
  const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_redirect=1&no_html=1&skip_disambig=1`;
  const res = await fetchWithTimeout(url);
  if (!res.ok) throw new Error('duckduckgo search failed');
  const data = await res.json();

  const results = [];
  if (data.AbstractText) {
    results.push({
      title: data.Heading || query,
      extract: data.AbstractText,
      url: data.AbstractURL || '',
    });
  }
  if (Array.isArray(data.RelatedTopics)) {
    for (const topic of data.RelatedTopics.slice(0, 4)) {
      if (topic.Text) {
        results.push({
          title: (topic.Text.split(' - ')[0] || '').trim(),
          extract: topic.Text,
          url: topic.FirstURL || '',
        });
      }
    }
  }

  return { source: 'duckduckgo', query, results };
}

// ============================================
// ARXIV (free Atom-feed API, no key, CORS-friendly)
// ============================================

export async function searchArxiv(query, { limit = 4 } = {}) {
  const url = `https://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&start=0&max_results=${limit}`;
  const res = await fetchWithTimeout(url);
  if (!res.ok) throw new Error('arxiv search failed');
  const xml = await res.text();

  // Light XML parse — Atom feed entries
  const entries = [];
  const entryRegex = /<entry>([\s\S]*?)<\/entry>/g;
  let m;
  while ((m = entryRegex.exec(xml)) !== null) {
    const block = m[1];
    const pick = (tag) => {
      const t = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\/${tag}>`).exec(block);
      return t ? t[1].replace(/\s+/g, ' ').trim() : '';
    };
    const title = pick('title');
    const summary = pick('summary');
    const link = (/<id>([\s\S]*?)<\/id>/.exec(block) || [])[1] || '';
    if (title && summary) {
      entries.push({
        title,
        extract: summary.length > 320 ? summary.slice(0, 320) + '…' : summary,
        url: link.trim(),
      });
    }
    if (entries.length >= limit) break;
  }

  return { source: 'arxiv', query, results: entries };
}

// ============================================
// TOOL REGISTRY (used by agents)
// ============================================

export const TOOLS = {
  wikipedia: {
    name: 'wikipedia',
    description: 'Search Wikipedia for an entity, topic, or concept. Returns 3 article summaries.',
    args: ['query'],
    run: searchWikipedia,
  },
  hackernews: {
    name: 'hackernews',
    description: 'Search Hacker News for recent discussions and links on a topic.',
    args: ['query'],
    run: searchHackerNews,
  },
  duckduckgo: {
    name: 'duckduckgo',
    description: 'Lookup an instant answer or related topics from DuckDuckGo. Useful as a generic-knowledge fallback when Wikipedia returns nothing.',
    args: ['query'],
    run: searchDuckDuckGo,
  },
  arxiv: {
    name: 'arxiv',
    description: 'Search arXiv for academic papers and preprints on technical/scientific topics. Returns paper titles, abstracts, and links.',
    args: ['query'],
    run: searchArxiv,
  },
};

/**
 * Run a tool by name with timing + error handling.
 */
export async function runTool(name, query) {
  const tool = TOOLS[name];
  if (!tool) throw new Error(`unknown tool: ${name}`);
  const start = performance.now();
  try {
    const result = await tool.run(query);
    const ms = Math.round(performance.now() - start);
    return { ok: true, tool: name, ms, ...result };
  } catch (err) {
    const ms = Math.round(performance.now() - start);
    return { ok: false, tool: name, ms, error: err.message };
  }
}

/**
 * Format tool results into a compact text block the model can read.
 */
export function formatResultsForModel(toolResult) {
  if (!toolResult.ok) {
    return `[${toolResult.tool} error: ${toolResult.error}]`;
  }
  const { source, results } = toolResult;
  if (!results?.length) {
    return `[${source}: no results]`;
  }
  const lines = [`[source: ${source}]`];
  for (const r of results) {
    if (r.extract) {
      lines.push(`- ${r.title}: ${r.extract.slice(0, 280)}`);
    } else if (r.title) {
      const meta = r.points ? ` (${r.points} pts, ${r.comments} comments)` : '';
      lines.push(`- ${r.title}${meta}`);
    }
  }
  return lines.join('\n');
}
