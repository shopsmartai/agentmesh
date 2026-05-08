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

export async function searchWikipedia(query, { limit = 3 } = {}) {
  // Step 1: search for matching pages
  const searchUrl = `https://en.wikipedia.org/w/api.php?action=opensearch&format=json&origin=*&limit=${limit}&search=${encodeURIComponent(query)}`;
  const searchRes = await fetchWithTimeout(searchUrl);
  if (!searchRes.ok) throw new Error('wikipedia search failed');
  const [, titles, descriptions, urls] = await searchRes.json();

  if (!titles?.length) {
    return { source: 'wikipedia', query, results: [] };
  }

  // Step 2: fetch summaries for top matches
  const summaries = await Promise.all(
    titles.slice(0, limit).map(async (title, i) => {
      try {
        const sumUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
        const sumRes = await fetchWithTimeout(sumUrl);
        if (!sumRes.ok) throw new Error('summary fetch failed');
        const data = await sumRes.json();
        return {
          title: data.title || title,
          extract: data.extract || descriptions[i] || '',
          url: data.content_urls?.desktop?.page || urls[i],
        };
      } catch (err) {
        return { title, extract: descriptions[i] || '', url: urls[i] };
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
    description: 'Lookup an instant answer or related topics from DuckDuckGo.',
    args: ['query'],
    run: searchDuckDuckGo,
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
