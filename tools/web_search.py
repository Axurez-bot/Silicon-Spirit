"""
tools/web_search.py
DuckDuckGo web search — no API key, fully local.
Returns top N results as clean text Spirit can reason over.
"""

import re
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Hits DuckDuckGo HTML search endpoint and parses results.
    Returns list of {title, url, snippet}.
    """
    results = []
    try:
        url  = "https://html.duckduckgo.com/html/"
        resp = requests.post(
            url,
            data={"q": query, "b": "", "kl": "us-en"},
            headers=HEADERS,
            timeout=10
        )
        resp.raise_for_status()
        html = resp.text

        # Parse result blocks — DDG HTML is stable enough to regex
        blocks = re.findall(
            r'class="result__title".*?href="(.*?)".*?>(.*?)</a>.*?'
            r'class="result__snippet".*?>(.*?)</span>',
            html, re.DOTALL
        )

        for url_raw, title_raw, snippet_raw in blocks[:max_results]:
            title   = re.sub(r"<[^>]+>", "", title_raw).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()
            # DDG wraps URLs — decode redirect
            real_url = re.sub(r"//duckduckgo\.com/l/\?uddg=", "", url_raw)
            real_url = requests.utils.unquote(real_url).split("&rut=")[0]
            if title and snippet:
                results.append({"title": title, "url": real_url, "snippet": snippet})

    except Exception as e:
        results.append({"title": "Search Error", "url": "", "snippet": str(e)})

    return results


def web_search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo. No API key required.
    Use this for: current events, facts Spirit doesn't know, anything that needs live data.
    Input: a search query string.
    Output: top search results with titles, URLs, and snippets.
    """
    results = _ddg_search(query, max_results=5)

    if not results:
        return "No results found for that query."

    lines = [f"Web search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['snippet']}")
        lines.append("")

    return "\n".join(lines)
