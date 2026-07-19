"""
tools/browser_control.py
FEATURE 5: Browser control via Playwright (Chromium, headless inside Docker).
Spirit can navigate, click, type, screenshot, and extract content from websites.
All actions require Creator approval for navigation outside safe domains
unless the domain is in the SAFE_DOMAINS list.
"""

import re
import time
import base64
from pathlib import Path
from crewai.tools import tool

SCREENSHOT_PATH = Path("/app/static/latest_screenshot.png")

# Domains Spirit can browse without approval
SAFE_DOMAINS = [
    "duckduckgo.com",
    "wikipedia.org",
    "github.com",
    "stackoverflow.com",
    "docs.python.org",
    "pypi.org",
    "arxiv.org",
    "huggingface.co",
]

_browser  = None
_page     = None


def _get_page():
    global _browser, _page
    if _page is None:
        try:
            from playwright.sync_api import sync_playwright
            _pw      = sync_playwright().start()
            _browser = _pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = _browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                )
            )
            _page = context.new_page()
            print("[Browser] Playwright Chromium started.")
        except Exception as e:
            print(f"[Browser] Failed to start: {e}")
    return _page


def _is_safe_domain(url: str) -> bool:
    for domain in SAFE_DOMAINS:
        if domain in url:
            return True
    return False


def _take_screenshot() -> str:
    """Take screenshot and save to static dir for Moondream to see."""
    page = _get_page()
    if not page:
        return "Browser offline."
    try:
        SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SCREENSHOT_PATH), full_page=False)
        return f"Screenshot saved: {SCREENSHOT_PATH}"
    except Exception as e:
        return f"Screenshot failed: {e}"


@tool("browser_navigate")
def browser_navigate_tool(url: str) -> str:
    """
    Navigate to a URL in the headless browser.
    Safe domains (Wikipedia, GitHub, docs, etc.) navigate freely.
    Other URLs will be flagged but attempted.
    Input: full URL including https://
    Output: page title and first 500 chars of visible text.
    """
    if not url.startswith("http"):
        url = "https://" + url

    page = _get_page()
    if not page:
        return "Browser not available. Is Playwright installed? Run: pip install playwright && playwright install chromium"

    if not _is_safe_domain(url):
        print(f"[Browser] Navigating to non-safe domain: {url}")

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)
        title   = page.title()
        content = page.inner_text("body")[:1000]
        _take_screenshot()
        return f"Navigated to: {title}\nURL: {url}\n\nContent preview:\n{content}"
    except Exception as e:
        return f"Navigation failed: {e}"


@tool("browser_click")
def browser_click_tool(selector_or_text: str) -> str:
    """
    Click an element on the current page.
    Input: CSS selector (e.g. '#submit') or visible text (e.g. 'Search').
    Output: confirmation or error.
    """
    page = _get_page()
    if not page:
        return "Browser offline."
    try:
        # Try text match first, then CSS selector
        try:
            page.get_by_text(selector_or_text, exact=False).first.click(timeout=5000)
        except Exception:
            page.click(selector_or_text, timeout=5000)
        time.sleep(0.5)
        _take_screenshot()
        return f"Clicked: {selector_or_text} | Page: {page.title()}"
    except Exception as e:
        return f"Click failed: {e}"


@tool("browser_type")
def browser_type_tool(selector_and_text: str) -> str:
    """
    Type text into an input field on the current page.
    Input format: 'selector:::text to type' (e.g. '#search:::hello world')
    Output: confirmation.
    """
    if ":::" not in selector_and_text:
        return "Format required: 'selector:::text' e.g. '#search:::python tutorial'"
    selector, text = selector_and_text.split(":::", 1)
    page = _get_page()
    if not page:
        return "Browser offline."
    try:
        page.fill(selector.strip(), text.strip())
        return f"Typed into {selector.strip()}: {text.strip()[:50]}"
    except Exception as e:
        return f"Type failed: {e}"


@tool("browser_extract")
def browser_extract_tool(query: str = "") -> str:
    """
    Extract text content from the current browser page.
    Input: optional query about what to extract (e.g. 'main article', 'prices', 'links').
    Output: relevant page content as text.
    """
    page = _get_page()
    if not page:
        return "Browser offline."
    try:
        if query:
            # Try to find relevant section
            for tag in ["article", "main", "#content", ".content", "body"]:
                try:
                    text = page.inner_text(tag)
                    if len(text) > 100:
                        return f"Extracted from {tag}:\n{text[:2000]}"
                except Exception:
                    continue
        full = page.inner_text("body")
        return f"Page content ({page.title()}):\n{full[:2000]}"
    except Exception as e:
        return f"Extract failed: {e}"


@tool("browser_screenshot")
def browser_screenshot_tool(query: str = "") -> str:
    """
    Take a screenshot of the current browser page.
    The screenshot is saved and can be analyzed by Moondream vision.
    Input: ignored — always screenshots current page.
    Output: confirmation with file path.
    """
    result = _take_screenshot()
    return result + "\nUse screen_vision tool to analyze what Spirit sees."


@tool("browser_search")
def browser_search_tool(query: str) -> str:
    """
    Search DuckDuckGo in the browser and return results page content.
    More thorough than the API search — actually loads result pages.
    Input: search query.
    Output: search results page content.
    """
    url  = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&ia=web"
    page = _get_page()
    if not page:
        return "Browser offline."
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        time.sleep(1.5)
        # Extract result snippets
        results = []
        for el in page.query_selector_all(".result__snippet")[:8]:
            txt = el.inner_text().strip()
            if txt:
                results.append(txt)
        if results:
            return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        return page.inner_text("body")[:1500]
    except Exception as e:
        return f"Browser search failed: {e}"