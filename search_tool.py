"""Search Tool - Real-Time Web Search for RAG (Retrieval-Augmented Generation)."""
import importlib
import logging
import os
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger("search_tool")

# Try to import official package first, then legacy package for compatibility.
DDGS = None
DDGS_AVAILABLE = False

for module_name in ("ddgs", "duckduckgo_search"):
    try:
        module = importlib.import_module(module_name)
        DDGS = getattr(module, "DDGS", None)
        if DDGS is not None:
            DDGS_AVAILABLE = True
            if module_name == "duckduckgo_search":
                logger.warning("Using legacy package duckduckgo-search. Consider installing ddgs.")
            break
    except ImportError:
        continue

if not DDGS_AVAILABLE:
    logger.warning("ddgs/duckduckgo-search not installed. Web search will be disabled.")

# Try to import Tavily client.
TAVILY_AVAILABLE = False
TavilyClient = None

try:
    from tavily import TavilyClient as _TavilyClient
    TavilyClient = _TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    logger.info("tavily-python not installed. Tavily search will be unavailable.")


def _normalize_text(text: str) -> str:
    """Normalize text for light-weight relevance matching."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _extract_query_terms(query: str) -> List[str]:
    """Extract informative terms from the query for relevance scoring."""
    stop_words = {
        "la", "là", "the", "is", "are", "what", "how", "when", "where", "who",
        "cua", "của", "va", "và", "cho", "voi", "với", "mot", "một", "nhung", "những",
        "toi", "tôi", "ban", "bạn", "gia", "giá", "hom", "hôm", "nay",
        "thong", "thông", "tin", "du", "dự", "an", "án", "ve", "về",
        "project", "info", "information", "latest", "update", "updates", "news", "about"
    }
    terms = re.findall(r"[\w\-]{2,}", _normalize_text(query))
    return [t for t in terms if t not in stop_words]


def _required_specific_terms(query_terms: List[str]) -> List[str]:
    """Select specific terms that should appear to avoid broad-topic noise."""
    generic = {
        "viet", "vietnam", "nam", "today", "current", "official", "source"
    }
    specific = [t for t in query_terms if len(t) >= 5 and t not in generic]
    return specific[:3]


def _detect_language(query: str) -> str:
    """Detect coarse query language for search tuning (vi/en)."""
    q = _normalize_text(query)
    # Vietnamese diacritics
    if re.search(r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", q):
        return "vi"

    vi_markers = {
        "thong", "tin", "du", "an", "gia", "xang", "dau", "hom", "nay", "viet", "nam"
    }
    q_terms = set(re.findall(r"[\w\-]{2,}", q))
    if len(vi_markers.intersection(q_terms)) >= 2:
        return "vi"

    return "en"


def _build_search_context(query: str) -> Tuple[str, List[str], bool]:
    """Build region-specific query variants and realtime intent flag."""
    normalized_query = _normalize_text(query)
    language = _detect_language(query)

    realtime_terms = {
        "vi": ["hôm nay", "hom nay", "hiện tại", "hien tai", "mới nhất", "cap nhat"],
        "en": ["today", "latest", "real-time", "realtime", "current", "breaking", "update"]
    }
    is_realtime_query = any(token in normalized_query for token in realtime_terms[language])

    search_queries = [query.strip()]
    project_intent_terms = ["project", "repository", "repo", "github", "library", "package", "framework", "dự án", "du an"]
    is_project_query = any(term in normalized_query for term in project_intent_terms)

    if language == "vi":
        search_queries.append(f"{query.strip()} viet nam")
        if is_project_query:
            search_queries.append(f"{query.strip()} github")
            search_queries.append(f"site:github.com {query.strip()}")
            search_queries.append(f"{query.strip()} tài liệu")
    else:
        search_queries.append(f"{query.strip()} official source")
        if is_project_query:
            search_queries.append(f"{query.strip()} github")
            search_queries.append(f"site:github.com {query.strip()}")
            search_queries.append(f"{query.strip()} documentation")
        if is_realtime_query:
            search_queries.append(f"{query.strip()} latest updates")

    # Remove duplicates while preserving order.
    deduped_queries = list(dict.fromkeys([q for q in search_queries if q]))
    return language, deduped_queries, is_realtime_query


def _is_relevant_result(result: Dict, query_terms: List[str]) -> bool:
    """Keep results that overlap meaningfully with query terms."""
    if not query_terms:
        return True

    text = _normalize_text(
        f"{result.get('title', '')} {result.get('body', '')} {result.get('snippet', '')}"
    )
    if not text:
        return False

    matches = sum(1 for term in query_terms if term in text)
    required_specific = _required_specific_terms(query_terms)
    if required_specific and not any(term in text for term in required_specific):
        return False

    # Require at least one meaningful term, and at least two for longer queries.
    required = 1 if len(query_terms) <= 2 else 2
    return matches >= required


def _deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Deduplicate by URL while preserving order."""
    deduped: List[Dict] = []
    seen = set()
    for item in results:
        url = item.get("href") or item.get("url") or item.get("link") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(item)
    return deduped


def _trusted_domain_bonus(url: str) -> int:
    """Assign a lightweight trust score bonus by source domain."""
    u = _normalize_text(url)
    if not u:
        return 0
    trusted_tokens = [
        ".gov", ".edu", "reuters.com", "apnews.com", "bbc.", "nytimes.com",
        "thanhnien.vn", "tuoitre.vn", "vietnamplus.vn", "chinhphu.vn", "moit.gov.vn",
        "github.com", "gitlab.com", "readthedocs.io", "pypi.org", "npmjs.com"
    ]
    return 2 if any(token in u for token in trusted_tokens) else 0


def _result_score(result: Dict, query_terms: List[str]) -> int:
    """Simple ranking score balancing relevance, source quality, and recency hints."""
    title = _normalize_text(str(result.get("title", "")))
    body = _normalize_text(str(result.get("body", result.get("snippet", ""))))
    text = f"{title} {body}"

    term_hits = sum(1 for t in query_terms if t in text)
    title_hits = sum(1 for t in query_terms if t in title)
    has_date = 1 if result.get("date") else 0
    trust = _trusted_domain_bonus(str(result.get("href") or result.get("url") or result.get("link") or ""))

    return (term_hits * 2) + (title_hits * 2) + has_date + trust


def _sort_results(results: List[Dict], query_terms: List[str]) -> List[Dict]:
    """Sort results by score descending while preserving deterministic ordering."""
    return sorted(results, key=lambda r: _result_score(r, query_terms), reverse=True)


def _run_text_search(ddgs: Any, query: str, max_results: int) -> List[Dict]:
    """Run text search with stable options tuned for timeliness."""
    # Some ddgs/duckduckgo_search versions differ in accepted kwargs.
    for kwargs in (
        {"max_results": max_results, "safesearch": "off", "region": "wt-wt", "timelimit": "m"},
        {"max_results": max_results, "safesearch": "off", "region": "wt-wt"},
        {"max_results": max_results}
    ):
        try:
            return list(ddgs.text(query, **kwargs))
        except TypeError:
            continue
    return []


def _run_news_search(ddgs: Any, query: str, max_results: int) -> List[Dict]:
    """Run news search to prioritize recent, time-sensitive information."""
    for kwargs in (
        {"max_results": max_results, "safesearch": "off", "timelimit": "m"},
        {"max_results": max_results, "timelimit": "m"},
        {"max_results": max_results}
    ):
        try:
            return list(ddgs.news(query, **kwargs))
        except TypeError:
            continue
    return []


def _get_search_provider() -> str:
    """Return the configured search provider name ('ddg' or 'tavily')."""
    return os.environ.get("SEARCH_PROVIDER", "ddg").strip().lower()


def _get_web_search_tavily(query: str, max_results: int = 3) -> str:
    """
    Perform a web search using Tavily and return formatted results.

    Returns an empty string on failure or if Tavily is not configured.
    """
    if not TAVILY_AVAILABLE or TavilyClient is None:
        logger.warning("Tavily search not available (tavily-python not installed)")
        return ""

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        logger.warning("TAVILY_API_KEY not set. Cannot use Tavily search.")
        return ""

    try:
        logger.info(f"Tavily search for: {query[:80]}...")
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            topic="general",
        )

        results: List[Dict] = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", "No title"),
                "body": item.get("content", "No description"),
                "href": item.get("url", ""),
                "source": "Tavily",
                "date": item.get("published_date", ""),
            })

        if not results:
            logger.info("Tavily returned no results")
            return ""

        formatted = format_search_results(results)
        logger.info(f"Tavily returned {len(results)} results")
        return formatted

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return ""


def get_web_search(query: str, max_results: int = 3) -> str:
    """
    Perform a web search and return formatted results.

    The search provider is selected via the SEARCH_PROVIDER env var
    ('ddg' or 'tavily', default 'ddg'). If the primary provider fails
    and Tavily is configured, Tavily is used as an automatic fallback.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 3)

    Returns:
        A formatted string containing search results with titles, snippets, and URLs.
        Returns an empty string if search fails or no results found.
    """
    provider = _get_search_provider()

    # If Tavily is explicitly selected, use it directly.
    if provider == "tavily":
        result = _get_web_search_tavily(query, max_results)
        if result:
            return result
        logger.warning("Tavily selected but returned no results; falling back to DuckDuckGo")

    # DuckDuckGo path (default).
    if not DDGS_AVAILABLE:
        logger.warning("DuckDuckGo Search not available")
        # Attempt Tavily fallback if key is configured.
        if TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY", "").strip():
            logger.info("Attempting Tavily fallback")
            return _get_web_search_tavily(query, max_results)
        return ""
    if DDGS is None:
        logger.warning("DDGS class not available at runtime")
        return ""

    try:
        logger.info(f"Searching web for: {query[:80]}...")

        query_terms = _extract_query_terms(query)
        language, search_queries, is_realtime_query = _build_search_context(query)
        region = "vn-vi" if language == "vi" else "us-en"

        collected: List[Dict] = []
        raw_fallback: List[Dict] = []

        ddgs_ctor = DDGS
        with ddgs_ctor() as ddgs:
            if is_realtime_query:
                logger.info("Realtime query detected. Prioritizing news search.")
                news_results = _run_news_search(ddgs, query, max(max_results * 3, 9))
                collected.extend([r for r in news_results if _is_relevant_result(r, query_terms)])

            for search_query in search_queries:
                if len(_deduplicate_results(collected)) >= max_results:
                    break
                # Prefer locale-aware query for first pass, then fallback to global region.
                text_results = []
                for text_region in (region, "wt-wt"):
                    candidate = []
                    for kwargs in (
                        {"max_results": max(max_results * 3, 9), "safesearch": "off", "region": text_region, "timelimit": "m"},
                        {"max_results": max(max_results * 3, 9), "safesearch": "off", "region": text_region},
                        {"max_results": max(max_results * 3, 9)}
                    ):
                        try:
                            candidate = list(ddgs.text(search_query, **kwargs))
                            break
                        except TypeError:
                            continue
                    if candidate:
                        text_results = candidate
                        break

                raw_fallback.extend(text_results)
                filtered = [r for r in text_results if _is_relevant_result(r, query_terms)]
                collected.extend(filtered)

                if len(_deduplicate_results(collected)) >= max_results:
                    break

            if not is_realtime_query and len(_deduplicate_results(collected)) < max_results:
                logger.info("Text search weak/insufficient. Falling back to news search.")
                news_results = _run_news_search(ddgs, query, max(max_results * 3, 9))
                collected.extend([r for r in news_results if _is_relevant_result(r, query_terms)])

        results = _sort_results(_deduplicate_results(collected), query_terms)[:max_results]
        if not results:
            # If relevance filter is too strict, return top raw results instead of empty.
            results = _sort_results(_deduplicate_results(raw_fallback), query_terms)[:max_results]

        if not results:
            logger.info("No search results found after fallbacks")
            return ""

        formatted_results = format_search_results(results)
        logger.info(f"Found {len(results)} usable search results (language={language}, realtime={is_realtime_query})")

        return formatted_results
    
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {str(e)}")
        # Automatic Tavily fallback when DuckDuckGo raises an exception.
        if TAVILY_AVAILABLE and os.environ.get("TAVILY_API_KEY", "").strip():
            logger.info("Attempting Tavily fallback after DuckDuckGo failure")
            return _get_web_search_tavily(query, max_results)
        return ""


def format_search_results(results: List[Dict]) -> str:
    """
    Format search results into a structured string for LLM consumption.
    
    Args:
        results: List of search result dictionaries from DuckDuckGo
    
    Returns:
        Formatted string with numbered results
    """
    if not results:
        return ""
    
    formatted_parts = []
    now = datetime.now(timezone.utc)
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        body = result.get("body", result.get("snippet", "No description"))
        url = result.get("href") or result.get("url") or result.get("link") or ""
        source = result.get("source", "Web")
        published = result.get("date", "")
        
        # Clean up the body text
        body = body.strip()
        if len(body) > 500:
            body = body[:497] + "..."
        
        formatted_parts.append(
            f"[{i}] {title}\n"
            f"    {body}\n"
            f"    Source Name: {source}\n"
            f"    Published: {published}\n"
            f"    Source: {url}"
        )
    
    header = f"=== Web Search Results (UTC {now.strftime('%Y-%m-%d %H:%M')}) ==="
    return header + "\n\n" + "\n\n".join(formatted_parts)


def build_research_prompt(user_query: str, search_results: str) -> str:
    """
    Build a system prompt that includes web search results for RAG.
    
    Args:
        user_query: The original user query
        search_results: Formatted search results string
    
    Returns:
        A system prompt instructing the LLM to use the search results
    """
    current_year = datetime.now().year
    
    if search_results:
        return f"""You are an intelligent research assistant with access to real-time information.

IMPORTANT: Use the following web search results to provide accurate, up-to-date information for the year {current_year}. 
Cite your sources when appropriate by referencing the result numbers [1], [2], etc.

{search_results}

INSTRUCTIONS:
1. Answer the user's question using the search results above when relevant.
2. If the search results contain the answer, prioritize that information.
3. If the search results don't fully answer the question, use your knowledge but mention any limitations.
4. Always provide clear, well-structured responses.
5. If information might be outdated, note that to the user."""
    else:
        return f"""You are an intelligent research assistant.
The current year is {current_year}. Provide accurate, well-researched responses.
If you're unsure about recent events or data, acknowledge the limitation."""


async def async_get_web_search(query: str, max_results: int = 3) -> str:
    """
    Async wrapper for web search (runs sync search in thread pool).
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
    
    Returns:
        Formatted search results string
    """
    import asyncio
    
    # Run the sync function in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_web_search, query, max_results)
