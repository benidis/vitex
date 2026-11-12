"""Keyless literature/web research tools the model drives during generation.

These let Claude ground the tutorial in real sources while it writes, rather than
as a postfix: it can look up papers, run a web search, and read a page to confirm
a formula, a definition, or a citation.

Everything here uses only the standard library over plain HTTPS, so there are no
API keys and no compiled dependencies to install:

  * ``arxiv_search``  -> the arXiv Atom API (ideal for ML / math papers)
  * ``web_search``    -> the DuckDuckGo HTML endpoint (general coverage)
  * ``fetch_url``     -> fetch a page and return readable text

Network failures are non-fatal: a tool returns an explanatory string and the
model continues. arXiv is the reliable backbone; web search is best-effort.

``fetch_url`` takes a URL chosen by the model, which may have been steered by
text on a page it read. So every request (and every redirect hop) is vetted:
only http(s), and only hosts that resolve to public addresses — never local
files, loopback, private networks, or the cloud instance-metadata endpoint.
"""

from __future__ import annotations

import html
import ipaddress
import json
import logging
import re
import socket
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable

log = logging.getLogger("vitex")

_UA = "Mozilla/5.0 (compatible; vitex/0.1; +https://example.invalid/vitex)"
_TIMEOUT = 20
_MAX_BYTES = 2_000_000  # cap on any single response body


class UnsafeURLError(ValueError):
    """The URL's scheme or destination address is not allowed."""


def check_public_url(url: str) -> None:
    """Raise :class:`UnsafeURLError` unless ``url`` is http(s) to a public host.

    Every address the host resolves to must be globally routable, which rules
    out loopback, RFC 1918 / ULA private ranges, link-local (including the
    169.254.169.254 metadata service), CGNAT, and reserved ranges.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError(f"only http(s) URLs are allowed, got {parts.scheme or 'none'!r}")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("URL has no host")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (OSError, ValueError) as exc:
        raise UnsafeURLError(f"cannot resolve {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise UnsafeURLError(f"{host!r} resolves to non-public address {ip}")


class _VettedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-apply :func:`check_public_url` to every redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_VettedRedirects)


def _get(url: str, *, data: bytes | None = None) -> str:
    check_public_url(url)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": _UA})
    with _OPENER.open(req, timeout=_TIMEOUT) as resp:  # noqa: S310 - vetted above
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read(_MAX_BYTES).decode(charset, errors="replace")


def _strip_html(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(text).strip()


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv and return the top hits (title, authors, year, id, abstract)."""
    q = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(max_results, 10)),
            "sortBy": "relevance",
        }
    )
    try:
        raw = _get(f"https://export.arxiv.org/api/query?{q}")
        root = ET.fromstring(raw)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, non-fatal
        return f"arxiv_search failed: {exc}"

    entries = []
    for e in root.findall("a:entry", _ATOM):
        title = _strip_html((e.findtext("a:title", default="", namespaces=_ATOM) or "").strip())
        summary = (e.findtext("a:summary", default="", namespaces=_ATOM) or "").strip()
        published = e.findtext("a:published", default="", namespaces=_ATOM) or ""
        abs_url = (e.findtext("a:id", default="", namespaces=_ATOM) or "").strip()
        authors = [
            (a.findtext("a:name", default="", namespaces=_ATOM) or "").strip()
            for a in e.findall("a:author", _ATOM)
        ]
        arxiv_id = abs_url.rsplit("/abs/", 1)[-1] if "/abs/" in abs_url else abs_url
        entries.append(
            {
                "title": " ".join(title.split()),
                "authors": authors,
                "year": published[:4],
                "arxiv_id": arxiv_id,
                "url": abs_url,
                "abstract": " ".join(summary.split())[:1200],
            }
        )
    if not entries:
        return f"No arXiv results for {query!r}."
    return json.dumps(entries, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# General web search (DuckDuckGo HTML endpoint, keyless)
# ---------------------------------------------------------------------------

_RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(?P<snip>.*?)</a>', re.DOTALL
)


def _unwrap_ddg(url: str) -> str:
    """DuckDuckGo wraps outbound links as /l/?uddg=<encoded>; decode when present."""
    if "uddg=" in url:
        try:
            qs = urllib.parse.urlparse(url).query
            uddg = urllib.parse.parse_qs(qs).get("uddg", [])
            if uddg:
                return urllib.parse.unquote(uddg[0])
        except Exception:  # noqa: BLE001 - fall back to the raw href
            pass
    if url.startswith("//"):
        return "https:" + url
    return url


def web_search(query: str, max_results: int = 5) -> str:
    """Run a keyless DuckDuckGo web search; return title/url/snippet per hit."""
    try:
        data = urllib.parse.urlencode({"q": query}).encode()
        raw = _get("https://html.duckduckgo.com/html/", data=data)
    except Exception as exc:  # noqa: BLE001 - non-fatal; arXiv still available
        return f"web_search unavailable ({exc}); try arxiv_search instead."

    titles = list(_RESULT_RE.finditer(raw))
    snippets = _SNIPPET_RE.findall(raw)
    hits = []
    for i, m in enumerate(titles[: max(1, min(max_results, 8))]):
        snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
        hits.append(
            {
                "title": _strip_html(m.group("title")),
                "url": _unwrap_ddg(m.group("url")),
                "snippet": snippet[:400],
            }
        )
    if not hits:
        return f"No web results for {query!r} (or the endpoint blocked the request)."
    return json.dumps(hits, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Fetch a page as readable text
# ---------------------------------------------------------------------------


def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Fetch ``url`` and return its main text with markup and scripts removed."""
    try:
        raw = _get(url)
    except Exception as exc:  # noqa: BLE001 - non-fatal
        return f"fetch_url failed for {url!r}: {exc}"
    raw = re.sub(r"(?is)<(script|style|head|nav|footer)[^>]*>.*?</\1>", " ", raw)
    text = _strip_html(raw)
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text))
    return text[: max(1, min(max_chars, 20000))]


# ---------------------------------------------------------------------------
# Tool specs + dispatch (Anthropic tool-use schema; works on Bedrock too)
# ---------------------------------------------------------------------------

TOOL_SPECS = [
    {
        "name": "arxiv_search",
        "description": (
            "Search arXiv for papers to ground a claim or find the correct "
            "citation (title, authors, year, arXiv id, abstract). Prefer this "
            "for ML, math, and CS topics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms."},
                "max_results": {"type": "integer", "description": "1-10 (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "General web search for definitions, documentation, or blog posts "
            "when arXiv is not the right source. Returns title, url, snippet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms."},
                "max_results": {"type": "integer", "description": "1-8 (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a page (e.g. an arXiv abstract) and return its readable text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "description": "Cap on returned text."},
            },
            "required": ["url"],
        },
    },
]

_DISPATCH: dict[str, Callable[..., str]] = {
    "arxiv_search": arxiv_search,
    "web_search": web_search,
    "fetch_url": fetch_url,
}


def dispatch(name: str, tool_input: dict) -> str:
    """Execute a tool call by name; never raise — return an error string instead."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Unknown tool: {name!r}"
    try:
        return fn(**tool_input)
    except TypeError as exc:
        return f"Bad arguments for {name!r}: {exc}"
    except Exception as exc:  # noqa: BLE001 - tools must not crash the run
        log.warning("Tool %s failed: %s", name, exc)
        return f"{name} error: {exc}"
