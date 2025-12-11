"""
Research Tools Pack

Tools for information gathering and research tasks:
- Web search
- URL content reading
- Document parsing
- Summarization
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import structlog

from squadron.connectivity.mcp_host import mcp_tool

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A web search result."""
    title: str
    url: str
    snippet: str
    position: int = 0


@dataclass
class WebPage:
    """Parsed web page content."""
    url: str
    title: str
    content: str
    links: list[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.utcnow)


class ResearchTools:
    """
    Research Tools Pack.
    
    Provides tools for information gathering:
    - Web search via multiple providers
    - URL content fetching and parsing
    - Document summarization
    - Knowledge extraction
    
    Example:
        ```python
        tools = ResearchTools(search_api_key="your-key")
        
        # Search the web
        results = await tools.web_search("Python async patterns")
        
        # Read a URL
        page = await tools.read_url("https://docs.python.org/3/library/asyncio.html")
        
        # Summarize content
        summary = await tools.summarize(page.content)
        ```
    """
    
    def __init__(
        self,
        search_api_key: str | None = None,
        search_provider: str = "serper",
        llm_client: Any | None = None,
        max_content_length: int = 50000,
    ):
        """
        Initialize research tools.
        
        Args:
            search_api_key: API key for search provider
            search_provider: Search provider (serper, tavily, brave)
            llm_client: LLM client for summarization
            max_content_length: Maximum content length to process
        """
        self.search_api_key = search_api_key
        self.search_provider = search_provider
        self.llm_client = llm_client
        self.max_content_length = max_content_length
        
        # HTTP client (lazy loaded)
        self._http_client: Any = None
    
    async def _get_http_client(self) -> Any:
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(
                    timeout=30.0,
                    follow_redirects=True,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; SquadronBot/1.0)",
                    },
                )
            except ImportError:
                raise RuntimeError("httpx not installed. Run: pip install httpx")
        return self._http_client
    
    @mcp_tool(description="Search the web for information")
    async def web_search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "search",
    ) -> list[SearchResult]:
        """
        Search the web.
        
        Args:
            query: Search query
            num_results: Number of results to return
            search_type: Type of search (search, news, images)
            
        Returns:
            List of search results
        """
        if not self.search_api_key:
            logger.warning("No search API key configured")
            return []
        
        if self.search_provider == "serper":
            return await self._search_serper(query, num_results, search_type)
        elif self.search_provider == "tavily":
            return await self._search_tavily(query, num_results)
        elif self.search_provider == "brave":
            return await self._search_brave(query, num_results)
        else:
            raise ValueError(f"Unknown search provider: {self.search_provider}")
    
    async def _search_serper(
        self,
        query: str,
        num_results: int,
        search_type: str,
    ) -> list[SearchResult]:
        """Search using Serper API."""
        client = await self._get_http_client()
        
        url = "https://google.serper.dev/search"
        if search_type == "news":
            url = "https://google.serper.dev/news"
        elif search_type == "images":
            url = "https://google.serper.dev/images"
        
        response = await client.post(
            url,
            headers={
                "X-API-KEY": self.search_api_key,
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "num": num_results,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for i, item in enumerate(data.get("organic", [])[:num_results]):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                position=i + 1,
            ))
        
        return results
    
    async def _search_tavily(
        self,
        query: str,
        num_results: int,
    ) -> list[SearchResult]:
        """Search using Tavily API."""
        client = await self._get_http_client()
        
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.search_api_key,
                "query": query,
                "max_results": num_results,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for i, item in enumerate(data.get("results", [])[:num_results]):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
                position=i + 1,
            ))
        
        return results
    
    async def _search_brave(
        self,
        query: str,
        num_results: int,
    ) -> list[SearchResult]:
        """Search using Brave Search API."""
        client = await self._get_http_client()
        
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": self.search_api_key,
            },
            params={
                "q": query,
                "count": num_results,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for i, item in enumerate(data.get("web", {}).get("results", [])[:num_results]):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                position=i + 1,
            ))
        
        return results
    
    @mcp_tool(description="Read and parse content from a URL")
    async def read_url(
        self,
        url: str,
        extract_links: bool = False,
    ) -> WebPage:
        """
        Read and parse content from a URL.
        
        Args:
            url: URL to read
            extract_links: Whether to extract links
            
        Returns:
            Parsed web page
        """
        client = await self._get_http_client()
        
        response = await client.get(url)
        response.raise_for_status()
        
        html = response.text
        
        # Parse HTML
        title, content, links = self._parse_html(html, url if extract_links else None)
        
        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n\n[Content truncated...]"
        
        return WebPage(
            url=url,
            title=title,
            content=content,
            links=links,
        )
    
    def _parse_html(
        self,
        html: str,
        base_url: str | None = None,
    ) -> tuple[str, str, list[str]]:
        """Parse HTML to extract title, content, and links."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Get title
            title = ""
            if soup.title:
                title = soup.title.string or ""
            
            # Get main content
            main = soup.find("main") or soup.find("article") or soup.body
            if main:
                content = main.get_text(separator="\n", strip=True)
            else:
                content = soup.get_text(separator="\n", strip=True)
            
            # Clean up content
            content = re.sub(r"\n{3,}", "\n\n", content)
            
            # Extract links
            links = []
            if base_url:
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("http"):
                        links.append(href)
                    elif href.startswith("/"):
                        links.append(urljoin(base_url, href))
            
            return title, content, links[:50]  # Limit links
            
        except ImportError:
            # Fallback without BeautifulSoup
            logger.warning("BeautifulSoup not installed, using basic parsing")
            
            # Basic title extraction
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            title = title_match.group(1) if title_match else ""
            
            # Basic content extraction (remove tags)
            content = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
            
            return title, content, []
    
    @mcp_tool(description="Summarize text content")
    async def summarize(
        self,
        content: str,
        max_length: int = 500,
        style: str = "concise",
    ) -> str:
        """
        Summarize text content.
        
        Args:
            content: Content to summarize
            max_length: Maximum summary length
            style: Summary style (concise, detailed, bullet_points)
            
        Returns:
            Summary
        """
        if not self.llm_client:
            # Fallback to extractive summarization
            return self._extractive_summary(content, max_length)
        
        style_instructions = {
            "concise": "Provide a brief, focused summary.",
            "detailed": "Provide a comprehensive summary covering all key points.",
            "bullet_points": "Summarize as a bullet-point list of key points.",
        }
        
        prompt = f"""Summarize the following content. {style_instructions.get(style, style_instructions['concise'])}

Content:
{content[:10000]}

Summary (max {max_length} characters):"""

        try:
            response = await self.llm_client.ainvoke(prompt)
            summary = response.content if hasattr(response, "content") else str(response)
            return summary[:max_length]
        except Exception as e:
            logger.warning("LLM summarization failed", error=str(e))
            return self._extractive_summary(content, max_length)
    
    def _extractive_summary(self, content: str, max_length: int) -> str:
        """Simple extractive summarization."""
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        summary = []
        current_length = 0
        
        for sentence in sentences[:10]:  # First 10 sentences
            if current_length + len(sentence) > max_length:
                break
            summary.append(sentence)
            current_length += len(sentence) + 2
        
        return ". ".join(summary) + "."
    
    @mcp_tool(description="Extract structured information from text")
    async def extract_info(
        self,
        content: str,
        schema: dict[str, str],
    ) -> dict[str, Any]:
        """
        Extract structured information from text.
        
        Args:
            content: Content to extract from
            schema: Schema defining what to extract
                    e.g., {"name": "person's name", "date": "event date"}
            
        Returns:
            Extracted information
        """
        if not self.llm_client:
            return {"error": "LLM client required for extraction"}
        
        schema_desc = "\n".join(f"- {k}: {v}" for k, v in schema.items())
        
        prompt = f"""Extract the following information from the text.

Fields to extract:
{schema_desc}

Text:
{content[:5000]}

Respond with a JSON object containing the extracted fields. Use null for missing fields.
JSON:"""

        try:
            response = await self.llm_client.ainvoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"error": "Could not parse extraction result"}
            
        except Exception as e:
            return {"error": str(e)}
    
    @mcp_tool(description="Compare multiple sources on a topic")
    async def compare_sources(
        self,
        query: str,
        num_sources: int = 3,
    ) -> dict[str, Any]:
        """
        Search and compare multiple sources on a topic.
        
        Args:
            query: Topic to research
            num_sources: Number of sources to compare
            
        Returns:
            Comparison of sources
        """
        # Search for sources
        results = await self.web_search(query, num_results=num_sources)
        
        if not results:
            return {"error": "No search results found"}
        
        # Fetch and summarize each source
        sources = []
        for result in results:
            try:
                page = await self.read_url(result.url)
                summary = await self.summarize(page.content, max_length=300)
                sources.append({
                    "title": result.title,
                    "url": result.url,
                    "summary": summary,
                })
            except Exception as e:
                logger.warning("Failed to fetch source", url=result.url, error=str(e))
        
        # Generate comparison if LLM available
        comparison = ""
        if self.llm_client and len(sources) > 1:
            sources_text = "\n\n".join(
                f"Source {i+1}: {s['title']}\n{s['summary']}"
                for i, s in enumerate(sources)
            )
            
            prompt = f"""Compare these sources on the topic "{query}":

{sources_text}

Provide a brief comparison noting:
1. Points of agreement
2. Points of disagreement
3. Unique insights from each source"""

            try:
                response = await self.llm_client.ainvoke(prompt)
                comparison = response.content if hasattr(response, "content") else str(response)
            except Exception:
                pass
        
        return {
            "query": query,
            "sources": sources,
            "comparison": comparison,
        }
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    def get_tools(self) -> list[Callable]:
        """Get all tools as a list of callables."""
        return [
            self.web_search,
            self.read_url,
            self.summarize,
            self.extract_info,
            self.compare_sources,
        ]
