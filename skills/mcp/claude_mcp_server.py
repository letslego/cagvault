"""Minimal MCP-style server using Claude skills for async web search and follow-up API calls."""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import aiohttp

from config import Config
from models import create_llm
from lancedb_chat import create_lancedb_chat, AdvancedSearchConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    status: str  # pending, running, done, error
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class _BackgroundLoop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: asyncio.Future) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class ClaudeMCPServer:
    """Lightweight MCP-style orchestrator for Claude-powered tools."""

    def __init__(self) -> None:
        self.loop = _BackgroundLoop()
        self.tasks: Dict[str, TaskState] = {}
        self.llm = create_llm(Config.MODEL)
        # Lazily initialized LanceDB chat for hybrid doc search
        self._lancedb_chat = None

    def start_web_search(self, query: str, max_results: int = 5, followup_api: Optional[Dict[str, Any]] = None) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = TaskState(status="pending")
        coro = self._run_web_search(task_id, query, max_results, followup_api)
        self.loop.submit(coro)
        return task_id

    def start_hybrid_doc_search(
        self,
        question: str,
        pre_filter: Optional[str] = None,
        post_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Run hybrid (vector + keyword) document search via LanceDB and return answer plus chunk refs."""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = TaskState(status="pending")
        coro = self._run_hybrid_doc_search(task_id, question, pre_filter, post_filter, top_k)
        self.loop.submit(coro)
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskState]:
        return self.tasks.get(task_id)

    async def _run_web_search(self, task_id: str, query: str, max_results: int, followup_api: Optional[Dict[str, Any]]) -> None:
        state = self.tasks[task_id]
        state.status = "running"
        try:
            search_results = await self._fetch_duckduckgo(query, max_results)
            # Skip slow LLM summarization - return results immediately
            api_response = None
            if followup_api and followup_api.get("url"):
                api_response = await self._call_followup_api(followup_api["url"], followup_api.get("payload"), followup_api.get("method", "POST"))
            state.result = {
                "query": query,
                "search_results": search_results,
                "followup_api": api_response,
            }
            state.status = "done"
            state.completed_at = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Claude MCP task failed: %s", exc)
            state.status = "error"
            state.error = str(exc)
            state.completed_at = time.time()

    async def _run_hybrid_doc_search(
        self,
        task_id: str,
        question: str,
        pre_filter: Optional[str],
        post_filter: Optional[str],
        top_k: int,
    ) -> None:
        state = self.tasks[task_id]
        state.status = "running"
        try:
            # Lazy init LanceDB chat in hybrid mode with boosting
            if self._lancedb_chat is None:
                self._lancedb_chat = create_lancedb_chat({
                    "db_path": "./lancedb",
                    "table_name": "documents",
                    "search_mode": "hybrid",
                    "hybrid_weight": 0.7,
                })

            adv = AdvancedSearchConfig(
                query_type="multi_match",
                fields=["text", "metadata.summary", "metadata.filename"],
                field_boosts={"text": 2.0, "metadata.summary": 1.5, "metadata.filename": 1.0},
                use_multivector=True,
                rerank_top_k=top_k * 2,
            )

            response = self._lancedb_chat.query(
                question,
                search_mode="hybrid",
                pre_filter=pre_filter,
                post_filter=post_filter,
                advanced_search_config=adv,
            )

            # Prepare chunk references
            chunks = [
                {
                    "filename": ctx.get("filename", "Unknown"),
                    "text": ctx.get("text", ""),
                }
                for ctx in (response.contexts or [])[:top_k]
            ]

            state.result = {
                "question": question,
                "answer": response.answer,
                "sources": response.source_nodes,
                "chunks": chunks,
            }
            state.status = "done"
            state.completed_at = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Hybrid doc search failed: %s", exc)
            state.status = "error"
            state.error = str(exc)
            state.completed_at = time.time()

    async def _fetch_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        cleaned = []
        raw_data: Dict[str, Any] = {}

        # Preferred: Tavily if key present
        if not cleaned:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                try:
                    from tavily import TavilyClient

                    client = TavilyClient(api_key=tavily_key)
                    res = await asyncio.to_thread(client.search, query, max_results=max_results)
                    raw_data = {"source": "tavily", **res}
                    for item in res.get("results", []):
                        title = (item.get("title") or "").strip()
                        url = (item.get("url") or "").strip()
                        snippet = (item.get("content") or item.get("snippet") or "").strip()
                        if title and url:
                            cleaned.append({
                                "title": title[:100],
                                "url": url,
                                "snippet": snippet[:200],
                            })
                        if len(cleaned) >= max_results:
                            break
                    logger.info(f"Tavily returned {len(cleaned)} items")
                except Exception as exc:  # noqa: BLE001
                    logger.info("Tavily search failed, falling through: %s", exc)

        # Next: serpapi (Google) if key present
        if not cleaned:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if serpapi_key:
                try:
                    params = {
                        "engine": "google",
                        "q": query,
                        "api_key": serpapi_key,
                        "num": max_results,
                    }
                    async with aiohttp.ClientSession() as session:
                        async with session.get("https://serpapi.com/search.json", params=params, timeout=20) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                    for item in data.get("organic_results", []):
                        title = (item.get("title") or "").strip()
                        link = (item.get("link") or "").strip()
                        snippet = (item.get("snippet") or item.get("snippet_highlighted", "") or "").strip()
                        if title and link:
                            cleaned.append({
                                "title": title[:100],
                                "url": link,
                                "snippet": snippet[:200],
                            })
                        if len(cleaned) >= max_results:
                            break
                    raw_data = {"source": "serpapi", "items": cleaned}
                    logger.info(f"SerpAPI returned {len(cleaned)} items")
                except Exception as exc:  # noqa: BLE001
                    logger.info("SerpAPI search failed, falling back to DDG: %s", exc)

        # Next: duckduckgo_search (organic results)
        if not cleaned:
            try:
                from duckduckgo_search import AsyncDDGS  # type: ignore

                async with AsyncDDGS() as ddgs:
                    async for r in ddgs.atext(query, max_results=max_results):
                        url = r.get("href") or r.get("url") or ""
                        title = (r.get("title") or "").strip()
                        body = (r.get("body") or r.get("snippet") or "").strip()
                        if url and title:
                            cleaned.append({
                                "title": title[:100],
                                "url": url,
                                "snippet": body[:200],
                            })
                        if len(cleaned) >= max_results:
                            break
                raw_data = {"source": "duckduckgo_search", "items": cleaned}
                logger.info(f"DDGS returned {len(cleaned)} items")
            except Exception as exc:  # noqa: BLE001
                logger.info("DDGS primary search failed, falling back to instant answer API: %s", exc)

        # Fallback: instant answer API if still empty
        if not cleaned:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=20) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            raw_data = data
            logger.info(f"DuckDuckGo response keys: {list(data.keys())}")

            # Try RelatedTopics first
            related = data.get("RelatedTopics", [])
            for item in related:
                if "Topics" in item:
                    for sub_item in item["Topics"]:
                        if "Text" in sub_item and "FirstURL" in sub_item:
                            cleaned.append({
                                "title": sub_item.get("Text", "")[:100],
                                "url": sub_item.get("FirstURL", ""),
                                "snippet": sub_item.get("Text", "")[:200]
                            })
                elif "Text" in item and "FirstURL" in item:
                    cleaned.append({
                        "title": item.get("Text", "")[:100],
                        "url": item.get("FirstURL", ""),
                        "snippet": item.get("Text", "")[:200]
                    })
                if len(cleaned) >= max_results:
                    break

            # Secondary: use Results array if present
            if not cleaned:
                for item in data.get("Results", []):
                    if "Text" in item and "FirstURL" in item:
                        cleaned.append({
                            "title": item.get("Text", "")[:100],
                            "url": item.get("FirstURL", ""),
                            "snippet": item.get("Text", "")[:200]
                        })
                    if len(cleaned) >= max_results:
                        break

            # Fallback to Abstract-like fields or any textual clue if still empty
            if not cleaned:
                abstract = (
                    data.get("Abstract")
                    or data.get("AbstractText")
                    or data.get("Definition")
                    or data.get("Answer")
                )
                title = data.get("Heading") or data.get("Entity") or query
                url_fallback = (
                    data.get("AbstractURL")
                    or data.get("DefinitionURL")
                    or data.get("Redirect")
                    or f"https://duckduckgo.com/?q={aiohttp.helpers.quote(query)}"
                )
                snippet = abstract or title
                if snippet:
                    cleaned.append({
                        "title": title or "Result",
                        "url": url_fallback,
                        "snippet": snippet,
                    })
                    logger.info(
                        f"Using fallback text: title={cleaned[0]['title']}, url={cleaned[0]['url']}"
                    )
                else:
                    logger.info(f"No textual fields available; raw keys={list(data.keys())}")

        logger.info(f"Extracted {len(cleaned)} results from DuckDuckGo")
        logger.info(f"Returning count={len(cleaned)}, items={len(cleaned)}, has_raw_data={bool(raw_data)}")
        return {"count": len(cleaned), "items": cleaned, "raw_data": raw_data}

    async def _summarize_results(self, query: str, search_results: Dict[str, Any]) -> str:
        if not self.llm:
            return "LLM unavailable; returning raw search results."
        prompt = (
            "You are a concise research assistant. Summarize the web search findings and "
            "provide 2-3 bullet insights. Query: " + query + "\n\n" + json.dumps(search_results, indent=2)
        )
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        return getattr(response, "content", str(response))

    async def _call_followup_api(self, url: str, payload: Optional[Dict[str, Any]], method: str = "POST") -> Dict[str, Any]:
        method = method.upper()
        json_payload = payload or {}
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, params=json_payload, timeout=20) as resp:
                    resp.raise_for_status()
                    return {"status": resp.status, "body": await resp.json(content_type=None)}
            async with session.post(url, json=json_payload, timeout=20) as resp:
                resp.raise_for_status()
                return {"status": resp.status, "body": await resp.json(content_type=None)}


def get_claude_mcp_server() -> ClaudeMCPServer:
    global _SERVER
    try:
        return _SERVER
    except NameError:
        _SERVER = ClaudeMCPServer()
        return _SERVER
