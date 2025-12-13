"""Minimal MCP-style server using Claude skills for async web search and follow-up API calls."""

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import aiohttp

from config import Config
from models import create_llm

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

    def start_web_search(self, query: str, max_results: int = 5, followup_api: Optional[Dict[str, Any]] = None) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = TaskState(status="pending")
        coro = self._run_web_search(task_id, query, max_results, followup_api)
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

    async def _fetch_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=20) as resp:
                resp.raise_for_status()
                # DuckDuckGo returns application/x-javascript, so ignore content-type
                data = await resp.json(content_type=None)
        
        logger.info(f"DuckDuckGo response keys: {list(data.keys())}")
        
        # Try RelatedTopics first
        related = data.get("RelatedTopics", [])
        cleaned = []
        for item in related:
            # Handle nested Topics arrays
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
        
        # Fallback to Abstract if no results
        if not cleaned and data.get("Abstract"):
            cleaned.append({
                "title": data.get("Heading", "Result"),
                "url": data.get("AbstractURL", "#"),
                "snippet": data.get("Abstract", "")[:200]
            })
        
        logger.info(f"Extracted {len(cleaned)} results from DuckDuckGo")
        return {"count": len(cleaned), "items": cleaned, "raw_data": data}

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
