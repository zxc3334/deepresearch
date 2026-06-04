"""
网页搜索工具 — 支持多后端：SerpAPI / Bing / 博查AI / 秘塔AI

设计理由：
  通过 .env 中的 SEARCH_BACKEND 切换后端，零源码修改。

后端对比：
  - serpapi: 每月 100 次免费，结果最全（Google 数据），国内可访问
  - bing:    微软搜索 API，国内稳定，需 Azure 订阅 Key
  - bocha:   博查AI搜索，国内索引最全，面向 AI Agent 优化
  - metaso:  秘塔AI搜索，中文语义强，有 research 多轮模式
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from ..utils.env_config import get_env
from ..utils.domain_tiers import authority_score, classify_url
from ..utils.text_cleanup import normalize_snippet

__all__ = ["WebSearchTool", "MockWebSearchTool", "OfficialSourceSearchTool", "BaseWebSearchTool"]


class BaseWebSearchTool(ABC):
    """网页搜索工具抽象基类。"""

    MAX_SNIPPET_CHARS = 500

    name: str = "web_search"
    description: str = (
        "Search the web for information. "
        "Supports SerpAPI / Bing / 博查AI(bocha) / 秘塔AI(metaso) backends. "
        "Input: {'query': str, 'top_n': int(optional, default=5)}. "
        "Output: list of {'title': str, 'url': str, 'snippet': str}."
    )

    @abstractmethod
    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        """执行搜索并返回结果。"""
        pass

    def _rank_results(self, results: list[dict[str, Any]], query: str, top_n: int | None = None) -> list[dict[str, Any]]:
        """Normalize, score, and rank search results by source quality and query relevance."""
        ranked: list[dict[str, Any]] = []
        query_terms = self._query_terms(query)

        for position, raw in enumerate(results):
            if not isinstance(raw, dict):
                continue
            item = self._normalize_result(raw)
            score = authority_score(item.get("url", ""))
            score += self._query_relevance_score(item, query_terms)
            score -= min(position, 10) * 0.1
            item["_quality_score"] = round(score, 2)
            item["_source_tier"] = classify_url(item.get("url", "")).value
            ranked.append(item)

        ranked.sort(key=lambda item: item.get("_quality_score", 0.0), reverse=True)
        if top_n is not None:
            return ranked[:top_n]
        return ranked

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        item = dict(result)
        item["title"] = str(item.get("title", "") or "")
        item["url"] = str(item.get("url", "") or "")
        item["snippet"] = self._truncate_snippet(str(item.get("snippet", "") or ""))
        return item

    def _truncate_snippet(self, snippet: str) -> str:
        return normalize_snippet(snippet, max_chars=self.MAX_SNIPPET_CHARS)

    def _query_terms(self, query: str) -> set[str]:
        return {term for term in re.findall(r"[a-zA-Z0-9_\-]{3,}", query.lower())}

    def _query_relevance_score(self, item: dict[str, Any], query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        title = str(item.get("title", "")).lower()
        snippet = str(item.get("snippet", "")).lower()
        url = str(item.get("url", "")).lower()
        title_hits = sum(1 for term in query_terms if term in title)
        snippet_hits = sum(1 for term in query_terms if term in snippet)
        url_hits = sum(1 for term in query_terms if term in url)
        return min(10.0, title_hits * 3.0 + snippet_hits * 1.0 + url_hits * 0.5)

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "top_n": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }


class MockWebSearchTool(BaseWebSearchTool):
    """Mock 搜索工具：用于无网络环境的测试和演示。"""

    def __init__(self, delay_ms: tuple[int, int] = (50, 200)) -> None:
        self.delay_ms = delay_ms

    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000.0)

        query_lower = query.lower()
        mock_db: dict[str, list[dict]] = {
            "transformer": [
                {
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "snippet": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "url": "https://arxiv.org/abs/1810.04805",
                    "snippet": "BERT obtains new state-of-the-art results on eleven natural language processing tasks.",
                },
            ],
            "llm": [
                {
                    "title": "Large Language Models: A Survey",
                    "url": "https://arxiv.org/abs/2303.18223",
                    "snippet": "This survey reviews the recent advances in large language models, including pre-training, adaptation, and applications.",
                },
            ],
            "python": [
                {
                    "title": "Python Documentation",
                    "url": "https://docs.python.org/3/",
                    "snippet": "Official Python programming language documentation.",
                },
            ],
        }

        results: list[dict] = []
        for keyword, entries in mock_db.items():
            if keyword in query_lower:
                results.extend(entries)

        seen = set()
        unique = []
        for r in results:
            key = r["url"]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        results = unique

        if not results:
            results = [
                {
                    "title": f"Mock result for '{query}'",
                    "url": "https://example.com/mock",
                    "snippet": "This is a mock search result for testing purposes.",
                }
            ]

        results = self._rank_results(results, query, top_n=top_n)

        return {
            "query": query,
            "results": results,
            "total": len(results),
        }


class WebSearchTool(BaseWebSearchTool):
    """真实网页搜索工具：支持 SerpAPI 和 Bing Search API 双后端。

    配置优先从 .env / .env.local 读取：
      - SEARCH_BACKEND: 后端选择，可选 "serpapi" | "bing"（默认 serpapi）
      - SERPAPI_KEY / SERPAPI_ENDPOINT: SerpAPI 配置
      - BING_SEARCH_KEY / BING_SEARCH_ENDPOINT: Bing API 配置
    """

    _session: aiohttp.ClientSession | None = None

    def __init__(self, backend: str | None = None, api_key: str | None = None, api_endpoint: str | None = None) -> None:
        self.backend = (backend or get_env("SEARCH_BACKEND", "serpapi")).lower().strip()

        # SerpAPI 配置
        self.serpapi_key = api_key or get_env("SERPAPI_KEY")
        self.serpapi_endpoint = api_endpoint or get_env("SERPAPI_ENDPOINT", "https://serpapi.com/search")

        # Bing API 配置
        self.bing_key = api_key or get_env("BING_SEARCH_KEY")
        self.bing_endpoint = api_endpoint or get_env("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")

        # 博查AI 配置
        self.bocha_key = api_key or get_env("BOCHA_API_KEY")
        self.bocha_endpoint = api_endpoint or get_env("BOCHA_API_ENDPOINT", "https://api.bochaai.com/v1/web-search")

        # 秘塔AI 配置
        self.metaso_key = api_key or get_env("METASO_API_KEY")
        self.metaso_endpoint = api_endpoint or get_env("METASO_API_ENDPOINT", "https://metaso.cn/api/open/search/v2")

    def _get_session(self) -> aiohttp.ClientSession:
        """获取复用的 ClientSession，避免每次搜索新建连接。"""
        if WebSearchTool._session is None or WebSearchTool._session.closed:
            WebSearchTool._session = aiohttp.ClientSession(
                headers={"Accept-Encoding": "gzip, deflate"}
            )
        return WebSearchTool._session

    @classmethod
    async def close_session(cls) -> None:
        """关闭类级别的共享 session。应在程序退出前调用。"""
        if cls._session is not None and not cls._session.closed:
            await cls._session.close()
            cls._session = None

    def __del__(self):
        """析构时尝试关闭 session（同步环境回退）。"""
        if WebSearchTool._session is not None and not WebSearchTool._session.closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.close_session())
            except RuntimeError:
                # 无运行中的事件循环，忽略
                pass

    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        if self.backend == "bing":
            return await self._bing_execute(query, top_n)
        if self.backend == "bocha":
            return await self._bocha_execute(query, top_n)
        if self.backend == "metaso":
            return await self._metaso_execute(query, top_n)
        return await self._serpapi_execute(query, top_n)

    async def _serpapi_execute(self, query: str, top_n: int) -> dict[str, Any]:
        if not self.serpapi_key:
            raise RuntimeError(
                "WebSearchTool (serpapi 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 SERPAPI_KEY，\n"
                "或构造函数传入: WebSearchTool(api_key='your_key')\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        params = {
            "q": query,
            "num": top_n,
            "api_key": self.serpapi_key,
            "engine": "google",
            "gl": "us",
            "hl": "en",
        }

        try:
            session = self._get_session()
            async with session.get(
                self.serpapi_endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("error", f"HTTP {resp.status}")
                        return {
                            "query": query,
                            "results": [],
                            "total": 0,
                            "error": f"SerpAPI 错误: {error_msg}",
                        }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"SerpAPI 网络错误: {e}",
            }

        # 解析 SerpAPI 响应
        organic = data.get("organic_results", [])
        results = []
        for item in organic[:top_n]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        results = self._rank_results(results, query, top_n=top_n)

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "serpapi",
        }

    async def _bing_execute(self, query: str, top_n: int) -> dict[str, Any]:
        if not self.bing_key:
            raise RuntimeError(
                "WebSearchTool (bing 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 BING_SEARCH_KEY，\n"
                "或在 Azure Portal 创建 Bing Search v7 资源获取 Key。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
        params = {"q": query, "count": top_n, "mkt": "en-US"}

        try:
            session = self._get_session()
            async with session.get(
                self.bing_endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("message", f"HTTP {resp.status}")
                        return {
                            "query": query,
                            "results": [],
                            "total": 0,
                            "error": f"Bing API 错误: {error_msg}",
                        }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"Bing API 网络错误: {e}",
            }

        # 解析 Bing 响应
        web_pages = data.get("webPages", {}).get("value", [])
        results = []
        for item in web_pages[:top_n]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        results = self._rank_results(results, query, top_n=top_n)

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "bing",
        }

    async def _bocha_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """博查AI搜索后端。

        文档: https://open.bochaai.com
        特点: 国内网页索引最全，面向 AI Agent 和 RAG 优化，返回结构化摘要。
        """
        if not self.bocha_key:
            raise RuntimeError(
                "WebSearchTool (bocha 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 BOCHA_API_KEY，\n"
                "或访问 https://open.bochaai.com 注册获取。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        payload = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self.bocha_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(
                self.bocha_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error_msg = data.get("message", f"HTTP {resp.status}")
                    return {
                        "query": query,
                        "results": [],
                        "total": 0,
                        "error": f"博查AI 错误: {error_msg}",
                    }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"博查AI 网络错误: {e}",
            }

        # 解析博查响应 — 兼容 web-search 和 ai-search 两种端点返回结构
        results: list[dict] = []

        # 结构 A: /v1/web-search → data.webPages.value[]
        web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
        for item in web_pages[:top_n]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        # 结构 B: /v1/ai-search → data.messages[] content 里含引用
        if not results:
            messages = data.get("data", {}).get("messages", [])
            for msg in messages[:top_n]:
                content = msg.get("content", "")
                if content:
                    results.append({
                        "title": msg.get("role", "引用")[:30],
                        "url": "",
                        "snippet": content[:500],
                    })

        # 去重：同一篇文章的不同 URL（移动端/PC端/转发）会被当作多条结果
        results = self._rank_results(self._deduplicate_results(results), query, top_n=top_n)

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "bocha",
        }

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """对搜索结果去重：基于规范化 URL 和清洗后的标题。"""
        from urllib.parse import urlparse
        import re

        seen_keys: set[str] = set()
        unique: list[dict] = []

        for r in results:
            raw_url = r.get("url", "")
            raw_title = r.get("title", "").strip()

            # --- URL 规范化 ---
            try:
                parsed = urlparse(raw_url)
                netloc = parsed.netloc.lower()
                path = parsed.path.lower().rstrip("/")

                # 去掉移动端前缀
                for prefix in ("m.", "wap.", "mobile.", "app."):
                    if netloc.startswith(prefix):
                        netloc = netloc[len(prefix):]
                        break
                # 去掉 www 前缀
                if netloc.startswith("www."):
                    netloc = netloc[4:]

                # 对常见新闻/博客站，只保留域名+路径（去掉查询参数）
                normalized_url = f"{netloc}{path}"
            except Exception:
                normalized_url = raw_url.lower().strip()

            # --- 标题清洗 ---
            # 去掉常见来源后缀，如 " - 虎嗅网"、"_CSDN博客"、"| 人人都是产品经理"
            cleaned_title = re.sub(
                r"[_\-\s|]*(CSDN博客|虎嗅网|人人都是产品经理|36氪|知乎|搜狐|新浪|网易|腾讯|今日头条|飞书云文档|简书|豆瓣|百度文库|原创力文档|道客巴巴|豆丁网|MBA智库文档|外唐智库|未来智库|中研网|中商产业研究院|三个皮匠报告|book118\.com|doc88\.com|docin\.com|mbalib\.com|askci\.com|chinairn\.com|vzkoo\.com|waitang\.com|sgpjbg\.com|toutiao\.com|sohu\.com|sina\.com|163\.com|qq\.com|ifeng\.com|huxiu\.com|36kr\.com|woshipm\.com|csdn\.net|zhihu\.com|juejin\.cn|segmentfault\.com|cnblogs\.com|简书|知乎专栏|百家号|大鱼号|企鹅号|新浪看点|一点资讯|趣头条|东方财富|雪球|同花顺|财联社|华尔街见闻|界面新闻|澎湃|新京报|南方周末|财新|第一财经|经济观察网|21世纪经济报道|新浪财经|腾讯财经|网易财经|凤凰财经|和讯网|中金在线|东方财富网|中国证券报|上海证券报|证券时报|证券日报|每日经济新闻|第一财经日报|经济参考报|人民日报|新华社|央视新闻|中央广播电视总台|中国日报|环球时报|参考消息|瞭望|半月谈|求是|学习强国|新华网|人民网|中国网|国际在线|中国新闻网|环球网等?)",
                "",
                raw_title,
                flags=re.IGNORECASE,
            ).strip()
            # 再去掉末尾的 " - "、" | "、"_"
            cleaned_title = re.sub(r"[_\-\s|]+$", "", cleaned_title).strip()

            # --- 去重键：优先用 URL，URL 为空时用清洗后的标题 ---
            key = normalized_url if normalized_url else cleaned_title.lower()
            if not key:
                unique.append(r)
                continue

            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(r)

        return unique

    async def _metaso_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """秘塔AI搜索后端。

        文档: https://metaso.cn/open
        特点: 中文语义搜索强，支持 detail / concise / research 模式。
        """
        if not self.metaso_key:
            raise RuntimeError(
                "WebSearchTool (metaso 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 METASO_API_KEY，\n"
                "或访问 https://metaso.cn/open 注册获取。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        payload = {
            "question": query,
            "lang": "zh",
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.metaso_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(
                self.metaso_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if resp.status != 200 or data.get("errCode"):
                    error_msg = data.get("errMsg", f"HTTP {resp.status}")
                    return {
                        "query": query,
                        "results": [],
                        "total": 0,
                        "error": f"秘塔AI 错误: {error_msg}",
                    }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"秘塔AI 网络错误: {e}",
            }

        # 解析秘塔响应
        results: list[dict] = []
        result_data = data.get("data", {})

        # 1. 优先把 text 字段（秘塔 AI 整理的完整答案）作为高价值结果
        text = result_data.get("text", "")
        if text:
            results.append({
                "title": "秘塔AI搜索总结",
                "url": "",
                "snippet": text[:1500],  # 给足上下文，让 LLM 能直接总结
            })

        # 2. 附加参考文献列表（用于溯源）
        refs = result_data.get("references", [])
        for item in refs[:top_n]:
            snippet_parts = []
            if item.get("title"):
                snippet_parts.append(item["title"])
            if item.get("article_type"):
                snippet_parts.append(f"类型: {item['article_type']}")
            if item.get("date"):
                snippet_parts.append(f"日期: {item['date']}")
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": " | ".join(snippet_parts)[:500],
            })

        results = self._rank_results(results, query, top_n=top_n)

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "metaso",
        }


class OfficialSourceSearchTool:
    """Search official GIS/remote-sensing documentation through WebSearchTool."""

    name = "official_source_search"
    description = (
        "Search official GIS/remote-sensing sources such as ESA, USGS, NASA, Copernicus, "
        "Google Earth Engine docs, and Microsoft Planetary Computer. "
        "Use this for sensor capabilities, product specs, bands, resolutions, algorithms, and data access facts. "
        "Input: {'query': str, 'top_n': int(optional), 'domains': list[str](optional)}."
    )

    default_domains = [
        "sentinel.esa.int",
        "sentinels.copernicus.eu",
        "sentiwiki.copernicus.eu",
        "documentation.dataspace.copernicus.eu",
        "esa.int",
        "usgs.gov",
        "nasa.gov",
        "modis.gsfc.nasa.gov",
        "lpdaac.usgs.gov",
        "copernicus.eu",
        "developers.google.com/earth-engine",
        "planetarycomputer.microsoft.com",
        "docs.python.org",
        "pip.pypa.io",
        "pandas.pydata.org",
        "numpy.org",
        "developer.mozilla.org",
        "docs.github.com",
        "learn.microsoft.com",
        "cloud.google.com",
        "docs.aws.amazon.com",
        "platform.openai.com",
        "openai.com",
    ]
    domain_routes = [
        (
            ("python", "pip", "asyncio", "pandas", "numpy"),
            ["docs.python.org", "pip.pypa.io", "pandas.pydata.org", "numpy.org"],
        ),
        (
            ("javascript", "html", "css", "web api", "browser api", "mdn"),
            ["developer.mozilla.org"],
        ),
        (
            ("github", "github actions", "pull request", "git"),
            ["docs.github.com"],
        ),
        (
            ("openai", "chatgpt", "responses api", "assistants api", "openai api"),
            ["platform.openai.com", "openai.com"],
        ),
        (
            ("azure", "microsoft", "windows", "powershell", "typescript"),
            ["learn.microsoft.com"],
        ),
        (
            ("aws", "s3", "lambda", "bedrock"),
            ["docs.aws.amazon.com"],
        ),
        (
            ("google cloud", "gcp", "bigquery", "vertex ai"),
            ["cloud.google.com"],
        ),
        (
            ("google earth engine", "earth engine", "gee"),
            ["developers.google.com/earth-engine"],
        ),
        (
            ("sentinel", "sentinel-2", "copernicus", "msi"),
            ["sentinels.copernicus.eu", "sentiwiki.copernicus.eu", "esa.int", "copernicus.eu"],
        ),
        (
            ("landsat", "tirs", "oli", "surface temperature"),
            ["usgs.gov", "landsat.gsfc.nasa.gov", "nasa.gov", "lpdaac.usgs.gov"],
        ),
        (
            ("worldcover", "esa worldcover"),
            ["esa-worldcover.org", "esa.int"],
        ),
        (
            ("modis", "mod11", "ecostress", "lp daac", "lpdaac"),
            ["lpdaac.usgs.gov", "nasa.gov", "modis.gsfc.nasa.gov"],
        ),
    ]

    def __init__(
        self,
        search_tool: WebSearchTool | None = None,
        max_domain_queries: int = 4,
        max_query_variants: int = 2,
        max_attempts: int = 6,
        preferred_domains: list[str] | None = None,
    ) -> None:
        self.search_tool = search_tool or WebSearchTool()
        self.max_domain_queries = max_domain_queries
        self.max_query_variants = max_query_variants
        self.max_attempts = max_attempts
        self.preferred_domains = self._deduplicate_domains(preferred_domains or [])

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Fact to verify from official GIS/RS sources."},
                        "top_n": {"type": "integer", "description": "Maximum merged results.", "default": 5},
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional official domains to prioritize.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(
        self,
        query: str,
        top_n: int = 5,
        domains: list[str] | None = None,
    ) -> dict[str, Any]:
        selected_domains = self._select_domains(query, domains)
        per_domain = max(1, min(3, top_n))
        merged: list[dict[str, Any]] = []
        errors: list[str] = []
        attempted_domains: list[str] = []
        attempted_queries: list[str] = []
        query_variants = self._query_variants(query)
        site_attempt_budget = max(1, self.max_attempts - len(query_variants))

        for variant in query_variants:
            for domain in selected_domains:
                if len(attempted_queries) >= site_attempt_budget:
                    break
                attempted_domains.append(domain)
                domain_query = f"{variant} site:{domain}"
                attempted_queries.append(domain_query)
                result = await self.search_tool.execute(domain_query, top_n=per_domain)
                if result.get("error"):
                    errors.append(f"{domain}: {result['error']}")
                    continue
                self._collect_official_results(merged, result.get("results", []), selected_domains, variant)
            if len(merged) >= top_n or len(attempted_queries) >= site_attempt_budget:
                break

        if len(merged) < top_n and len(attempted_queries) < self.max_attempts:
            for variant in query_variants:
                if len(attempted_queries) >= self.max_attempts:
                    break
                attempted_queries.append(variant)
                result = await self.search_tool.execute(variant, top_n=max(top_n, per_domain))
                if result.get("error"):
                    errors.append(f"fallback: {result['error']}")
                    continue
                self._collect_official_results(merged, result.get("results", []), selected_domains, variant)
                if len(merged) >= top_n:
                    break

        results = self.search_tool._rank_results(self.search_tool._deduplicate_results(merged), query, top_n=top_n)
        return {
            "query": query,
            "query_variants": query_variants,
            "domains": selected_domains,
            "attempted_domains": attempted_domains,
            "attempted_queries": attempted_queries,
            "results": results,
            "total": len(results),
            "source": f"official_source_search:{getattr(self.search_tool, 'backend', self.search_tool.name)}",
            "errors": errors,
            "evidence_level": "evidence_backed" if results else "speculative",
        }

    def _select_domains(self, query: str, domains: list[str] | None = None) -> list[str]:
        if domains:
            return self._deduplicate_domains(domains)[:self.max_domain_queries]

        query_lower = query.lower()
        route_matches: list[str] = []
        for keywords, route_domains in self.domain_routes:
            if any(keyword in query_lower for keyword in keywords):
                route_matches.extend(route_domains)
        routed: list[str] = []
        routed.extend(route_matches)
        routed.extend(self.preferred_domains)
        routed.extend(self.default_domains)
        return self._deduplicate_domains(routed)[:self.max_domain_queries]

    def _query_variants(self, query: str) -> list[str]:
        query = str(query or "").strip()
        variants: list[str] = [query] if query else []
        lower = query.lower()
        if any(term in lower for term in ("sentinel-2", "sentinel 2", "msi")):
            variants.extend([
                "Sentinel-2 MSI spectral bands thermal infrared official documentation",
                "Sentinel-2 MSI instrument bands specifications",
            ])
        if any(term in lower for term in ("landsat", "tirs", "surface temperature")):
            variants.extend([
                "Landsat Collection 2 surface temperature product official documentation",
                "Landsat 8 9 TIRS thermal infrared surface temperature product",
            ])
        if any(term in lower for term in ("modis", "mod11")):
            variants.extend([
                "MODIS MOD11 land surface temperature product official documentation",
            ])
        if "ecostress" in lower:
            variants.extend([
                "ECOSTRESS land surface temperature product official documentation",
            ])
        if any(term in lower for term in ("google earth engine", "earth engine", "gee")):
            variants.extend([
                "Google Earth Engine Landsat surface temperature official documentation",
            ])
        if "python" in lower or "asyncio" in lower:
            variants.extend([
                "Python official documentation asyncio",
            ])
        if "github actions" in lower:
            variants.extend([
                "GitHub Actions official documentation",
            ])
        if "openai" in lower:
            variants.extend([
                "OpenAI API official documentation",
            ])
        if "powershell" in lower:
            variants.extend([
                "Microsoft PowerShell official documentation",
            ])
        return self._deduplicate_domains(variants)[:self.max_query_variants]

    def _collect_official_results(
        self,
        merged: list[dict[str, Any]],
        results: list[Any],
        domains: list[str],
        query: str,
    ) -> None:
        for item in results:
            if not isinstance(item, dict):
                continue
            item = {**item, "_official_query": query}
            matched_domain = self._matched_domain(item.get("url", ""), domains)
            if not matched_domain:
                continue
            if not self._passes_official_relevance(item):
                continue
            normalized = dict(item)
            normalized["official_domain"] = matched_domain
            normalized["source_type"] = "official"
            merged.append(normalized)

    def _matched_domain(self, url: str, domains: list[str]) -> str:
        for domain in domains:
            if self._url_matches_domain(url, domain):
                return domain
        return ""

    def _passes_official_relevance(self, item: dict[str, Any]) -> bool:
        query_text = str(item.get("_official_query", "") or "")
        haystack = " ".join([
            str(item.get("title", "") or ""),
            str(item.get("url", "") or ""),
            str(item.get("snippet", "") or ""),
        ]).lower()
        required_groups = self._required_relevance_groups(query_text)
        if not required_groups:
            return True
        return all(any(term in haystack for term in group) for group in required_groups)

    def _required_relevance_groups(self, query: str) -> list[tuple[str, ...]]:
        lower = query.lower()
        groups: list[tuple[str, ...]] = []
        if "sentinel-2" in lower or "sentinel 2" in lower or re.search(r"\bmsi\b", lower):
            groups.append(("sentinel-2", "sentinel 2", "/s2", "s2-", "msi"))
        if "landsat" in lower:
            groups.append(("landsat", "tirs", "oli"))
        if "modis" in lower or "mod11" in lower:
            groups.append(("modis", "mod11"))
        if "ecostress" in lower:
            groups.append(("ecostress",))
        if "earth engine" in lower or re.search(r"\bgee\b", lower):
            groups.append(("earth engine", "google earth engine"))
        if "python" in lower or "asyncio" in lower:
            groups.append(("python", "asyncio"))
        if "github" in lower:
            groups.append(("github",))
        if "openai" in lower:
            groups.append(("openai",))
        if "powershell" in lower:
            groups.append(("powershell",))
        return groups

    def _deduplicate_domains(self, domains: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for domain in domains:
            normalized = str(domain).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique

    def _url_matches_domain(self, url: str, domain: str) -> bool:
        from urllib.parse import urlparse

        if not url:
            return False
        try:
            parsed_url = urlparse(url)
            domain_spec = urlparse(domain if "://" in domain else f"https://{domain}")
        except Exception:
            return False
        host = parsed_url.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        expected_host = domain_spec.netloc.lower()
        if expected_host.startswith("www."):
            expected_host = expected_host[4:]
        if not (host == expected_host or host.endswith("." + expected_host)):
            return False
        expected_path = domain_spec.path.rstrip("/")
        if expected_path:
            return parsed_url.path == expected_path or parsed_url.path.startswith(expected_path + "/")
        return True
