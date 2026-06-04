"""Pipe-based prompt builders for executable agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

from ..orchestrator.schemas import SubTask, TaskType
from ..utils.output_language import output_language_instruction


PipeFn = Callable[["PromptContext"], str | None]


@dataclass
class PromptContext:
    """Input context for prompt section pipes."""

    task: SubTask | None = None
    runtime_context: dict[str, Any] = field(default_factory=dict)
    available_tools: list[Any] = field(default_factory=list)
    tool_recommendations: list[str] = field(default_factory=list)
    user_instructions: str = ""
    wiki_context: str = ""
    external_prefetch: str = ""
    domain_profile: dict[str, Any] = field(default_factory=dict)


class PromptBuilder:
    """Stable ordered section renderer."""

    def __init__(self, pipes: list[tuple[str, PipeFn]]) -> None:
        self.pipes = pipes
        self.last_debug_sections: list[str] = []

    def build(self, context: PromptContext) -> str:
        parts: list[str] = []
        self.last_debug_sections = []
        for name, pipe in self.pipes:
            content = pipe(context)
            if not content:
                continue
            self.last_debug_sections.append(name)
            parts.append(str(content).strip())
        return "\n\n".join(parts)


class ResearchPromptBuilder:
    """Build system and user prompts for ResearcherAgent tasks."""

    TOOL_GUIDE: dict[str, str] = {
        "wiki_search": (
            "Local wiki search for previously saved evidence-gated research drafts. "
            "Use this first when the task may reuse prior findings, then verify important claims with external sources."
        ),
        "web_search": (
            "General web search for news, market data, industry reports, current events. "
            "Use this as the FIRST tool for most general tasks."
        ),
        "official_source_search": (
            "Official GIS/remote-sensing documentation search. USE for ESA/USGS/NASA/"
            "Copernicus/GEE product specs, sensor bands, algorithms, and official data access facts."
        ),
        "official_doc_fetcher": (
            "Fetch and read an official documentation URL returned by official_source_search. "
            "USE to turn an official URL into page-grounded evidence snippets."
        ),
        "paper_search": (
            "Academic paper search through OpenAlex/Semantic Scholar/ArXiv. USE for GIS/RS methods, "
            "formulas, peer-reviewed evidence, publications, and citation counts."
        ),
        "arxiv_reader": "Legacy academic paper search. Prefer paper_search for new GIS/RS tasks.",
        "browser": (
            "Open a URL and extract full webpage text. USE after web_search when search results are too short."
        ),
        "code_sandbox": (
            "Execute Python code for calculations, data processing, simulations, statistics, and transformations."
        ),
        "calculator": "Quick math evaluation for simple calculations.",
        "notepad": "Write/read intermediate notes during multi-step research.",
        "file_reader": "Read local files only when the task explicitly references a local file path.",
    }

    def __init__(self, domain_profile: dict[str, Any] | None = None) -> None:
        if domain_profile is None:
            from src.core.domain_profiles import resolve_domain_profile

            domain_profile = resolve_domain_profile({"domain_adapter": {"mode": "general"}})
        self.domain_profile = domain_profile
        self._system_builder = PromptBuilder([
            ("identity", self._system_identity),
            ("system_rules", self._system_rules),
            ("output_language", self._system_output_language),
            ("domain_rules", self._system_domain_rules),
            ("tool_guide", self._system_tool_guide),
            ("evidence_checklist", self._system_evidence_checklist),
            ("output_format", self._system_output_format),
        ])
        self._task_builder = PromptBuilder([
            ("task_context", self._task_context),
            ("tool_recommendations", self._task_tool_recommendations),
            ("memory_context", self._task_memory_context),
            ("wiki_context", self._task_wiki_context),
            ("user_instructions", self._task_user_instructions),
            ("external_prefetch", self._task_external_prefetch),
            ("output_language", self._task_output_language),
            ("output_format", self._task_output_format),
        ])

    def system_prompt(self, available_tools: list[Any] | None = None) -> str:
        context = PromptContext(available_tools=available_tools or [], domain_profile=self.domain_profile)
        return self._system_builder.build(context)

    def task_prompt(self, task: SubTask, context: dict, available_tools: list[Any] | None = None) -> str:
        """Build the user prompt for one SubTask."""
        prompt_context = PromptContext(
            task=task,
            runtime_context=context,
            available_tools=available_tools or [],
            tool_recommendations=self.recommend_tools(task, available_tools=available_tools),
            user_instructions=str(context.get("user_instructions", "") or ""),
            wiki_context=str(context.get("wiki_context", "") or ""),
            external_prefetch=str(context.get("external_prefetch", "") or ""),
            domain_profile=self.domain_profile,
        )
        return self._task_builder.build(prompt_context)

    def prompt_debug(self) -> dict[str, list[str]]:
        """Return the last rendered section names for tests and trace/debug use."""
        return {
            "system_sections": list(self._system_builder.last_debug_sections),
            "task_sections": list(self._task_builder.last_debug_sections),
        }

    def direct_analysis_system_prompt(self) -> str:
        language_instruction = output_language_instruction(self.domain_profile.get("output_language", "zh-CN"))
        return (
            "You are a thoughtful analyst. The user has asked a question that cannot be answered by web search "
            "(e.g., analyzing a specific private individual, personal advice, or subjective judgment). "
            "Your job is to provide a reasoned analysis based ONLY on the information already provided in the context. "
            "Do NOT make up facts. Clearly state what is known, what can be reasonably inferred, and what remains unknown. "
            "End with a confidence score (0-1). "
            f"{language_instruction}"
        )

    def recommend_tools(self, task: SubTask, available_tools: list[Any] | None = None) -> list[str]:
        desc_lower = (task.description or "").lower()
        tool_recommendations: list[str] = []

        if self._is_gis_data_fact_task(task):
            tool_recommendations.extend(["official_source_search", "official_doc_fetcher", "paper_search"])

        if any(kw in desc_lower for kw in ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]):
            tool_recommendations.append("paper_search")

        if any(kw in desc_lower for kw in [
            "official", "documentation", "docs", "handbook", "user guide",
            "esa", "usgs", "nasa", "copernicus", "earth engine", "gee",
            "官方", "文档", "手册", "产品说明", "技术报告", "数据门户",
        ]):
            tool_recommendations.extend(["official_source_search", "official_doc_fetcher"])

        if any(kw in desc_lower for kw in ["计算", "flops", "显存", "内存", "参数量", "延迟", "成本", "公式", "数值", "统计", "数学", "推导"]):
            tool_recommendations.extend(["calculator", "code_sandbox"])

        if any(kw in desc_lower for kw in ["详细", "原文", "全文", "深度", "详细内容", "网页内容", "文章正文"]):
            tool_recommendations.append("browser")

        if any(kw in desc_lower for kw in ["文件", "文档", "dataset", "数据集", "pdf", "csv", "json"]):
            tool_recommendations.append("file_reader")

        if any(kw in desc_lower for kw in [
            "landsat", "sentinel", "modis", "era5", "数据源", "数据集", "传感器",
            "波段", "分辨率", "lst", "ndvi", "ndbi", "地表温度",
        ]):
            tool_recommendations.extend(["official_source_search", "official_doc_fetcher", "paper_search"])

        if any(kw in desc_lower for kw in [
            "方法", "公式", "指数", "反演", "lst", "ndvi", "ndbi", "gwr",
            "地理加权回归", "单窗", "单通道", "split-window",
        ]):
            tool_recommendations.extend(["paper_search", "official_source_search"])

        if any(kw in desc_lower for kw in [
            "验证", "兼容", "检查", "风险", "限制", "crs", "云", "云掩膜",
            "空间分辨率", "时间一致性", "验证清单",
        ]):
            tool_recommendations.extend(["official_source_search", "official_doc_fetcher", "paper_search"])

        if self._is_gis_data_fact_task(task):
            priority = [
                "official_source_search",
                "official_doc_fetcher",
                "paper_search",
                "web_search",
                "browser",
                "calculator",
                "code_sandbox",
            ]
        else:
            priority = list(self.domain_profile.get("recommended_tools", []) or [])
        if not priority:
            priority = [
                "official_source_search",
                "official_doc_fetcher",
                "paper_search",
                "browser",
                "web_search",
            ]
        if not tool_recommendations:
            tool_recommendations.append("web_search")

        first_seen: dict[str, int] = {}
        for index, tool_name in enumerate(tool_recommendations):
            first_seen.setdefault(tool_name, index)
        recommendations = sorted(
            first_seen,
            key=lambda name: (
                priority.index(name) if name in priority else len(priority),
                first_seen[name],
            ),
        )
        tool_by_name = {
            str(getattr(tool, "name", "")): tool
            for tool in available_tools or []
            if getattr(tool, "name", "")
        }
        available_names = set(tool_by_name)
        if available_names:
            recommendations = [name for name in recommendations if name in available_names]
            if (
                self._wiki_has_pages(tool_by_name.get("wiki_search"))
                and not self._is_gis_data_fact_task(task)
                and "wiki_search" not in recommendations
            ):
                recommendations.append("wiki_search")
            if not recommendations and "web_search" in available_names:
                recommendations = ["web_search"]
        return recommendations

    def _is_gis_data_fact_task(self, task: SubTask) -> bool:
        desc_lower = (task.description or "").lower()
        if task.task_type == TaskType.DATA_DISCOVERY:
            return True
        return any(kw in desc_lower for kw in [
            "landsat", "sentinel", "modis", "era5", "dataset", "数据源", "数据集", "传感器",
            "波段", "分辨率", "重访", "产品", "tirs", "oli", "msi",
        ])

    def _wiki_has_pages(self, tool: Any | None) -> bool:
        if tool is None or not hasattr(tool, "has_pages"):
            return False
        try:
            return bool(tool.has_pages())
        except Exception:
            return False

    def _system_identity(self, context: PromptContext) -> str:
        return (
            "You are a meticulous research assistant. "
            "Your job is to gather and analyze information using the RIGHT tool for each task."
        )

    def _system_rules(self, context: PromptContext) -> str:
        return (
            "## IMPORTANT RULES\n"
            "1. You MUST use a tool to find factual information. Do NOT answer from your own knowledge.\n"
            "2. Choose the RIGHT tool based on the task type. You can use MULTIPLE tools in sequence.\n"
            "3. For official product specs, standards, documentation, or API facts, START with official_source_search. "
            "If you get an official URL, use official_doc_fetcher to read it before finalizing the claim. "
            "For academic method evidence, use paper_search. For broad discovery, START with web_search.\n"
            "4. If search results are too short, use browser to read the full article.\n"
            "5. If the task involves numbers/calculations, use calculator or code_sandbox.\n"
            "6. You may call tools AT MOST 2 times total. After that you MUST summarize.\n"
            "7. NEVER greet the user or ask what they want to search; execute immediately."
        )

    def _system_output_language(self, context: PromptContext) -> str:
        return "## OUTPUT LANGUAGE\n" + output_language_instruction(
            context.domain_profile.get("output_language", "zh-CN")
        )

    def _system_domain_rules(self, context: PromptContext) -> str | None:
        sections = set(context.domain_profile.get("prompt_sections", []) or [])
        if not sections:
            return None
        parts = ["## DOMAIN PROFILE: " + str(context.domain_profile.get("name", "domain"))]
        if "gis_rs_data_rules" in sections:
            parts.append(
                "### GIS/RS Data Rules\n"
                "- Use official documentation for sensor bands, spatial resolution, revisit cycle, product level, and access constraints.\n"
                "- Do not claim Sentinel-2 can directly retrieve LST; it has no thermal infrared band.\n"
                "- Distinguish native thermal resolution from resampled product resolution, especially for Landsat TIRS."
            )
        if "gis_rs_method_validation" in sections:
            parts.append(
                "### GIS/RS Method Validation\n"
                "- LST workflows must name the thermal data source and retrieval method.\n"
                "- NDVI/NDBI formulas must name the bands or wavelengths used.\n"
                "- Method claims should be supported by papers or authoritative technical documentation."
            )
        if "gis_rs_risk_checklist" in sections:
            parts.append(
                "### GIS/RS Risk Checklist\n"
                "- Check cloud masking, season consistency, CRS alignment, mixed pixels, scale mismatch, water/bare-soil confusion, and clear-sky LST bias.\n"
                "- Report limitations and uncertainty instead of presenting remote-sensing outputs as exact ground truth."
            )
        if "gis_rs_constraint_extraction" in sections:
            parts.append(
                "### GIS/RS Constraint Extraction\n"
                "- Extract AOI, time range, target variable, candidate sensors, required spatial/temporal resolution, and validation needs.\n"
                "- If the user did not specify a constraint, mark it as Needs Clarification instead of inventing it."
            )
        domains = context.domain_profile.get("preferred_official_domains", []) or []
        if domains:
            parts.append("### Preferred Official Domains\n" + "\n".join(f"- {domain}" for domain in domains[:12]))
        output_sections = context.domain_profile.get("output_sections", []) or []
        if output_sections:
            parts.append(
                "### Required Domain Output Sections\n"
                + "\n".join(f"- {section}" for section in output_sections)
            )
        return "\n\n".join(parts)

    def _system_tool_guide(self, context: PromptContext) -> str | None:
        tool_by_name = {
            str(getattr(tool, "name", "")): tool
            for tool in context.available_tools
            if getattr(tool, "name", "")
        }
        tool_names = list(tool_by_name)
        if not tool_names:
            return None
        lines = ["## AVAILABLE TOOLS"]
        for name in tool_names:
            guide = self.TOOL_GUIDE.get(name)
            if guide:
                lines.append(f"- {name}: {guide}")
            else:
                lines.append(f"- {name}: {getattr(tool_by_name[name], 'description', 'Registered tool.')}")
        return "\n".join(lines)

    def _system_evidence_checklist(self, context: PromptContext) -> str | None:
        checklist = context.domain_profile.get("evidence_checklist", []) or []
        if not checklist:
            return None
        return "## EVIDENCE CHECKLIST\n" + "\n".join(f"- {item}" for item in checklist)

    def _system_output_format(self, context: PromptContext) -> str:
        output_sections = context.domain_profile.get("output_sections", []) or []
        if not output_sections:
            return (
                "## OUTPUT FORMAT\n"
                "Only after gathering information, provide a concise summary and include a confidence score (0-1)."
            )
        return (
            "## OUTPUT FORMAT\n"
            "Only after gathering information, provide a concise summary and include a confidence score (0-1).\n"
            "When the task matches the domain, include these domain-specific sections when relevant:\n"
            + "\n".join(f"- {section}" for section in output_sections)
        )

    def _task_context(self, context: PromptContext) -> str:
        task = context.task
        if task is None:
            return ""
        lines = [
            f"## Task: {task.description}",
            f"Type: {task.task_type.value}",
            f"Expected output: {task.expected_type}",
        ]
        if task.search_hints:
            lines.insert(1, f"Search hints (MUST use these as primary keywords): {', '.join(task.search_hints)}")
        return "\n".join(lines)

    def _task_tool_recommendations(self, context: PromptContext) -> str:
        recommendations = context.tool_recommendations or ["web_search"]
        primary_tool = recommendations[0]
        secondary_tools = recommendations[1:]
        lines = [
            f"## RECOMMENDED TOOLS (in priority order): {', '.join(recommendations)}",
        ]
        if secondary_tools:
            lines.append(f"Start with '{primary_tool}'. If needed, also use {', '.join(secondary_tools)}.")
        else:
            lines.append(f"Use '{primary_tool}' to gather information.")
        return "\n".join(lines)

    def _task_memory_context(self, context: PromptContext) -> str | None:
        task = context.task
        if task is None or not task.context_keys:
            return None
        ctx_parts = []
        for key in task.context_keys:
            if key in context.runtime_context:
                ctx_parts.append(f"- {key}: {context.runtime_context[key]}")
        if not ctx_parts:
            return None
        return "## Memory Context\n" + "\n".join(ctx_parts)

    def _task_wiki_context(self, context: PromptContext) -> str | None:
        if not context.wiki_context:
            return None
        if context.wiki_context.lstrip().startswith("## Wiki Context"):
            return context.wiki_context
        return "## Wiki Context\n" + context.wiki_context

    def _task_user_instructions(self, context: PromptContext) -> str | None:
        if not context.user_instructions:
            return None
        return "## User Instructions\n" + context.user_instructions

    def _task_external_prefetch(self, context: PromptContext) -> str | None:
        if not context.external_prefetch:
            return None
        return "## External Prefetch Evidence\n" + context.external_prefetch

    def _task_output_language(self, context: PromptContext) -> str:
        return "## OUTPUT LANGUAGE\n" + output_language_instruction(
            context.domain_profile.get("output_language", "zh-CN")
        )

    def _task_output_format(self, context: PromptContext) -> str:
        primary_tool = (context.tool_recommendations or ["web_search"])[0]
        return (
            "## INSTRUCTIONS\n"
            f"1. First, call the '{primary_tool}' tool with a relevant query to gather information.\n"
            "2. Review the results.\n"
            f"3. If needed, call '{primary_tool}' ONE MORE time with a refined query.\n"
            "   You may call tools AT MOST 2 times total. After the 2nd call, you MUST write the final summary.\n"
            "4. If search results are too short, you may use 'browser' to read the full article (counts as 1 tool call).\n"
            "5. If calculations are needed, use 'calculator' or 'code_sandbox' (counts as 1 tool call).\n"
            "6. Finally, summarize your findings with a confidence score (0-1), following the configured output language.\n"
            "7. DO NOT greet the user or ask clarifying questions; execute immediately.\n"
            "8. IMPORTANT: Your query MUST directly address the task description."
        )

    def is_non_searchable(self, task: SubTask, context: dict) -> bool:
        """Heuristically detect tasks that cannot be answered by web search."""
        desc = (task.description or "").lower()
        query = context.get("query", "").lower()
        combined = desc + " " + query

        if "朋友" in combined or "同学" in combined or "同事" in combined:
            if any(w in combined for w in ["分析", "评价", "是什么样", "性格", "人品"]):
                return True

        if any(w in combined for w in ["建议我", "我该怎么", "适合我吗", "要不要"]):
            if "朋友" in combined or "我" in query:
                return True

        if "叫" in combined and any(w in combined for w in ["分析", "评价", "是什么样"]):
            return True

        return False
