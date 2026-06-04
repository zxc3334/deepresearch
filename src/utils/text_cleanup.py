"""Utilities for normalizing extracted web text before it enters prompts."""
from __future__ import annotations

import re

_SHORT_LINE_MAX = 80
_SENTENCE_END_RE = re.compile(r"[\u3002\uff01\uff1f.!?;\uff1b:\uff1a]$")
_LIST_OR_HEADING_RE = re.compile(
    r"^(\s*[-*+]\s+|\s*\d+[\.)\u3001]\s+|\s*#{1,6}\s+|\s*\|)"
)


def normalize_extracted_text(text: str) -> str:
    """Normalize whitespace while preserving meaningful paragraph breaks.

    Search engines and HTML/PDF extractors often return one word or phrase per
    line. That hurts report readability and also wastes prompt attention. This
    function merges soft line breaks inside a paragraph, but keeps blank-line
    paragraph boundaries and Markdown/list/table-like lines.
    """
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ \u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = re.split(r"\n\s*\n", text.strip())
    normalized = [_normalize_paragraph(paragraph) for paragraph in paragraphs]
    return "\n\n".join(part for part in normalized if part).strip()


def normalize_snippet(text: str, max_chars: int | None = None) -> str:
    """Normalize a compact one-line snippet and optionally truncate it."""
    snippet = normalize_extracted_text(text)
    snippet = re.sub(r"\s*\n+\s*", " ", snippet)
    snippet = re.sub(r" {2,}", " ", snippet).strip()
    if max_chars is not None and len(snippet) > max_chars:
        return snippet[:max_chars].rstrip()
    return snippet


def _normalize_paragraph(paragraph: str) -> str:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return re.sub(r" {2,}", " ", lines[0]).strip()

    output: list[str] = []
    current = lines[0]
    for line in lines[1:]:
        if _should_keep_line_break(current, line):
            output.append(current.strip())
            current = line
        else:
            current = _join_lines(current, line)
    output.append(current.strip())
    return "\n".join(line for line in output if line)


def _should_keep_line_break(previous: str, current: str) -> bool:
    if _LIST_OR_HEADING_RE.match(previous) or _LIST_OR_HEADING_RE.match(current):
        return True
    if previous.endswith("|") or current.startswith("|"):
        return True
    if len(previous) > _SHORT_LINE_MAX and _SENTENCE_END_RE.search(previous):
        return True
    return False


def _join_lines(previous: str, current: str) -> str:
    if previous.endswith("-") and current and current[0].islower():
        return previous[:-1] + current
    if _is_cjk(previous[-1:]) or _is_cjk(current[:1]):
        return previous + current
    return previous + " " + current


def _is_cjk(char: str) -> bool:
    return bool(char and "\u4e00" <= char <= "\u9fff")
