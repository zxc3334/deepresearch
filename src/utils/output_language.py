"""Output language configuration helpers."""
from __future__ import annotations


def normalize_output_language(language: str | None) -> str:
    value = str(language or "zh-CN").strip()
    aliases = {
        "zh": "zh-CN",
        "zh_cn": "zh-CN",
        "zh-cn": "zh-CN",
        "chinese": "zh-CN",
        "中文": "zh-CN",
        "简体中文": "zh-CN",
        "en": "en-US",
        "en_us": "en-US",
        "en-us": "en-US",
        "english": "en-US",
    }
    return aliases.get(value.lower(), value)


def output_language_instruction(language: str | None) -> str:
    normalized = normalize_output_language(language)
    if normalized == "zh-CN":
        return (
            "Output language: Simplified Chinese. "
            "All final answers, summaries, analysis, confidence explanations, and wiki-style knowledge text "
            "must be written in Simplified Chinese. Keep necessary technical terms, paper titles, API names, "
            "URLs, model names, and product names in their original form."
        )
    if normalized == "en-US":
        return (
            "Output language: English. "
            "Write final answers, summaries, analysis, and confidence explanations in English."
        )
    return f"Output language: {normalized}. Follow this language for final answers and summaries."
