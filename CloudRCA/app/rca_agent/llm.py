from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def get_chat_model() -> ChatGoogleGenerativeAI:
    model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash").strip()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_retries=2,
    )


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model response: {cleaned[:800]}")

    return json.loads(cleaned[start : end + 1])


def invoke_json(prompt: str) -> dict[str, Any]:
    response = get_chat_model().invoke([HumanMessage(content=prompt)])
    return extract_json_object(message_text(response.content))
