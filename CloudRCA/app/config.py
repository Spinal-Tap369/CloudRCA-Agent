import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    llm_model: str = "gpt-4.1-mini"


def get_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    return Settings(
        openai_api_key=api_key,
        llm_model=model,
    )