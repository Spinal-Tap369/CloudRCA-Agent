from fastapi import FastAPI

from app.config import get_settings


app = FastAPI(
    title="CloudRCA ITBench SRE Agent",
    description="Agentic root-cause analysis prototype for ITBench SRE snapshots.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict:
    return {
        "service": "CloudRCA ITBench SRE Agent",
        "status": "running",
    }


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
    }