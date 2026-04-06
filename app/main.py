import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db as DB
from app.ai.supervisor import AIWorkflowAgent
from app.fleet_client import FleetClient
from app.routes.ai import router as ai_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB.init_schema()
    fleet = FleetClient()
    app.state.ai_agent = AIWorkflowAgent(fleet=fleet)
    log.info("lucid-ai started")
    yield
    log.info("lucid-ai stopped")


app = FastAPI(title="LUCID AI", lifespan=lifespan)
app.include_router(ai_router)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "lucid-ai",
        "model": os.environ.get("OLLAMA_MODEL", ""),
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", ""),
    }
