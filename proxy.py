"""
BFL FLUX.2 OpenAI-Compatible Image Generation Proxy

Translates OpenAI /v1/images/generations requests to BFL's async API.
Supports all FLUX.2 models via the BFL API.
"""

import asyncio
import base64
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Configuration ──────────────────────────────────────────────────────────

BFL_API_KEY = os.environ["BFL_API_KEY"]
BFL_BASE_URL = os.environ.get("BFL_BASE_URL", "https://api.eu.bfl.ai/v1").rstrip("/")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8765"))

# Map OpenAI model names → BFL endpoint paths
MODEL_MAP = {
    "flux-klein-4b": "flux-2-klein-4b",
    "flux-klein-9b": "flux-2-klein-9b",
    "flux-klein-9b-preview": "flux-2-klein-9b-preview",
    "flux-2-pro": "flux-2-pro",
    "flux-2-pro-preview": "flux-2-pro-preview",
    "flux-2-flex": "flux-2-flex",
    "flux-2-max": "flux-2-max",
    # Also accept raw BFL names
    "flux-2-klein-4b": "flux-2-klein-4b",
    "flux-2-klein-9b": "flux-2-klein-9b",
    "flux-2-klein-9b-preview": "flux-2-klein-9b-preview",
}

# Size → width/height mapping
SIZE_MAP = {
    "256x256": (256, 256),
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "1792x1024": (1792, 1024),
    "1024x1792": (1024, 1792),
    "2048x2048": (2048, 2048),
}

# ── Pydantic Models ────────────────────────────────────────────────────────

class ImageGenerationRequest(BaseModel):
    model: str = "flux-klein-4b"
    prompt: str
    n: Optional[int] = Field(default=1, ge=1, le=4)
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url"
    style: Optional[str] = None
    user: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "black-forest-labs"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── FastAPI App ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
        headers={"x-key": BFL_API_KEY, "Content-Type": "application/json"},
    )
    yield
    await app.state.http.close()


app = FastAPI(
    title="BFL FLUX.2 Proxy",
    description="OpenAI-compatible image generation proxy for Black Forest Labs FLUX.2",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ────────────────────────────────────────────────────────────────

async def _bfl_submit(model: str, payload: dict) -> dict:
    """Submit generation job to BFL, return {id, polling_url}."""
    async with app.state.http.post(
        f"{BFL_BASE_URL}/{model}",
        json=payload,
        headers={"x-key": BFL_API_KEY},
    ) as resp:
        if resp.status == 402:
            raise HTTPException(402, "BFL API: insufficient credits")
        if resp.status == 429:
            raise HTTPException(429, "BFL API: rate limited (max 24 active tasks)")
        if resp.status != 200:
            text = await resp.text()
            raise HTTPException(502, f"BFL submit error {resp.status}: {text}")
        return await resp.json()


async def _bfl_poll(polling_url: str, max_wait: float = 120.0) -> dict:
    """Poll BFL until result is ready. Returns result dict with 'sample' URL."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        async with app.state.http.get(polling_url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(502, f"BFL poll error {resp.status}: {text}")
            data = await resp.json()
            status = data.get("status")
            if status == "Ready":
                return data
            if status == "Error":
                raise HTTPException(502, f"BFL generation failed: {data}")
        await asyncio.sleep(0.5)
    raise HTTPException(504, "BFL generation timed out")


async def _fetch_image(url: str) -> bytes:
    """Download image from signed URL."""
    async with app.state.http.get(url) as resp:
        if resp.status != 200:
            raise HTTPException(502, f"Failed to fetch image: {resp.status}")
        return await resp.read()


def _size_to_wh(size: Optional[str]) -> tuple[int, int]:
    if size and size in SIZE_MAP:
        return SIZE_MAP[size]
    return 1024, 1024


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models() -> ModelList:
    now = int(time.time())
    return ModelList(
        data=[
            ModelInfo(id=name, created=now)
            for name in MODEL_MAP.keys()
        ]
    )


@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest):
    bfl_model = MODEL_MAP.get(req.model)
    if not bfl_model:
        raise HTTPException(
            400,
            f"Unknown model '{req.model}'. Supported: {', '.join(MODEL_MAP.keys())}"
        )

    width, height = _size_to_wh(req.size)

    # Build BFL payload
    bfl_payload = {
        "prompt": req.prompt,
        "width": width,
        "height": height,
        "prompt_upsampling": req.quality == "hd",
        "seed": None,
    }

    # Submit to BFL
    submit = await _bfl_submit(bfl_model, bfl_payload)
    polling_url = submit.get("polling_url")
    if not polling_url:
        raise HTTPException(502, "BFL did not return polling_url")

    # Poll for result
    result = await _bfl_poll(polling_url)
    image_url = result.get("sample")
    if not image_url:
        raise HTTPException(502, "BFL result missing image URL")

    # Build OpenAI-compatible response
    created = int(time.time())

    if req.response_format == "b64_json":
        image_bytes = await _fetch_image(image_url)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "created": created,
            "data": [{"b64_json": b64}],
        }

    return {
        "created": created,
        "data": [{"url": image_url}],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
