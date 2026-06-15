"""
BFL FLUX.2 OpenAI-Compatible Image Generation, Editing & Variations Proxy

Translates OpenAI /v1/images/generations, /v1/images/edits and
/v1/images/variations requests to BFL's async API.
Supports all FLUX.2 models via the BFL API.
"""

import asyncio
import base64
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# ── Configuration ──────────────────────────────────────────────────────────

BFL_API_KEY = os.environ["BFL_API_KEY"]
BFL_BASE_URL = os.environ.get("BFL_BASE_URL", "https://api.eu.bfl.ai/v1").rstrip("/")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8765"))

# Default prompt used for /v1/images/variations when no prompt is provided
DEFAULT_VARIATION_PROMPT = os.environ.get(
    "BFL_VARIATION_PROMPT",
    "Create a creative variation of this image, keeping the same subject and style but with subtle differences in composition, lighting, and details."
)

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

# Size → aspect_ratio mapping for editing
ASPECT_RATIO_MAP = {
    "256x256": "1:1",
    "512x512": "1:1",
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "2048x2048": "1:1",
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
    # Editing fields (optional — when present, triggers edit mode)
    input_image: Optional[str] = None
    input_image_2: Optional[str] = None
    input_image_3: Optional[str] = None
    input_image_4: Optional[str] = None
    input_image_5: Optional[str] = None
    input_image_6: Optional[str] = None
    input_image_7: Optional[str] = None
    input_image_8: Optional[str] = None
    output_format: Optional[str] = "jpeg"
    safety_tolerance: Optional[int] = Field(default=2, ge=0, le=6)
    seed: Optional[int] = None


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
    description="OpenAI-compatible image generation, editing & variations proxy for Black Forest Labs FLUX.2",
    version="1.2.2",
    lifespan=lifespan,
)


# ── Helpers ────────────────────────────────────────────────────────────────

async def _bfl_submit(model: str, payload: dict) -> dict:
    """Submit generation/editing job to BFL, return {id, polling_url}."""
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
            if status in ("Error", "Failed"):
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


def _size_to_aspect_ratio(size: Optional[str]) -> Optional[str]:
    if size and size in ASPECT_RATIO_MAP:
        return ASPECT_RATIO_MAP[size]
    return None


async def _uploadfile_to_base64(file) -> str:
    """Convert an uploaded file (starlette UploadFile) to raw base64 string."""
    content = await file.read()
    return base64.b64encode(content).decode("utf-8")


def _strip_data_url_prefix(value: Optional[str]) -> Optional[str]:
    """Strip data:image/...;base64, prefix if present, returning raw base64."""
    if not value:
        return value
    match = re.match(r"^data:[^;]+;base64,(.+)$", value)
    if match:
        return match.group(1)
    return value


def _build_openai_response(image_url: str, response_format: str, created: int) -> dict:
    """Build OpenAI-compatible response."""
    if response_format == "b64_json":
        return {
            "created": created,
            "data": [{"url": image_url}],
        }
    return {
        "created": created,
        "data": [{"url": image_url}],
    }


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

    # Determine if this is an editing request
    is_edit = req.input_image is not None

    if is_edit:
        # Build BFL editing payload
        bfl_payload = {
            "prompt": req.prompt,
            "input_image": _strip_data_url_prefix(req.input_image),
            "output_format": req.output_format or "jpeg",
            "safety_tolerance": req.safety_tolerance if req.safety_tolerance is not None else 2,
        }

        # Add optional reference images
        for i in range(2, 9):
            field = f"input_image_{i}"
            value = getattr(req, field, None)
            if value:
                bfl_payload[field] = _strip_data_url_prefix(value)

        aspect_ratio = _size_to_aspect_ratio(req.size)
        if aspect_ratio:
            bfl_payload["aspect_ratio"] = aspect_ratio

        if req.seed is not None:
            bfl_payload["seed"] = req.seed

    else:
        # Build BFL text-to-image payload
        width, height = _size_to_wh(req.size)
        bfl_payload = {
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "prompt_upsampling": req.quality == "hd",
            "seed": req.seed,
        }

    # Submit to BFL
    submit = await _bfl_submit(bfl_model, bfl_payload)
    polling_url = submit.get("polling_url")
    if not polling_url:
        raise HTTPException(502, "BFL did not return polling_url")

    # Poll for result
    result = await _bfl_poll(polling_url)
    image_url = result.get("result", {}).get("sample")
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


@app.post("/v1/images/edits")
async def image_edits(request: Request):
    """
    OpenAI-compatible image editing endpoint.
    Accepts multipart/form-data with image file(s).
    Supports both 'image' and 'image[]' field names (OpenWebUI sends 'image[]').
    Converts uploaded file to raw base64 and forwards to BFL's editing API.
    """
    form = await request.form()

    # Find the first image file — OpenWebUI sends 'image[]', but also accept 'image'
    image_file = None
    for key in ("image", "image[]"):
        files = form.getlist(key)
        for f in files:
            if hasattr(f, "read"):  # starlette UploadFile
                image_file = f
                break
        if image_file:
            break

    if not image_file:
        raise HTTPException(400, "No image file found in request. Expected 'image' or 'image[]' field.")

    # Extract other form fields
    prompt = form.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "Missing required field: prompt")

    model = form.get("model", "flux-2-pro")
    size = form.get("size", "1024x1024")
    response_format = form.get("response_format", "url")

    bfl_model = MODEL_MAP.get(model)
    if not bfl_model:
        raise HTTPException(
            400,
            f"Unknown model '{model}'. Supported: {', '.join(MODEL_MAP.keys())}"
        )

    # Convert uploaded image to raw base64 (BFL native format)
    input_image = await _uploadfile_to_base64(image_file)

    # Build BFL editing payload
    bfl_payload = {
        "prompt": prompt,
        "input_image": input_image,
        "output_format": "jpeg",
        "safety_tolerance": 2,
    }

    # Add aspect_ratio if size is specified
    aspect_ratio = _size_to_aspect_ratio(size)
    if aspect_ratio:
        bfl_payload["aspect_ratio"] = aspect_ratio

    # Submit to BFL
    submit = await _bfl_submit(bfl_model, bfl_payload)
    polling_url = submit.get("polling_url")
    if not polling_url:
        raise HTTPException(502, "BFL did not return polling_url")

    # Poll for result
    result = await _bfl_poll(polling_url)
    image_url = result.get("result", {}).get("sample")
    if not image_url:
        raise HTTPException(502, "BFL result missing image URL")

    # Build OpenAI-compatible response
    created = int(time.time())

    if response_format == "b64_json":
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


@app.post("/v1/images/variations")
async def image_variations(request: Request):
    """
    OpenAI-compatible image variations endpoint.
    Accepts multipart/form-data with image file(s).
    Supports both 'image' and 'image[]' field names (OpenWebUI sends 'image[]').
    Converts uploaded file to raw base64 and forwards to BFL's editing API.
    """
    form = await request.form()

    # Find the first image file — OpenWebUI sends 'image[]', but also accept 'image'
    image_file = None
    for key in ("image", "image[]"):
        files = form.getlist(key)
        for f in files:
            if hasattr(f, "read"):  # starlette UploadFile
                image_file = f
                break
        if image_file:
            break

    if not image_file:
        raise HTTPException(400, "No image file found in request. Expected 'image' or 'image[]' field.")

    # Extract other form fields
    model = form.get("model", "flux-2-pro")
    size = form.get("size", "1024x1024")
    response_format = form.get("response_format", "url")

    bfl_model = MODEL_MAP.get(model)
    if not bfl_model:
        raise HTTPException(
            400,
            f"Unknown model '{model}'. Supported: {', '.join(MODEL_MAP.keys())}"
        )

    # Convert uploaded image to raw base64 (BFL native format)
    input_image = await _uploadfile_to_base64(image_file)

    # Build BFL editing payload with variation prompt
    bfl_payload = {
        "prompt": DEFAULT_VARIATION_PROMPT,
        "input_image": input_image,
        "output_format": "jpeg",
        "safety_tolerance": 2,
    }

    # Add aspect_ratio if size is specified
    aspect_ratio = _size_to_aspect_ratio(size)
    if aspect_ratio:
        bfl_payload["aspect_ratio"] = aspect_ratio

    # Submit to BFL
    submit = await _bfl_submit(bfl_model, bfl_payload)
    polling_url = submit.get("polling_url")
    if not polling_url:
        raise HTTPException(502, "BFL did not return polling_url")

    # Poll for result
    result = await _bfl_poll(polling_url)
    image_url = result.get("result", {}).get("sample")
    if not image_url:
        raise HTTPException(502, "BFL result missing image URL")

    # Build OpenAI-compatible response
    created = int(time.time())

    if response_format == "b64_json":
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
