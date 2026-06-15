# BFL FLUX.2 Proxy

OpenAI-compatible image generation proxy for Black Forest Labs FLUX.2 models.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BFL_API_KEY` | Yes | — | Your BFL API key |
| `BFL_BASE_URL` | No | `https://api.eu.bfl.ai/v1` | BFL API base URL |
| `PROXY_PORT` | No | `8765` | Proxy listen port |

## Build & Run (Podman)

```bash
podman build -t bfl-proxy:latest .
podman run -d --name bfl-proxy -p 8765:8765 \
  -e BFL_API_KEY=your_key \
  -e BFL_BASE_URL=https://api.eu.bfl.ai/v1 \
  bfl-proxy:latest
```

## OpenWebUI Settings

| Setting | Value |
|---------|-------|
| Image Generation Engine | `openai` |
| OpenAI API Base URL | `http://localhost:8765/v1` |
| OpenAI API Key | any non-empty string |
| Default Model | `flux-klein-4b` |
