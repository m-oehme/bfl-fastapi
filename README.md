# BFL FLUX.2 Proxy

OpenAI-compatible image generation proxy for Black Forest Labs FLUX.2 models.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BFL_API_KEY` | Yes | — | Your BFL API key |
| `BFL_BASE_URL` | No | `https://api.eu.bfl.ai/v1` | BFL API base URL |
| `PROXY_PORT` | No | `8765` | Proxy listen port |

## Deploy via Quadlet (Rootful)

```bash
sudo cp bfl-proxy.container /etc/containers/systemd/
sudo systemctl daemon-reload
sudo systemctl start bfl-proxy
```

## OpenWebUI Settings

| Setting | Value |
|---------|-------|
| Image Generation Engine | `openai` |
| OpenAI API Base URL | `http://localhost:8765/v1` |
| OpenAI API Key | any non-empty string |
| Default Model | `flux-klein-4b` |
