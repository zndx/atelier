<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Anthropic Models API -- Dynamic Model Discovery

Research findings on programmatically listing available Anthropic models.

## 1. REST API Endpoint

**Endpoint:** `GET https://api.anthropic.com/v1/models`

```bash
curl https://api.anthropic.com/v1/models \
    -H 'anthropic-version: 2023-06-01' \
    -H "X-Api-Key: $ANTHROPIC_API_KEY"
```

### Query Parameters

| Parameter  | Type   | Description |
|------------|--------|-------------|
| `limit`    | int    | Items per page. Default 20, max 1000 |
| `after_id` | string | Cursor for next page |
| `before_id`| string | Cursor for previous page |

### Response Schema

```json
{
  "data": [
    {
      "id": "claude-opus-4-6",
      "type": "model",
      "display_name": "Claude Opus 4.6",
      "created_at": "2026-02-04T00:00:00+00:00",
      "max_input_tokens": 1000000,
      "max_tokens": 128000,
      "capabilities": {
        "batch": { "supported": true },
        "citations": { "supported": true },
        "code_execution": { "supported": true },
        "thinking": { "supported": true, "types": { "adaptive": { "supported": true }, "enabled": { "supported": true } } },
        "effort": { "supported": true, "low": {...}, "medium": {...}, "high": {...}, "max": {...} },
        "image_input": { "supported": true },
        "pdf_input": { "supported": true },
        "structured_outputs": { "supported": true },
        "context_management": { "supported": true, ... }
      }
    }
  ],
  "first_id": "...",
  "last_id": "...",
  "has_more": false
}
```

### Single Model Retrieval

**Endpoint:** `GET https://api.anthropic.com/v1/models/{model_id}`

Accepts aliases (e.g., `claude-opus-4-6`) -- resolves to the concrete model ID.

## 2. Python SDK (anthropic v0.89.0)

### Classes

- `Anthropic().models` -- sync client, returns `SyncPage[ModelInfo]`
- `AsyncAnthropic().models` -- async client, returns `AsyncPage[ModelInfo]`
- `AnthropicVertex().models` -- Vertex AI client
- `AnthropicBedrock().models` -- Bedrock client
- Beta variants at `client.beta.models`

### Methods

```python
from anthropic import Anthropic

client = Anthropic()

# List all models (paginated)
page = client.models.list(limit=50)
for model in page.data:
    print(model.id, model.display_name, model.max_input_tokens)

# Retrieve a specific model (or resolve an alias)
model = client.models.retrieve("claude-opus-4-6")
```

### ModelInfo Fields

| Field              | Type                   | Description |
|--------------------|------------------------|-------------|
| `id`               | `str`                  | Unique model identifier |
| `display_name`     | `str`                  | Human-readable name |
| `created_at`       | `datetime`             | Release timestamp |
| `max_input_tokens` | `Optional[int]`        | Context window size |
| `max_tokens`       | `Optional[int]`        | Max output tokens |
| `capabilities`     | `ModelCapabilities`    | Feature support flags |
| `type`             | `Literal["model"]`     | Always "model" |

Source: `.devenv/state/venv/lib/python3.12/site-packages/anthropic/types/model_info.py`

### Pagination

```python
page = client.models.list(limit=3)
while page.has_more:
    page = client.models.list(limit=3, after_id=page.last_id)
```

## 3. Model Aliases

### Working aliases (confirmed via live API)

| Alias               | Resolves To                       | Display Name       |
|---------------------|-----------------------------------|--------------------|
| `claude-opus-4-6`   | `claude-opus-4-6`                 | Claude Opus 4.6    |
| `claude-sonnet-4-6` | `claude-sonnet-4-6`               | Claude Sonnet 4.6  |
| `claude-haiku-4-5`  | `claude-haiku-4-5-20251001`       | Claude Haiku 4.5   |
| `claude-opus-4-5`   | `claude-opus-4-5-20251101`        | Claude Opus 4.5    |
| `claude-sonnet-4-0` | `claude-sonnet-4-20250514`        | Claude Sonnet 4    |
| `claude-opus-4-0`   | `claude-opus-4-20250514`          | Claude Opus 4      |

### NOT working (404)

- `claude-opus-latest`
- `claude-sonnet-latest`
- `claude-haiku-latest`
- `claude-3-5-sonnet-latest`
- `claude-3-opus-latest`

There are **no** `*-latest` aliases. The alias pattern is `claude-{family}-{major}-{minor}` which resolves
to the latest snapshot within that generation (e.g., `claude-haiku-4-5` -> `claude-haiku-4-5-20251001`).

### Claude Code aliases

Claude Code supports short aliases (`sonnet`, `opus`, `haiku`) configured via environment variables:
- `ANTHROPIC_DEFAULT_SONNET_MODEL`
- `ANTHROPIC_DEFAULT_OPUS_MODEL`
- `ANTHROPIC_DEFAULT_HAIKU_MODEL`

These are Claude Code specific, not API-level.

## 4. Current Models (live API, 2026-04-07)

Ordered by recency (as returned by the API):

1. `claude-sonnet-4-6` -- 1M ctx, 128k out (2026-02-17)
2. `claude-opus-4-6` -- 1M ctx, 128k out (2026-02-04)
3. `claude-opus-4-5-20251101` -- 200k ctx, 64k out
4. `claude-haiku-4-5-20251001` -- 200k ctx, 64k out
5. `claude-sonnet-4-5-20250929` -- 1M ctx, 64k out
6. `claude-opus-4-1-20250805` -- 200k ctx, 32k out
7. `claude-opus-4-20250514` -- 200k ctx, 32k out
8. `claude-sonnet-4-20250514` -- 1M ctx, 64k out
9. `claude-3-haiku-20240307` -- 200k ctx, 4k out (deprecated, retiring 2026-04-19)
