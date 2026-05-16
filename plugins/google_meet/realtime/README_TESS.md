# Google Meet Plugin — Tess Voice Modification

## What was changed and why

The upstream `google_meet` plugin (v0.2.0, NousResearch) ships with a realtime
speaking mode that is **hardwired to OpenAI's Realtime API**.  This means
`meet_say("hello")` routes through:

    wss://api.openai.com/v1/realtime  →  PCM  →  speaker.pcm  →  paplay  →  Chrome

We replaced the OpenAI client with Tess's own voice stack:

    Groq (openai/gpt-oss-120b)  →  text response  →  ElevenLabs Heather (pcm_24000)  →  PCM
    →  speaker.pcm  →  paplay  →  Chrome

## Files

### Created (new)

| File | Purpose |
|------|---------|
| `realtime/tess_client.py` | Drop-in replacement for `openai_client.py`. Same API surface: `RealtimeSession` (aliased as `TessRealtimeSession`) and `RealtimeSpeaker` (aliased as `TessRealtimeSpeaker`). |

### Modified (existing)

| File | Change | Lines affected |
|------|--------|----------------|
| `meet_bot.py` | Swap one import: `openai_client` → `tess_client` with aliasing | ~1 line |
| (nothing else) | All other files unchanged | — |

## The one-line change in `meet_bot.py`

**Before** (line 287-290):
```python
from plugins.google_meet.realtime.openai_client import (
    RealtimeSession,
    RealtimeSpeaker,
)
```

**After**:
```python
from plugins.google_meet.realtime.tess_client import (
    TessRealtimeSession as RealtimeSession,
    TessRealtimeSpeaker as RealtimeSpeaker,
)
```

The aliasing (`as RealtimeSession`) means the 30+ references to `RealtimeSession`
and `RealtimeSpeaker` throughout `_start_realtime_speaker()` and the drain loop
(lines 667-687 in `meet_bot.py`) work without any further changes.

## How to revert

Restore the original import line (swap back to `openai_client`).  The
original `openai_client.py` is **not deleted** — it's left intact in
`realtime/openai_client.py` as the fallback.

## How `tess_client.py` works

### Architecture

```
meet_say("Hello team")
    │
    ▼
say_queue.jsonl          ←  enqueue_say() writes text line
    │
    ▼
TessRealtimeSpeaker      ←  polls queue, calls TessRealtimeSession.speak()
    │
    ▼
TessRealtimeSession.speak()
    ├─ _groq_chat()      ←  subprocess + curl (Groq blocks Python urllib)
    │   model: openai/gpt-oss-120b
    │   max_tokens: 256
    │
    ├─ _split_sentences() ←  sentence-boundary splitting
    │
    ├─ _elevenlabs_tts()  ←  per-sentence (urllib, works fine)
    │   voice: Heather (qufOgIh4LTqIKBQT7wWc)
    │   model: eleven_flash_v2_5
    │   format: pcm_24000 (24kHz s16le mono)
    │
    ├─ _check_budget()    ←  budget guard per sentence
    └─ _log_budget()      ←  records usage after successful TTS
    │
    ▼
raw PCM bytes appended to speaker.pcm
    │
    ▼
paplay reads growing file → PulseAudio null-sink → Chrome fake mic → Meet hears Tess
```

### Barge-in (interruption)

OpenAI Realtime: `response.cancel` sent over WebSocket — instant.
Tess client: `threading.Event` set, checked between sentences — best-effort.
Captions take ~500ms to appear, so like the upstream, Tess will talk over
the first ~1s of a human interruption.  Acceptable for v1.

### Credentials

Read from `~/.hermes/.env`:
- `GROQ_API_KEY` — for LLM text generation
- `ELEVENLABS_API_KEY` — for Heather TTS

No hardcoded keys.  Falls back if keys missing — `speak()` returns `ok: False`.

### Budget guard

Reuses the same `~/tess/tts_budget_guard.py` module as the voice service.
Every TTS sentence is checked.  If budget blocks mid-speech, remaining
sentences are dropped.

## Tests

Run from the plugin directory:

```bash
python3 -m pytest tests/test_tess_client.py -v
```

Or from the Hermes venv:

```bash
~/.hermes/hermes-agent/venv/bin/python3 -m pytest \
  ~/.hermes/hermes-agent/plugins/google_meet/tests/test_tess_client.py -v
```
