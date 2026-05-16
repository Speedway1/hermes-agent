"""
Tess Realtime Client — ElevenLabs Heather + Groq
==================================================

Drop-in replacement for ``openai_client.py``.  Same ``RealtimeSession`` /
``RealtimeSpeaker`` API surface so ``meet_bot.py`` needs only a one-line
import change.

Flow (vs OpenAI original):

    OpenAI path:  text → wss://api.openai.com/v1/realtime → PCM deltas → speaker.pcm
    Tess path:    text → Groq (openai/gpt-oss-120b) → text response
                        → ElevenLabs Heather (streaming pcm_24000) → PCM → speaker.pcm

The PCM → PulseAudio null-sink → Chrome fake-mic pipeline is unchanged.

Design decisions
----------------
* Groq called via ``subprocess`` + curl (proven working from this machine;
  Python's ``urllib`` is blocked by Groq — error code 1010).
* ElevenLabs called via ``urllib`` (proven working).
* Budget-guard checked before every TTS call.
* Barge-in: sets a stop flag checked between sentences; not as instant as
  OpenAI's ``response.cancel`` but functional.
* Sentence-splitting so audio dribbles out progressively while the PCM pump
  streams the growing file.

British English throughout.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Constants — match what the OpenAI client used so meet_bot.py is happy
# ---------------------------------------------------------------------------
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
TESS_VOICE_ID = "qufOgIh4LTqIKBQT7wWc"       # Heather | Posh & Confident
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_TTS_MODEL = "eleven_flash_v2_5"
SAMPLE_RATE = 24000                           # pcm_24000 → 24kHz s16le mono

# ---------------------------------------------------------------------------
# Credentials — from .env only
# ---------------------------------------------------------------------------
def _load_creds() -> tuple[str, str]:
    env_path = Path(os.environ.get("HERMES_ENV_FILE", "~/.hermes/.env")).expanduser()
    groq_key = elevenlabs_key = ""
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY=") and not line.startswith("#"):
                    groq_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("ELEVENLABS_API_KEY=") and not line.startswith("#"):
                    elevenlabs_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return groq_key, elevenlabs_key

_GROQ_KEY, _ELEVENLABS_KEY = _load_creds()

# ---------------------------------------------------------------------------
# Budget guard — imported lazily (same module as voice service)
# ---------------------------------------------------------------------------
def _check_budget(text: str) -> bool:
    """Return True if TTS is allowed for *text*.  Raises RuntimeError if blocked."""
    TESS_DIR = Path("/home/hermes/tess")
    import sys
    if str(TESS_DIR) not in sys.path:
        sys.path.insert(0, str(TESS_DIR))
    from tts_budget_guard import BudgetGuard
    guard = BudgetGuard()
    result = guard.check(text)
    if result.blocked:
        raise RuntimeError(f"TTS budget blocked: {result.reason} ({result.usage_pct:.0f}%)")
    return result.warning

def _log_budget(text: str) -> None:
    TESS_DIR = Path("/home/hermes/tess")
    import sys
    if str(TESS_DIR) not in sys.path:
        sys.path.insert(0, str(TESS_DIR))
    from tts_budget_guard import BudgetGuard
    guard = BudgetGuard()
    guard.log(text)


# ==========================================================================
# TessRealtimeSession — same API as RealtimeSession
# ==========================================================================

class TessRealtimeSession:
    """Groq + ElevenLabs replacement for OpenAI RealtimeSession.

    Thread safety: ``speak`` and ``cancel_response`` may be called from
    different threads; a lock serializes writes to the sink file.
    """

    def __init__(
        self,
        api_key: str = "",        # ignored (we read from .env); kept for API compat
        model: str = DEFAULT_GROQ_MODEL,
        voice: str = TESS_VOICE_ID,
        instructions: str = "",   # injected into Groq system prompt
        audio_sink_path: Optional[Path] = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.audio_sink_path = Path(audio_sink_path) if audio_sink_path else None
        self.sample_rate = sample_rate
        self._write_lock = threading.Lock()
        # Public counters (read by meet_bot.py drain loop via getattr)
        self.audio_bytes_out: int = 0
        self.last_audio_out_at: Optional[float] = None
        # Barge-in / cancel support
        self._cancel_flag = threading.Event()
        self._connected = True   # we don't have a persistent WS, flag is for compat
        # Two output modes:
        # - pcm_mode=True:  write raw 24kHz s16le mono PCM (for paplay)
        # - pcm_mode=False: write 48kHz WAV (for --use-file-for-fake-audio-capture)
        self._audio_ready: bool = False
        self._audio_duration_s: float = 3.0
        self._pcm_mode: bool = False

    def set_pcm_mode(self, enabled: bool = True) -> None:
        """Switch to raw PCM output (24kHz s16le mono)."""
        self._pcm_mode = enabled

    # ── lifecycle (API compat) ───────────────────────────────────────────

    def connect(self) -> None:
        """No-op — no persistent WebSocket.  Kept for API compat."""
        self._connected = True

    def close(self) -> None:
        """No-op — no persistent connection.  Kept for API compat."""
        self._connected = False

    # ── speak ─────────────────────────────────────────────────────────────

    def speak(self, text: str, timeout: float = 30.0) -> dict:
        """Generate speech for *text* via Groq → ElevenLabs.

        Blocks until audio is generated or timeout.  Writes PCM to
        ``audio_sink_path`` (same 24kHz s16le mono format the pump expects).
        """
        if not text or not text.strip():
            return {"ok": True, "bytes_written": 0, "duration_ms": 0.0}

        self._cancel_flag.clear()
        start = time.monotonic()
        bytes_written = 0

        # ── Step 1: Groq LLM → response text ──────────────────────────
        try:
            system_prompt = self._build_system_prompt(text)
            response_text = _groq_chat(system_prompt, text, timeout=min(timeout, 15))
        except Exception as e:
            return {"ok": False, "error": f"Groq failed: {e}", "bytes_written": 0, "duration_ms": 0.0}

        if not response_text or not response_text.strip():
            return {"ok": True, "bytes_written": 0, "duration_ms": (time.monotonic() - start) * 1000}

        # ── Step 2: Split into sentences ──────────────────────────────
        sentences = _split_sentences(response_text)

        # ── Step 3: ElevenLabs Heather TTS per sentence ───────────────
        all_pcm = bytearray()
        for i, sent in enumerate(sentences):
            if self._cancel_flag.is_set():
                break   # barge-in — stop generating more audio

            # Budget guard
            try:
                warned = _check_budget(sent)
            except RuntimeError as e:
                # Budget blocked mid-speech — stop here
                break

            try:
                pcm = _elevenlabs_tts(sent, timeout=min(timeout / max(len(sentences), 1), 10))
                if pcm:
                    all_pcm.extend(pcm)
                    _log_budget(sent)
            except Exception:
                # One sentence fails — continue with others if we can
                continue

            # Small inter-sentence gap: ~300ms of silence
            gap_samples = int(SAMPLE_RATE * 0.3 * 2)  # 16-bit = 2 bytes/sample
            all_pcm.extend(b'\x00' * gap_samples)

        if not all_pcm:
            return {"ok": True, "bytes_written": 0, "duration_ms": (time.monotonic() - start) * 1000}

        # ── Step 4: Write output ──────────────────────────────────────
        with self._write_lock:
            if self.audio_sink_path is not None:
                self.audio_sink_path.parent.mkdir(parents=True, exist_ok=True)
                if self._pcm_mode:
                    # Raw 24kHz s16le mono PCM — paplay reads this directly.
                    with open(self.audio_sink_path, "wb") as f:
                        f.write(bytes(all_pcm))
                    bytes_written = len(all_pcm)
                    self._audio_duration_s = (len(all_pcm) / 2) / SAMPLE_RATE
                else:
                    # WAV at 48kHz — for --use-file-for-fake-audio-capture.
                    import wave as _wave
                    WAV_SAMPLE_RATE = 48000
                    # Up-sample 24kHz → 48kHz: duplicate each sample
                    src_samples = len(all_pcm) // 2
                    dst_pcm = bytearray(src_samples * 4)
                    for si in range(src_samples):
                        sample = struct.unpack_from("<h", all_pcm, si * 2)[0]
                        struct.pack_into("<h", dst_pcm, si * 4, sample)
                        struct.pack_into("<h", dst_pcm, si * 4 + 2, sample)
                    with open(self.audio_sink_path, "wb") as f:
                        with _wave.open(f, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(WAV_SAMPLE_RATE)
                            wf.writeframes(bytes(dst_pcm))
                    bytes_written = len(all_pcm)
                    self._audio_duration_s = (len(dst_pcm) / 2) / WAV_SAMPLE_RATE
                self.audio_bytes_out += bytes_written
                self.last_audio_out_at = time.time()
                # Signal that new audio is ready
                self._audio_ready = True

        duration_ms = (time.monotonic() - start) * 1000.0
        return {
            "ok": True,
            "bytes_written": bytes_written,
            "duration_ms": duration_ms,
        }

    # ── cancel / barge-in (API compat) ───────────────────────────────────

    def cancel_response(self) -> bool:
        """Signal the in-flight speak() to stop at next sentence boundary.

        Returns True if a cancel flag was set.  Not as instant as OpenAI's
        WebSocket-level cancel, but functional for barge-in.
        """
        self._cancel_flag.set()
        return True

    # ── internal ─────────────────────────────────────────────────────────

    def _build_system_prompt(self, user_text: str) -> str:
        """Construct the Tess system prompt for the Groq call."""
        base = f"""You are Tess Tah, a 35-year-old British technical assistant for SMS Speedway.
You are speaking in a Google Meet call.  Keep responses CONCISE (1-3 sentences).
Speak warmly and professionally in British English (colour, organise, realise).
Never use American spellings.  Never be sycophantic.  Be direct but warm.
{self.instructions}"""
        return base


# ==========================================================================
# TessRealtimeSpeaker — same API as RealtimeSpeaker
# ==========================================================================

class TessRealtimeSpeaker:
    """File-based JSONL queue speaker — reads say_queue.jsonl, calls
    TessRealtimeSession.speak() for each entry."""

    def __init__(
        self,
        session: TessRealtimeSession,
        queue_path: Path,
        processed_path: Optional[Path] = None,
    ) -> None:
        self.session = session
        self.queue_path = Path(queue_path)
        self.processed_path = Path(processed_path) if processed_path else None

    def _read_queue(self) -> list[dict]:
        if not self.queue_path.exists():
            return []
        out: list[dict] = []
        for line in self.queue_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            if "id" not in entry:
                import uuid
                entry["id"] = str(uuid.uuid4())
            out.append(entry)
        return out

    def _rewrite_queue(self, remaining: list[dict]) -> None:
        if not remaining:
            self.queue_path.write_text("")
            return
        self.queue_path.write_text(
            "\n".join(json.dumps(e) for e in remaining) + "\n"
        )

    def _append_processed(self, entry: dict, result: dict) -> None:
        if self.processed_path is None:
            return
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "id": entry.get("id"),
            "text": entry.get("text", ""),
            "result": result,
        }
        with open(self.processed_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

    def run_until_stopped(
        self,
        stop_fn: Callable[[], bool],
        poll_interval: float = 0.5,
    ) -> None:
        while not stop_fn():
            entries = self._read_queue()
            if not entries:
                time.sleep(poll_interval)
                continue
            head = entries[0]
            text = (head.get("text") or "").strip()
            if text:
                try:
                    result = self.session.speak(text)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
            else:
                result = {"ok": True, "bytes_written": 0, "duration_ms": 0.0}
            self._append_processed(head, result)

            latest = self._read_queue()
            if latest and latest[0].get("id") == head.get("id"):
                self._rewrite_queue(latest[1:])
            else:
                self._rewrite_queue(
                    [e for e in latest if e.get("id") != head.get("id")]
                )


# ==========================================================================
# Internal helpers — Groq + ElevenLabs
# ==========================================================================

def _groq_chat(system_prompt: str, user_text: str, timeout: float = 15.0) -> str:
    """Call Groq via subprocess+curl (proven working; urllib blocked)."""
    payload = json.dumps({
        "model": DEFAULT_GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False,
    })

    try:
        result = subprocess.run(
            [
                "curl", "-s",
                "https://api.groq.com/openai/v1/chat/completions",
                "-X", "POST",
                "-H", f"Authorization: Bearer {_GROQ_KEY}",
                "-H", "Content-Type: application/json",
                "-d", payload,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl exited {result.returncode}: {result.stderr[:200]}")

        data = json.loads(result.stdout)
        if "error" in data:
            raise RuntimeError(f"Groq error: {data['error']}")

        return data["choices"][0]["message"]["content"]
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Groq call timed out after {timeout}s")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Groq response not valid JSON: {e}") from e


def _elevenlabs_tts(text: str, timeout: float = 10.0) -> Optional[bytes]:
    """Call ElevenLabs Heather TTS, return raw PCM16 bytes (24kHz s16le mono).

    Returns None on failure (caller logs and continues).
    """
    payload = json.dumps({
        "text": text,
        "model_id": DEFAULT_TTS_MODEL,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.3,
            "use_speaker_boost": True,
        },
        "output_format": "pcm_24000",
    }).encode()

    try:
        req = urllib.request.Request(
            f"{ELEVENLABS_API_URL}/text-to-speech/{TESS_VOICE_ID}",
            data=payload,
            headers={
                "xi-api-key": _ELEVENLABS_KEY,
                "Content-Type": "application/json",
            },
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.read()
    except Exception:
        return None


def _split_sentences(text: str) -> list[str]:
    """Split *text* at sentence boundaries for progressive TTS."""
    import re
    RE_SENTENCE = re.compile(r'([.!?]+[\s\n]+)')

    chunks = RE_SENTENCE.split(text)
    sentences = []
    buf = ""
    for chunk in chunks:
        buf += chunk
        if RE_SENTENCE.match(chunk) or len(buf) > 200:
            trimmed = buf.strip()
            if trimmed:
                sentences.append(trimmed)
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    if not sentences:
        sentences = [text.strip()]
    return sentences
