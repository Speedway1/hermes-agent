"""Headless Google Meet bot — Playwright + live-caption scraping.

Runs as a standalone subprocess spawned by ``process_manager.py``. Reads config
from env vars, writes status + transcript to files under
``$HERMES_HOME/workspace/meetings/<meeting-id>/``. The main hermes process
reads those files via the ``meet_*`` tools — no IPC beyond filesystem.

The scraping strategy mirrors OpenUtter (sumansid/openutter): we don't parse
WebRTC audio, we enable Google Meet's built-in live captions and observe the
captions container in the DOM via a MutationObserver. This is lossy and
English-biased but it is:

* deterministic (no API keys, no STT billing),
* works behind Meet's normal login / admission,
* survives Meet UI rewrites fairly well because the caption container has a
  stable ARIA role.

Run standalone for debugging::

    HERMES_MEET_URL=https://meet.google.com/abc-defg-hij \\
    HERMES_MEET_OUT_DIR=/tmp/meet-debug \\
    HERMES_MEET_HEADED=1 \\
    python -m plugins.google_meet.meet_bot

No meet.google.com URL → exits non-zero. Any URL that doesn't start with
``https://meet.google.com/`` is rejected (explicit-by-design).
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Match ``https://meet.google.com/abc-defg-hij`` or ``.../lookup/...`` — the
# short three-segment code or a lookup URL. Anything else is rejected.
MEET_URL_RE = re.compile(
    r"^https://meet\.google\.com/("
    r"[a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,}"
    r"|lookup/[^/?#]+"
    r"|new"
    r")(?:[/?#].*)?$"
)


# Filenames the bot reads/writes in ``HERMES_MEET_OUT_DIR``.
SAY_QUEUE_FILENAME = "say_queue.jsonl"


def _is_safe_meet_url(url: str) -> bool:
    """Return True if *url* is a Google Meet URL we're willing to navigate to."""
    if not isinstance(url, str):
        return False
    return bool(MEET_URL_RE.match(url.strip()))


def _meeting_id_from_url(url: str) -> str:
    """Extract the 3-segment meeting code from a Meet URL.

    For ``https://meet.google.com/abc-defg-hij`` → ``abc-defg-hij``.
    For ``.../lookup/<id>`` or ``/new`` we fall back to a timestamped id — the
    bot won't know the real code until after redirect, and callers pass this
    through to filename anyway.
    """
    m = re.search(
        r"meet\.google\.com/([a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,})",
        url or "",
    )
    if m:
        return m.group(1)
    return f"meet-{int(time.time())}"


# ---------------------------------------------------------------------------
# Status + transcript file writers
# ---------------------------------------------------------------------------

class _BotState:
    """Single-process mutable state, flushed to ``status.json`` on each change."""

    def __init__(self, out_dir: Path, meeting_id: str, url: str):
        self.out_dir = out_dir
        self.meeting_id = meeting_id
        self.url = url
        self.in_call = False
        self.captioning = False
        self.captions_enabled_attempted = False
        self.lobby_waiting = False
        self.join_attempted_at: Optional[float] = None
        self.joined_at: Optional[float] = None
        self.last_caption_at: Optional[float] = None
        self.transcript_lines = 0
        self.error: Optional[str] = None
        self.exited = False
        # v2 realtime fields.
        self.realtime = False
        self.realtime_ready = False
        self.realtime_device: Optional[str] = None
        self.audio_bytes_out: int = 0
        self.last_audio_out_at: Optional[float] = None
        self.last_barge_in_at: Optional[float] = None
        self.leave_reason: Optional[str] = None
        # Scraped captions, in order, deduped. Each entry is a dict of
        # {"ts": <epoch>, "speaker": str, "text": str}.
        self._seen: set = set()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = out_dir / "transcript.txt"
        self.status_path = out_dir / "status.json"
        self._flush()

    # -------- transcript ------------------------------------------------

    def record_caption(self, speaker: str, text: str) -> None:
        """Append a caption line if we haven't seen this exact (speaker, text)."""
        speaker = (speaker or "").strip() or "Unknown"
        text = (text or "").strip()
        if not text:
            return
        key = f"{speaker}|{text}"
        if key in self._seen:
            return
        self._seen.add(key)
        self.transcript_lines += 1
        self.last_caption_at = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(self.last_caption_at))
        line = f"[{ts}] {speaker}: {text}\n"
        # Atomic-ish append — good enough for a single-writer.
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(line)
        self._flush()

    # -------- status file ----------------------------------------------

    def _flush(self) -> None:
        data = {
            "meetingId": self.meeting_id,
            "url": self.url,
            "inCall": self.in_call,
            "captioning": self.captioning,
            "captionsEnabledAttempted": self.captions_enabled_attempted,
            "lobbyWaiting": self.lobby_waiting,
            "joinAttemptedAt": self.join_attempted_at,
            "joinedAt": self.joined_at,
            "lastCaptionAt": self.last_caption_at,
            "transcriptLines": self.transcript_lines,
            "transcriptPath": str(self.transcript_path),
            "error": self.error,
            "exited": self.exited,
            "pid": os.getpid(),
            # v2 realtime telemetry.
            "realtime": self.realtime,
            "realtimeReady": self.realtime_ready,
            "realtimeDevice": self.realtime_device,
            "audioBytesOut": self.audio_bytes_out,
            "lastAudioOutAt": self.last_audio_out_at,
            "lastBargeInAt": self.last_barge_in_at,
            "leaveReason": self.leave_reason,
        }
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.status_path)

    def set(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._flush()


# ---------------------------------------------------------------------------
# Playwright bot entry point
# ---------------------------------------------------------------------------

# JavaScript injected into the Meet tab to observe captions. Captures
# {speaker, text} tuples via a MutationObserver on the caption container,
# and exposes ``window.__hermesMeetDrain()`` to pull new entries. This
# mirrors the OpenUtter caption scraping approach.
_CAPTION_OBSERVER_JS = r"""
(() => {
  if (window.__hermesMeetInstalled) return;
  window.__hermesMeetInstalled = true;
  window.__hermesMeetQueue = [];

  const captionSelector = '[role="region"][aria-label*="aption" i], ' +
                          'div[jsname="YSxPC"], ' +  // legacy
                          'div[jsname="tgaKEf"]';    // current (Apr 2026)

  function pushEntry(speaker, text) {
    if (!text || !text.trim()) return;
    window.__hermesMeetQueue.push({
      ts: Date.now(),
      speaker: (speaker || '').trim(),
      text: text.trim(),
    });
  }

  function scan(root) {
    // Meet captions render as a list of rows; each row contains a speaker
    // label and a text block. Selectors vary across Meet rewrites; we try
    // a few shapes and fall back to raw text.
    const rows = root.querySelectorAll('div[jsname="dsyhDe"], div.CNusmb, div.TBMuR');
    if (rows.length) {
      rows.forEach((row) => {
        const spkEl = row.querySelector('div.KcIKyf, div.zs7s8d, span[jsname="YSxPC"]');
        const txtEl = row.querySelector('div.bh44bd, span[jsname="tgaKEf"], div.iTTPOb');
        const speaker = spkEl ? spkEl.innerText : '';
        const text = txtEl ? txtEl.innerText : row.innerText;
        pushEntry(speaker, text);
      });
      return;
    }
    // Fallback: treat the whole region's innerText as one anonymous line.
    const text = (root.innerText || '').split('\n').filter(Boolean).pop();
    pushEntry('', text);
  }

  function attach() {
    const el = document.querySelector(captionSelector);
    if (!el) return false;
    const obs = new MutationObserver(() => scan(el));
    obs.observe(el, { childList: true, subtree: true, characterData: true });
    scan(el);
    return true;
  }

  // Try now and retry on interval — the caption region only appears after
  // captions are enabled and someone speaks.
  if (!attach()) {
    const iv = setInterval(() => { if (attach()) clearInterval(iv); }, 1500);
  }

  window.__hermesMeetDrain = () => {
    const out = window.__hermesMeetQueue.slice();
    window.__hermesMeetQueue = [];
    return out;
  };
})();
"""


def _enable_captions_js() -> str:
    """Return a small JS snippet that tries to click the 'Turn on captions' button.

    Best-effort — Meet's caption toggle is keyboard-accessible via ``c``. We
    dispatch that keystroke as a cheap fallback. Real click targeting is too
    brittle to rely on.
    """
    return r"""
    (() => {
      const ev = new KeyboardEvent('keydown', {
        key: 'c', code: 'KeyC', keyCode: 67, which: 67, bubbles: true,
      });
      document.body.dispatchEvent(ev);
      return true;
    })();
    """


def _start_realtime_speaker(
    *,
    rt: dict,
    out_dir: Path,
    stop_flag: dict,
    state: "_BotState",
) -> None:
    """Wire up TessRealtimeSession + speaker thread + PCM pump (pacat via FIFO).

    Creates a named FIFO; pacat reads from it and streams to the PulseAudio sink.
    The speaker thread opens the FIFO and writes ElevenLabs PCM bytes directly —
    no file overwrites, no EOF issues. Kill pacat for barge-in.
    """
    try:
        from plugins.google_meet.realtime.tess_client import (
            TessRealtimeSession,
            TessRealtimeSpeaker,
        )
    except Exception as e:
        state.set(error=f"tess_client import failed: {e}")
        return

    import os as _os

    fifo_path = out_dir / "speaker.fifo"
    queue_path = out_dir / SAY_QUEUE_FILENAME
    processed_path = out_dir / "say_processed.jsonl"

    # Remove stale fifo and recreate
    if fifo_path.exists():
        fifo_path.unlink()
    _os.mkfifo(str(fifo_path))

    queue_path.touch()

    try:
        session = TessRealtimeSession(
            audio_sink_path=fifo_path,
            sample_rate=24000,
            pulse_sink_name=(rt.get("bridge_info") or {}).get("write_target", "tess_mic_sink"),
            pulse_source_name=(rt.get("bridge_info") or {}).get("device_name", "tess_mic_src"),
        )
        session.set_fifo_mode(True)
        session.connect()
    except Exception as e:
        state.set(error=f"realtime connect failed: {e}")
        return

    rt["session"] = session

    def _stop_fn():
        return stop_flag.get("stop", False)

    rt["speaker_stop"] = lambda: stop_flag.__setitem__("stop", True)

    speaker = TessRealtimeSpeaker(
        session=session,
        queue_path=queue_path,
        processed_path=processed_path,
    )

    def _speaker_loop():
        try:
            speaker.run_until_stopped(_stop_fn)
        except Exception as e:
            state.set(error=f"realtime speaker crashed: {e}")

    t_speaker = threading.Thread(target=_speaker_loop, name="meet-speaker", daemon=True)
    t_speaker.start()
    rt["speaker_thread"] = t_speaker

    # PCM pump: pacat reads from FIFO, streams to the null-sink.
    # pacat exits cleanly when the FIFO writer closes; we restart per utterance.
    # This avoids the file-overwrite/EOF race that plagued the paplay approach.
    import subprocess as _sp
    sink = (rt.get("bridge_info") or {}).get("write_target") or "tess_mic_sink"
    try:
        proc = _sp.Popen(
            [
                "pacat",
                "--playback",
                "--raw",
                "--rate=24000",
                "--format=s16le",
                "--channels=1",
                f"--device={sink}",
                str(fifo_path),
            ],
            stdin=_sp.DEVNULL,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
        )
        rt["pcm_pump"] = proc
    except FileNotFoundError:
        state.set(error="pacat not found — install pulseaudio-utils")


def run_bot() -> int:  # noqa: C901 — orchestration, explicit branches
    url = os.environ.get("HERMES_MEET_URL", "").strip()
    out_dir_env = os.environ.get("HERMES_MEET_OUT_DIR", "").strip()
    headed = os.environ.get("HERMES_MEET_HEADED", "").lower() in ("1", "true", "yes")
    auth_state = os.environ.get("HERMES_MEET_AUTH_STATE", "").strip()
    guest_name = os.environ.get("HERMES_MEET_GUEST_NAME", "Hermes Agent")
    duration_s = _parse_duration(os.environ.get("HERMES_MEET_DURATION", ""))
    # v2: optional realtime mode. Enabled when HERMES_MEET_MODE=realtime.
    mode = os.environ.get("HERMES_MEET_MODE", "transcribe").strip().lower()
    realtime_model = os.environ.get("HERMES_MEET_REALTIME_MODEL", "gpt-realtime")
    realtime_voice = os.environ.get("HERMES_MEET_REALTIME_VOICE", "alloy")
    realtime_instructions = os.environ.get("HERMES_MEET_REALTIME_INSTRUCTIONS", "")
    realtime_api_key = os.environ.get("HERMES_MEET_REALTIME_KEY") or os.environ.get("OPENAI_API_KEY", "")

    if not url or not _is_safe_meet_url(url):
        sys.stderr.write(
            "google_meet bot: refusing to launch — HERMES_MEET_URL must be a "
            "meet.google.com URL. got: %r\n" % url
        )
        return 2
    if not out_dir_env:
        sys.stderr.write("google_meet bot: HERMES_MEET_OUT_DIR is required\n")
        return 2

    out_dir = Path(out_dir_env)
    meeting_id = _meeting_id_from_url(url)
    state = _BotState(out_dir=out_dir, meeting_id=meeting_id, url=url)

    # SIGTERM → exit cleanly so the parent ``meet_leave`` gets a finalized
    # transcript. We set a flag instead of raising so the Playwright context
    # teardown runs in the finally block below.
    stop_flag = {"stop": False}

    def _on_signal(_sig, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # v2 realtime: Tess voice pipeline (Groq → ElevenLabs → PCM → paplay → PulseAudio).
    # Chrome reads from null-sink virtual source via PULSE_SOURCE.
    rt = {
        "enabled": mode == "realtime",
        "bridge": None,            # AudioBridge | None
        "bridge_info": None,       # dict | None
        "session": None,           # TessRealtimeSession | None
        "speaker_thread": None,    # threading.Thread | None
        "speaker_stop": None,      # callable | None
        "pcm_pump": None,          # subprocess.Popen | None
        "_post_admit_done": False, # whether post-admission setup ran
    }
    if rt["enabled"]:
        # Detect Groq + ElevenLabs keys from .env file.
        env_path = Path(os.environ.get("HERMES_ENV_FILE", "~/.hermes/.env")).expanduser()
        groq_key_found = False
        elevenlabs_key_found = False
        if env_path.exists():
            with open(env_path) as ef:
                for line in ef:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY=") and not line.startswith("#"):
                        groq_key_found = True
                    elif line.startswith("ELEVENLABS_API_KEY=") and not line.startswith("#"):
                        elevenlabs_key_found = True
        if groq_key_found and elevenlabs_key_found:
            try:
                from plugins.google_meet.audio_bridge import AudioBridge
                bridge = AudioBridge()
                rt["bridge_info"] = bridge.setup()
                rt["bridge"] = bridge
                state.set(realtime=True, realtime_device=rt["bridge_info"].get("device_name"))
            except Exception as e:
                state.set(error=f"audio bridge setup failed: {e} — falling back to transcribe")
                rt["enabled"] = False
        else:
            state.set(error="realtime mode requested but API keys not found — falling back to transcribe")
            rt["enabled"] = False

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        state.set(error=f"playwright not installed: {e}", exited=True)
        sys.stderr.write(
            "google_meet bot: playwright is not installed. Run "
            "`pip install playwright && python -m playwright install chromium`\n"
        )
        return 3

    # Chrome uses PulseAudio for real audio routing — NO fake device flags.
    # `PULSE_SOURCE` points at the virtual mic; `PULSE_SINK` routes Meet output.
    # `--use-fake-ui-for-media-stream` auto-accepts the mic/cam permission prompt
    # without bypassing the actual OS audio stack.
    chrome_env = os.environ.copy()
    # Clean Chrome args modelled on Vexa getAuthenticatedBrowserArgs() +
    # MeetBeats audio-processing bypass. NO aggressive flags
    # (--disable-web-security, --ignore-certificate-errors) — those trigger
    # Google bot detection on authenticated joins.
    # --disable-features=WebRtcApmInAudioService kills the WebRTC audio
    # processing module that classifies TTS as "non-speech noise".
    chrome_args = [
        "--use-fake-ui-for-media-stream",
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-features=AudioServiceAudioProcessing,WebRtcApmInAudioService",
    ]
    if rt["enabled"] and rt.get("bridge_info", {}).get("platform") == "linux":
        chrome_env["PULSE_SOURCE"] = rt["bridge_info"].get("device_name", "")

    try:
        with sync_playwright() as pw:
            # Pass PULSE_SOURCE via process env so Chrome inherits it.
            for k, v in chrome_env.items():
                os.environ[k] = v

            # Standard launch + storageState — the proven approach used by
            # recallai and our diagnostic script.  Cookies from auth.json are
            # loaded into a regular context.  auth.json must be fresh
            # (generated within the last few minutes by `hermes meet auth`).
            browser = pw.chromium.launch(
                headless=not headed,
                args=chrome_args,
            )
            context_options = {
                "viewport": {"width": 1280, "height": 800},
                "permissions": ["microphone", "camera"],
                "user_agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
                ),
            }
            auth_json = out_dir.parent / "auth.json"
            if auth_json.exists():
                context_options["storage_state"] = str(auth_json)
            context = browser.new_context(**context_options)
            page = context.new_page()

            # ── MeetBeats-style getUserMedia() override ──
            # Kills echo cancellation, noise suppression, auto-gain, and
            # Chrome-specific goog* audio processing.  Without this, Google
            # Meet's "Studio sound" will classify TTS as non-speech noise and
            # filter it out before it reaches other participants.
            # NOTE: the goog* flags are Chrome-only and harmless no-ops on
            # other platforms.  The standard echoCancellation:false is safe
            # for Meet but may cause echo on Teams — gated by hostname check.
            page.add_init_script("""
                (() => {
                  if (window.__hermesGUMOverridden) return;
                  window.__hermesGUMOverridden = true;
                  const orig = navigator.mediaDevices.getUserMedia.bind(
                    navigator.mediaDevices
                  );
                  // Google Meet: disable all processing to let TTS through.
                  // Non-Google platforms: keep standard echo cancellation.
                  const isGoogleMeet = window.location.hostname.includes(
                    'meet.google.com'
                  );
                  const noProcessing = isGoogleMeet ? {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    googEchoCancellation: false,
                    googAutoGainControl: false,
                    googNoiseSuppression: false,
                    googHighpassFilter: false,
                    googAudioMirroring: false,
                  } : {
                    // Keep echo cancellation for Teams/Zoom — disabling it
                    // can cause echo loops on those platforms.
                    noiseSuppression: false,
                    autoGainControl: false,
                  };
                  navigator.mediaDevices.getUserMedia = function (constraints) {
                    if (constraints && typeof constraints.audio === 'object') {
                      Object.assign(constraints.audio, noProcessing);
                    } else if (constraints && constraints.audio === true) {
                      constraints.audio = noProcessing;
                    }
                    return orig(constraints);
                  };
                  console.log('[Tess] getUserMedia override installed');
                })();
            """)


            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                state.set(error=f"navigate failed: {e}", exited=True)
                return 4

            # ── Vexa FM-001 fix: framenavigated handler ──
            # Registered AFTER initial navigation to avoid false-triggering
            # on the page.goto() itself.  When a Google Meet ends naturally,
            # Meet auto-navigates to its post-call page (/landing, "How was
            # your call?", etc.) destroying the Playwright execution context.
            # Without this handler, the bot misreports successful meetings as
            # crashes (~25% false failure rate in Vexa prod).
            def _on_framenavigated(frame):
                if frame != page.main_frame:
                    return
                url = frame.url
                if any(marker in url for marker in (
                    "/landing", "/_meet/", "meet.google.com/end",
                )):
                    state.set(leave_reason="meeting_ended_via_navigation")
                    stop_flag["stop"] = True
                    print("[meet_bot] post-meeting navigation detected — "
                          f"exiting gracefully: {url}", file=sys.stderr)

            page.on("framenavigated", _on_framenavigated)

            # ── end framenavigated handler ──

            # Guest-mode: Meet shows a name field before "Ask to join". When
            # we're authed, we instead see "Join now".
            _try_guest_name(page, guest_name)
            _click_join(page, state)

            # Record join attempt time and enter drain loop IMMEDIATELY.
            # NO JavaScript is injected yet — that only happens AFTER
            # _detect_admission() confirms we're past the lobby.
            state.set(join_attempted_at=time.time())
            # Note: in_call=False until admission is confirmed (we detect
            # either the Leave button or the caption region, signalling we
            # made it past the lobby). DO NOT set captioning=True yet.

            # Admission + drain loop. Runs until SIGTERM, duration expiry,
            # or the page detects "You were removed / you left the
            # meeting". Responsible for:
            #   * detecting admission (Leave button visible → in_call=True)
            #   * ONCE admitted: installing captions + audio injection JS,
            #     starting the realtime speaker
            #   * timing out stuck-in-lobby (default 5 minutes)
            #   * draining scraped captions into the transcript
            #   * triggering realtime barge-in when a human speaks while
            #     the bot is generating audio
            #   * periodically flushing realtime counters into status.json
            deadline = (time.time() + duration_s) if duration_s else None
            lobby_deadline = time.time() + float(
                os.environ.get("HERMES_MEET_LOBBY_TIMEOUT", "300")
            )
            last_admission_check = 0.0
            while not stop_flag["stop"]:
                now = time.time()
                if deadline and now > deadline:
                    state.set(leave_reason="duration_expired")
                    break

                # Admission detection every ~3s until admitted.
                if not state.in_call and (now - last_admission_check) > 3.0:
                    last_admission_check = now
                    admitted = _detect_admission(page)
                    if admitted:
                        state.set(
                            in_call=True,
                            lobby_waiting=False,
                            joined_at=now,
                        )
                        # ----- POST-ADMISSION SETUP (runs ONCE) -----
                        if not rt["_post_admit_done"]:
                            rt["_post_admit_done"] = True
                            # Install captions + observer JS
                            try:
                                page.evaluate(_enable_captions_js())
                                state.set(captions_enabled_attempted=True)
                            except Exception:
                                pass
                            try:
                                page.evaluate(_CAPTION_OBSERVER_JS)
                            except Exception as e:
                                state.set(error=f"caption observer install failed: {e}")
                            # Start realtime speaker if enabled
                            if rt["enabled"]:
                                _start_realtime_speaker(
                                    rt=rt,
                                    out_dir=out_dir,
                                    stop_flag=stop_flag,
                                    state=state,
                                )
                                if rt["session"] is not None:
                                    state.set(realtime_ready=True)
                            state.set(captioning=True)
                            print("[meet_bot] post-admission setup complete")
                        # ---- END POST-ADMISSION SETUP ----
                    elif now > lobby_deadline:
                        state.set(
                            error=(
                                "lobby timeout — host never admitted the bot "
                                f"within {int(lobby_deadline - state.join_attempted_at) if state.join_attempted_at else 0}s"
                            ),
                            leave_reason="lobby_timeout",
                        )
                        break
                    elif _detect_denied(page, state, out_dir):
                        state.set(
                            error="host denied admission",
                            leave_reason="denied",
                        )
                        break

                # Drain captions (only works after JS is installed).
                if state.in_call:
                    try:
                        queued = page.evaluate("window.__hermesMeetDrain && window.__hermesMeetDrain()")
                        if isinstance(queued, list):
                            for entry in queued:
                                if not isinstance(entry, dict):
                                    continue
                                speaker = str(entry.get("speaker", ""))
                                text = str(entry.get("text", ""))
                                state.record_caption(speaker=speaker, text=text)
                                # Barge-in: if the bot is currently generating
                                # audio AND a real human just spoke, cancel the
                                # in-flight response so we don't talk over them.
                                if rt["enabled"] and rt["session"] is not None:
                                    if _looks_like_human_speaker(speaker, guest_name):
                                        try:
                                            cancelled = rt["session"].cancel_response()
                                            if cancelled:
                                                state.set(last_barge_in_at=now)
                                        except Exception:
                                            pass
                    except Exception:
                        # Meet reloaded or we got booted — try to detect and
                        # exit gracefully rather than spinning.
                        if page.is_closed():
                            state.set(leave_reason="page_closed")
                            break

                # Fold the realtime session's byte/timestamp counters into
                # the status file so meet_status can surface them.
                if rt["session"] is not None:
                    state.set(
                        audio_bytes_out=getattr(rt["session"], "audio_bytes_out", 0),
                        last_audio_out_at=getattr(rt["session"], "last_audio_out_at", None),
                    )

                time.sleep(1.0)

            # Try to leave cleanly — click "Leave call" button if present.
            try:
                page.evaluate(
                    "() => { const b = document.querySelector('button[aria-label*=\"eave call\"]');"
                    " if (b) b.click(); }"
                )
            except Exception:
                pass

            context.close()
            try:
                browser.close()
            except Exception:
                pass
            # v2: teardown realtime speaker + pcm pump + audio bridge.
            if rt["speaker_stop"]:
                try:
                    rt["speaker_stop"]()
                except Exception:
                    pass
            if rt["speaker_thread"] is not None:
                try:
                    rt["speaker_thread"].join(timeout=5.0)
                except Exception:
                    pass
            if rt["pcm_pump"]:
                try:
                    rt["pcm_pump"].terminate()
                    rt["pcm_pump"].wait(timeout=3)
                except Exception:
                    pass
            # Remove the stale FIFO so the next session starts clean.
            fifo = out_dir / "speaker.fifo"
            try:
                fifo.unlink(missing_ok=True)
            except Exception:
                pass
            if rt["session"]:
                try:
                    rt["session"].close()
                except Exception:
                    pass
            if rt["bridge"]:
                try:
                    rt["bridge"].teardown()
                except Exception:
                    pass
            state.set(in_call=False, captioning=False, exited=True)
            return 0

    except Exception as e:
        state.set(error=f"unhandled: {e}", exited=True)
        return 1


def _try_guest_name(page, guest_name: str) -> None:
    """If Meet is showing a guest-name input, type *guest_name* into it."""
    try:
        # Meet's guest name input has placeholder "Your name".
        locator = page.locator('input[aria-label*="name" i]').first
        if locator.count() and locator.is_visible():
            locator.fill(guest_name, timeout=2_000)
    except Exception:
        pass


def _detect_admission(page) -> bool:
    """True if we're clearly past the lobby and in the call itself.

    Uses a JS-side probe because Meet's DOM structure varies by client
    version. We check several high-signal indicators and declare admission
    on the first hit:

      1. Leave-call button is present (``aria-label`` contains "eave call").
      2. Caption region has appeared (we installed the observer and it attached).
      3. The participant list container is visible.

    Conservative by default — returns False on any error.
    """
    probe = r"""
    (() => {
      const leave = document.querySelector('button[aria-label*="eave call" i]');
      if (leave) return true;
      if (window.__hermesMeetInstalled) {
        const caps = document.querySelector(
          '[role="region"][aria-label*="aption" i], ' +
          'div[jsname="YSxPC"], div[jsname="tgaKEf"]'
        );
        if (caps) return true;
      }
      const parts = document.querySelector('[aria-label*="articipants" i]');
      if (parts) return true;
      return false;
    })();
    """
    try:
        return bool(page.evaluate(probe))
    except Exception:
        return False


def _detect_denied(page, state: _BotState = None, out_dir: Path = None) -> bool:
    """True when Meet is showing a 'you were denied' / 'no one admitted' page.

    Includes an 8-second grace period after join_attempted_at to avoid
    false positives during the page transition. Captures a debug screenshot
    on denial for root-cause analysis.
    """
    # Grace period: refuse to check during the first 8 seconds after join.
    if state is not None and state.join_attempted_at is not None:
        if time.time() - state.join_attempted_at < 8.0:
            return False

    probe = r"""
    (() => {
      const text = document.body ? document.body.innerText || '' : '';
      // English only — matches what shows up when the host denies or
      // removes a guest.
      if (/You can't join this video call/i.test(text)) return true;
      if (/You were removed from the meeting/i.test(text)) return true;
      if (/No one responded to your request to join/i.test(text)) return true;
      return false;
    })();
    """
    try:
        result = bool(page.evaluate(probe))
        if result and out_dir is not None:
            # Capture debug screenshot on denial.
            try:
                page.screenshot(path=str(out_dir / "denial_debug.png"))
            except Exception:
                pass
        return result
    except Exception:
        return False


def _looks_like_human_speaker(speaker: str, bot_guest_name: str) -> bool:
    """Whether a caption line's speaker is probably a human, not our bot echo.

    Meet attributes captions to the speaker's display name. When Chrome is
    reading our fake mic, Meet still attributes captions to *our* bot name
    (because the bot is the one "speaking"). We don't want those to trigger
    barge-in. Anything else — real participant names — does.

    Conservative: unknown / blank speakers (common when caption scraping
    falls back to raw text) do NOT trigger barge-in, because we can't tell
    whether it was a human or us.
    """
    if not speaker or not speaker.strip():
        return False
    spk = speaker.strip().lower()
    if spk in ("unknown", "you", bot_guest_name.strip().lower()):
        return False
    return True


def _click_join(page, state: _BotState) -> None:
    """Click 'Join now' or 'Ask to join' if either button is visible.

    Flags ``lobby_waiting`` when we hit the "waiting for host to admit you"
    state so the agent can surface that in status.
    
    After clicking "Join now" (authenticated path), Google Meet may show a
    "Do you want people to hear you?" modal when headless Chrome has no
    speaker.  We dismiss it with "Continue without microphone".
    
    Uses standard click with timeout (NO no_wait_after) — exactly matching
    the working diagnostic script. Playwright waits for the page transition
    to complete before returning, so the DOM is settled when the drain loop
    starts checking admission.
    """
    for label in ("Join now", "Join now without camera", "Ask to join"):
        try:
            btn = page.get_by_role("button", name=label, exact=False).first
            if btn.count() and btn.is_visible():
                btn.click(timeout=3_000)
                if label == "Ask to join":
                    state.set(lobby_waiting=True)
                break
        except Exception:
            continue

    # Dismiss the "Do you want people to hear you?" modal that appears
    # when headless Chrome has no speaker device.
    try:
        cont = page.locator(
            'button:has-text("Continue without microphone")'
        ).first
        if cont.count() and cont.is_visible():
            cont.click(timeout=3_000)
            print("[meet_bot] dismissed 'Continue without microphone' modal")
    except Exception:
        pass

    time.sleep(2)


def _parse_duration(raw: str) -> Optional[float]:
    """Parse ``30m`` / ``2h`` / ``90`` (seconds) → float seconds, or None."""
    if not raw:
        return None
    raw = raw.strip().lower()
    try:
        if raw.endswith("h"):
            return float(raw[:-1]) * 3600
        if raw.endswith("m"):
            return float(raw[:-1]) * 60
        if raw.endswith("s"):
            return float(raw[:-1])
        return float(raw)
    except ValueError:
        return None


if __name__ == "__main__":  # pragma: no cover — subprocess entry point
    sys.exit(run_bot())
