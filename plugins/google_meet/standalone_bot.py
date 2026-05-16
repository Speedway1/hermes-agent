#!/usr/bin/env python3
"""
Standalone Meet bot — FORKED FROM THE WORKING DIAGNOSTIC.
Joins + generates audio. No meet_bot.py dependency at all.

Usage (via process_manager env vars):
    HERMES_MEET_URL=https://meet.google.com/... \
    HERMES_MEET_OUT_DIR=/path/to/out \
    HERMES_MEET_AUTH_STATE=/path/to/auth.json \
    HERMES_MEET_MODE=realtime \
    python3 standalone_bot.py
"""
import os, sys, time, json, base64, signal, threading
from pathlib import Path

# ── Read env ────────────────────────────────────────────────────────────
url = os.environ.get("HERMES_MEET_URL", "").strip()
out_dir = Path(os.environ.get("HERMES_MEET_OUT_DIR", "/tmp/meet_bot"))
auth_state = os.environ.get("HERMES_MEET_AUTH_STATE", "").strip()
mode = os.environ.get("HERMES_MEET_MODE", "transcribe").strip()
guest_name = os.environ.get("HERMES_MEET_GUEST_NAME", "Tess (Hermes)")
duration_s = float(os.environ.get("HERMES_MEET_DURATION", "0")) or None

if not url or "meet.google.com" not in url:
    print("ERROR: HERMES_MEET_URL required", file=sys.stderr)
    sys.exit(2)

out_dir.mkdir(parents=True, exist_ok=True)
status_path = out_dir / "status.json"
queue_path = out_dir / "say_queue.jsonl"
processed_path = out_dir / "say_processed.jsonl"
wav_path = out_dir / "speaker.wav"

# ── Status file ─────────────────────────────────────────────────────────
_status = {
    "meetingId": url.split("/")[-1].split("?")[0],
    "url": url, "inCall": False, "captioning": False,
    "captionsEnabledAttempted": False, "lobbyWaiting": False,
    "joinAttemptedAt": None, "joinedAt": None, "lastCaptionAt": None,
    "transcriptLines": 0, "transcriptPath": str(out_dir / "transcript.txt"),
    "error": None, "exited": False, "pid": os.getpid(),
    "realtime": mode == "realtime", "realtimeReady": False,
    "realtimeDevice": "wav-file" if mode == "realtime" else None,
    "audioBytesOut": 0, "lastAudioOutAt": None, "lastBargeInAt": None,
    "leaveReason": None,
}
def flush():
    with open(status_path, "w") as f:
        json.dump(_status, f)

# ── Audio pipeline (lazy import) ────────────────────────────────────────
_stop_flag = {"stop": False}
_session = None
_speaker_thread = None

def start_speaker():
    global _session, _speaker_thread
    try:
        from plugins.google_meet.realtime.tess_client import (
            TessRealtimeSession, TessRealtimeSpeaker,
        )
        _session = TessRealtimeSession(audio_sink_path=wav_path)
        _session.connect()
        speaker = TessRealtimeSpeaker(
            session=_session, queue_path=queue_path,
            processed_path=processed_path,
        )
        def loop():
            try:
                speaker.run_until_stopped(lambda: _stop_flag["stop"])
            except Exception as e:
                _status["error"] = f"speaker crashed: {e}"
                flush()
        _speaker_thread = threading.Thread(target=loop, daemon=True)
        _speaker_thread.start()
        _status["realtimeReady"] = True
        flush()
        return True
    except Exception as e:
        _status["error"] = f"speaker init failed: {e}"
        flush()
        return False

# ── Seed silence WAV (exactly like diagnostic) ──────────────────────────
import wave as _wv
with open(wav_path, "wb") as f:
    with _wv.open(f, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000)
        wf.writeframes(b'\x00' * (48000 * 5 * 2))

# ── Audio injection JS ──────────────────────────────────────────────────
_INJECT_JS = """
window.__hermesInjectAudioBytes = async function(b64) {
  try {
    if (!window.__hermesAudioSender) {
      for (const k of Object.keys(window)) {
        try {
          const v = window[k];
          if (v && typeof v.getSenders === 'function') {
            const s = v.getSenders().find(s => s.track && s.track.kind === 'audio');
            if (s) { window.__hermesAudioSender = s; break; }
          }
        } catch(_) {}
      }
    }
    if (!window.__hermesAudioSender) return {ok: false, error: 'no sender'};
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const ctx = new AudioContext({sampleRate: 48000});
    const buf = await ctx.decodeAudioData(bytes.buffer);
    const src = ctx.createBufferSource(); src.buffer = buf;
    const dest = ctx.createMediaStreamDestination();
    src.connect(dest); src.start(0);
    await window.__hermesAudioSender.replaceTrack(dest.stream.getAudioTracks()[0]);
    return {ok: true, durationMs: Math.round(buf.duration * 1000)};
  } catch(e) { return {ok: false, error: String(e)}; }
};
"""

# ── Launch browser ─────────────────────────────────────────────────────
from playwright.sync_api import sync_playwright

signal.signal(signal.SIGTERM, lambda *_: _stop_flag.__setitem__("stop", True))
signal.signal(signal.SIGINT, lambda *_: _stop_flag.__setitem__("stop", True))

try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={wav_path}",
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ])
        ctx_args = {"viewport": {"width": 1280, "height": 800},
                     "permissions": ["microphone", "camera"]}
        if auth_state and Path(auth_state).is_file():
            ctx_args["storage_state"] = auth_state
        context = browser.new_context(**ctx_args)
        page = context.new_page()

        # ── Navigate ──
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(3000)

        # ── Guest name ──
        try:
            loc = page.locator('input[aria-label*="name" i]').first
            if loc.count() and loc.is_visible():
                loc.fill(guest_name, timeout=2000)
        except Exception:
            pass

        # ── Click Join ──
        clicked = False
        for label in ("Join now", "Ask to join"):
            try:
                btn = page.get_by_role("button", name=label, exact=False).first
                if btn.count() and btn.is_visible():
                    btn.click(timeout=3000)  # EXACTLY like diagnostic
                    clicked = True
                    if label == "Ask to join":
                        _status["lobbyWaiting"] = True
                    break
            except Exception:
                continue

        _status["joinAttemptedAt"] = time.time()
        flush()

        if not clicked:
            _status["error"] = "no join button found"
            _status["exited"] = True
            flush()
            browser.close()
            sys.exit(1)

        page.wait_for_timeout(5000)

        # ── Check result ──
        text = page.evaluate("() => document.body?.innerText || ''")
        if "You can't join" in text:
            _status["error"] = "host denied admission"
            _status["leaveReason"] = "denied"
            _status["exited"] = True
            flush()
            page.screenshot(path=str(out_dir / "denial_debug.png"))
            browser.close()
            sys.exit(1)
        elif "Leave call" in text or "Turn on captions" in text:
            _status["inCall"] = True
            _status["joinedAt"] = time.time()
            _status["lobbyWaiting"] = False
            flush()
        elif "Ask to join" in text or "Someone will let you in" in text:
            _status["lobbyWaiting"] = True
            flush()
            # Wait for admission
            deadline = time.time() + 300
            while time.time() < deadline and not _stop_flag["stop"]:
                time.sleep(2)
                t = page.evaluate("() => document.body?.innerText || ''")
                if "Leave call" in t:
                    _status["inCall"] = True
                    _status["joinedAt"] = time.time()
                    _status["lobbyWaiting"] = False
                    flush()
                    break
                elif "You can't join" in t:
                    _status["error"] = "host denied admission"
                    _status["leaveReason"] = "denied"
                    _status["exited"] = True
                    flush()
                    page.screenshot(path=str(out_dir / "denial_debug.png"))
                    browser.close()
                    sys.exit(1)

        if not _status["inCall"]:
            _status["error"] = "lobby timeout"
            _status["leaveReason"] = "lobby_timeout"
            _status["exited"] = True
            flush()
            browser.close()
            sys.exit(1)

        # ── Post-admission: install JS + start speaker ──
        page.evaluate(_INJECT_JS)
        if mode == "realtime":
            start_speaker()
        flush()

        # ── Main loop ──
        deadline = (time.time() + duration_s) if duration_s else None
        while not _stop_flag["stop"]:
            if deadline and time.time() > deadline:
                _status["leaveReason"] = "duration_expired"
                break

            # Audio injection
            if _session is not None and getattr(_session, "_audio_ready", False):
                try:
                    wav_bytes = wav_path.read_bytes()
                    b64 = base64.b64encode(wav_bytes).decode("ascii")
                    result = page.evaluate(
                        "(b64) => window.__hermesInjectAudioBytes(b64)", b64
                    )
                    if isinstance(result, dict) and result.get("ok"):
                        _status["lastAudioOutAt"] = time.time()
                    _session._audio_ready = False
                except Exception as e:
                    _status["error"] = f"audio inject failed: {e}"
                _status["audioBytesOut"] = getattr(_session, "audio_bytes_out", 0)
                flush()

            time.sleep(1.0)

        # Cleanup
        _stop_flag["stop"] = True
        if _speaker_thread:
            _speaker_thread.join(timeout=5)
        if _session:
            _session.close()
        _status["exited"] = True
        _status["inCall"] = False
        flush()
        context.close()
        browser.close()

except Exception as e:
    _status["error"] = f"unhandled: {e}"
    _status["exited"] = True
    flush()
    sys.exit(1)
