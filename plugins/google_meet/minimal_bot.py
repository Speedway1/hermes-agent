"""
Minimal meet bot — mirrors the proven diagnostic flow.
NO JS injection during page transition. Audio only after admission.
"""
import json, os, sys, time, threading, base64, signal
from pathlib import Path
from playwright.sync_api import sync_playwright

# ── Re-use existing functions from meet_bot ──
from plugins.google_meet.meet_bot import (
    _BotState, _is_safe_meet_url, _meeting_id_from_url, _parse_duration,
    _try_guest_name, _click_join, _detect_admission, _detect_denied,
    _enable_captions_js, _CAPTION_OBSERVER_JS, _INJECT_AUDIO_FN_JS,
    _start_realtime_speaker,
)

SAY_QUEUE_FILENAME = "say_queue.jsonl"

def run_minimal() -> int:
    url = os.environ.get("HERMES_MEET_URL", "").strip()
    out_dir_env = os.environ.get("HERMES_MEET_OUT_DIR", "").strip()
    headed = os.environ.get("HERMES_MEET_HEADED", "").lower() in ("1", "true", "yes")
    auth_state = os.environ.get("HERMES_MEET_AUTH_STATE", "").strip()
    guest_name = os.environ.get("HERMES_MEET_GUEST_NAME", "Hermes Agent")
    duration_s = _parse_duration(os.environ.get("HERMES_MEET_DURATION", ""))
    mode = os.environ.get("HERMES_MEET_MODE", "transcribe").strip().lower()
    realtime_model = os.environ.get("HERMES_MEET_REALTIME_MODEL", "gpt-realtime")
    realtime_voice = os.environ.get("HERMES_MEET_REALTIME_VOICE", "alloy")
    realtime_instructions = os.environ.get("HERMES_MEET_REALTIME_INSTRUCTIONS", "")
    realtime_api_key = os.environ.get("HERMES_MEET_REALTIME_KEY") or os.environ.get("OPENAI_API_KEY", "")

    if not _is_safe_meet_url(url) or not out_dir_env:
        return 2

    out_dir = Path(out_dir_env)
    state = _BotState(out_dir=out_dir, meeting_id=_meeting_id_from_url(url), url=url)
    stop_flag = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: stop_flag.__setitem__("stop", True))
    signal.signal(signal.SIGINT, lambda *_: stop_flag.__setitem__("stop", True))

    rt = {"enabled": mode == "realtime", "session": None, "speaker_thread": None}
    if rt["enabled"]:
        env_path = Path(os.environ.get("HERMES_ENV_FILE", "~/.hermes/.env")).expanduser()
        if env_path.exists():
            env_text = env_path.read_text()
            has_groq = any(l.strip().startswith("GROQ_API_KEY=") and not l.strip().startswith("#") for l in env_text.splitlines())
            has_11 = any(l.strip().startswith("ELEVENLABS_API_KEY=") and not l.strip().startswith("#") for l in env_text.splitlines())
            if has_groq and has_11:
                realtime_api_key = "tess_client"
                state.set(realtime=True, realtime_device="wav-file")
            else:
                rt["enabled"] = False

    chrome_args = [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        "--disable-blink-features=AutomationControlled",
    ]

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not headed, args=chrome_args)
            ctx_args = {
                "viewport": {"width": 1280, "height": 800},
                "permissions": ["microphone", "camera"],
            }
            if auth_state and Path(auth_state).is_file():
                ctx_args["storage_state"] = auth_state
            context = browser.new_context(**ctx_args)
            page = context.new_page()

            # ── Step 1: Navigate (exactly like diagnostic) ──
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            time.sleep(1)

            # ── Step 2: Click Join ──
            _try_guest_name(page, guest_name)
            _click_join(page, state)
            state.set(join_attempted_at=time.time())

            # ── Step 3: Wait for admission (minimal loop, like diagnostic) ──
            deadline = (time.time() + duration_s) if duration_s else None
            lobby_deadline = time.time() + float(os.environ.get("HERMES_MEET_LOBBY_TIMEOUT", "300"))
            admitted = False

            while not stop_flag["stop"] and not admitted:
                now = time.time()
                if deadline and now > deadline:
                    state.set(leave_reason="duration_expired", exited=True)
                    return 0
                if now > lobby_deadline:
                    state.set(error="lobby timeout", leave_reason="lobby_timeout", exited=True)
                    return 0

                if _detect_admission(page):
                    admitted = True
                    state.set(in_call=True, joined_at=now, lobby_waiting=False)
                elif _detect_denied(page, state, out_dir):
                    state.set(error="host denied admission", leave_reason="denied", exited=True)
                    return 0

                # Write status every loop
                state.set()  # flush
                time.sleep(1.5)

            if not admitted:
                return 0

            # ── Step 4: NOW install observers + audio (post-admission) ──
            try:
                page.evaluate(_enable_captions_js())
                state.set(captions_enabled_attempted=True)
            except Exception:
                pass
            try:
                page.evaluate(_CAPTION_OBSERVER_JS)
            except Exception:
                pass
            try:
                page.evaluate(_INJECT_AUDIO_FN_JS)
            except Exception:
                pass
            state.set(captioning=True)

            if rt["enabled"]:
                _start_realtime_speaker(
                    rt=rt, out_dir=out_dir, api_key=realtime_api_key,
                    model=realtime_model, voice=realtime_voice,
                    instructions=realtime_instructions, stop_flag=stop_flag,
                    state=state, page=page,
                )
                if rt["session"] is not None:
                    state.set(realtime_ready=True)

            # ── Step 5: Main drain loop ──
            while not stop_flag["stop"]:
                now = time.time()
                if deadline and now > deadline:
                    state.set(leave_reason="duration_expired")
                    break

                # Drain captions
                try:
                    queued = page.evaluate("window.__hermesMeetDrain && window.__hermesMeetDrain()")
                    if isinstance(queued, list):
                        for entry in queued:
                            if isinstance(entry, dict):
                                state.record_caption(
                                    speaker=str(entry.get("speaker", "")),
                                    text=str(entry.get("text", ""))
                                )
                except Exception:
                    if page.is_closed():
                        state.set(leave_reason="page_closed")
                        break

                # Audio injection
                if rt["session"] is not None:
                    state.set(
                        audio_bytes_out=getattr(rt["session"], "audio_bytes_out", 0),
                        last_audio_out_at=getattr(rt["session"], "last_audio_out_at", None),
                    )
                    ses = rt["session"]
                    if getattr(ses, "_audio_ready", False):
                        wav_file = out_dir / "speaker.wav"
                        if wav_file.exists():
                            try:
                                wav_bytes = wav_file.read_bytes()
                                b64 = base64.b64encode(wav_bytes).decode("ascii")
                                result = page.evaluate(
                                    "(b64) => window.__hermesInjectAudioBytes(b64)", b64
                                )
                                if isinstance(result, dict) and result.get("ok"):
                                    state.set(last_audio_out_at=time.time())
                            except Exception as e:
                                state.set(error=f"audio inject failed: {e}")
                        ses._audio_ready = False

                time.sleep(1.0)

            # Cleanup
            try:
                context.close()
                browser.close()
            except Exception:
                pass
            stop_flag["stop"] = True
            if rt["speaker_thread"] is not None:
                try:
                    rt["speaker_thread"].join(timeout=5.0)
                except Exception:
                    pass
            if rt["session"]:
                try:
                    rt["session"].close()
                except Exception:
                    pass
            state.set(in_call=False, captioning=False, exited=True)
            return 0

    except Exception as e:
        state.set(error=f"unhandled: {e}", exited=True)
        return 1

if __name__ == "__main__":
    sys.exit(run_minimal())
