"""Virtual audio bridge for feeding generated speech into Chrome's mic.

v3 module. Uses MeetBeats-proven patterns:
- module-remap-source (NOT module-virtual-source — PipeWire renames it)
- TWO sinks: one for TTS injection, one for Chrome speaker output (echo loop fix)
- pactl set-default-source + set-default-sink (not just PULSE_SOURCE env var)
- Max volumes on all pipeline nodes

Linux (primary): uses pactl (PulseAudio) to create:
  1. hermes_meet_sink   — null-sink, 48kHz stereo (TTS target)
  2. chrome_output      — null-sink (Chrome speaker sink, prevents echo loop)
  3. hermes_meet_src    — remap-source from hermes_meet_sink.monitor (virtual mic)

Audio pipeline:
  paplay/ffmpeg → hermes_meet_sink → monitor → hermes_meet_src → Chrome mic → WebRTC
  Meeting audio → chrome_output (kept separate, not captured by monitor)

macOS: requires BlackHole 2ch to be installed.
Windows: not supported in v3.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional


_BLACKHOLE_DEVICE = "BlackHole 2ch"


class AudioBridge:
    """Manages virtual audio devices for Chrome fake-mic input.

    Call ``setup()`` once before launching the Meet bot and
    ``teardown()`` when the session ends. ``teardown()`` is idempotent.

    Creates two sinks and one remap source:
    - hermes_meet_sink: where TTS audio is written
    - chrome_output: separate sink for Chrome's received audio (prevents echo loop)
    - hermes_meet_src: virtual mic that captures ONLY hermes_meet_sink.monitor
    """

    SINK_NAME = "hermes_meet_sink"
    CHROME_OUTPUT_NAME = "chrome_output"
    SOURCE_NAME = "hermes_meet_src"

    def __init__(self) -> None:
        self._platform: Optional[str] = None
        self._device_name: Optional[str] = None
        self._write_target: Optional[str] = None
        self._module_ids: list[int] = []
        self._torn_down = False

    # ── public properties ─────────────────────────────────────────────────

    @property
    def device_name(self) -> str:
        """The virtual mic source name for PULSE_SOURCE and set-default-source."""
        if not self._device_name:
            raise RuntimeError("AudioBridge not set up yet")
        return self._device_name

    @property
    def write_target(self) -> str:
        """The sink name where TTS audio should be written."""
        if not self._write_target:
            raise RuntimeError("AudioBridge not set up yet")
        return self._write_target

    @property
    def chrome_sink(self) -> str:
        """The sink name for Chrome's speaker output (prevents echo loop)."""
        return self.CHROME_OUTPUT_NAME

    # ── lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> dict:
        """Provision all virtual audio devices.

        Returns a dict with device info. Raises RuntimeError on failure.
        """
        system = platform.system()
        if system == "Linux":
            return self._setup_linux()
        if system == "Darwin":
            return self._setup_darwin()
        if system == "Windows":
            raise RuntimeError("windows not supported in v3")
        raise RuntimeError(f"unsupported platform: {system}")

    def teardown(self) -> None:
        """Release all virtual audio devices. Idempotent."""
        if self._torn_down:
            return
        if self._platform == "linux" and self._module_ids:
            # Unload in reverse order.
            for mod_id in reversed(self._module_ids):
                try:
                    subprocess.run(
                        ["pactl", "unload-module", str(mod_id)],
                        check=False,
                        capture_output=True,
                    )
                except Exception:
                    pass
            self._module_ids = []
        self._torn_down = True

    # ── platform impls ────────────────────────────────────────────────────

    def _setup_linux(self) -> dict:
        module_ids: list[int] = []

        # 1. TTS sink — where paplay/ffmpeg writes audio.
        #    48kHz to match typical audio and avoid resampling artifacts.
        try:
            sink_out = subprocess.run(
                [
                    "pactl", "load-module", "module-null-sink",
                    f"sink_name={self.SINK_NAME}",
                    "sink_properties=device.description=HermesMeetSink",
                    "rate=48000",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "pactl not found — install PulseAudio/pipewire-pulse"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"pactl load-module null-sink failed: {exc.stderr or exc}"
            ) from exc

        sink_mod_id = self._parse_module_id(sink_out.stdout)
        module_ids.append(sink_mod_id)

        # 2. Chrome output sink — SEPARATE sink so meeting audio doesn't
        #    feed back into the monitor and trip WebRTC echo cancellation.
        try:
            chrome_out = subprocess.run(
                [
                    "pactl", "load-module", "module-null-sink",
                    f"sink_name={self.CHROME_OUTPUT_NAME}",
                    "sink_properties=device.description=Chrome_Output",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            self._rollback(module_ids)
            raise RuntimeError(
                f"pactl load-module chrome_output failed: {exc.stderr or exc}"
            ) from exc

        chrome_mod_id = self._parse_module_id(chrome_out.stdout)
        module_ids.append(chrome_mod_id)

        # 3. Use the null-sink's .monitor directly as our virtual mic.
        #    On PipeWire (pulse-server), neither module-virtual-source nor
        #    module-remap-source reliably auto-links monitor → source.
        #    The .monitor IS a valid PulseAudio source — Chrome reads from
        #    it via PULSE_SOURCE or set-default-source just fine.
        #    We keep the separate chrome_output sink to prevent echo loops.
        monitor_name = f"{self.SINK_NAME}.monitor"

        # Set defaults — Chrome uses these when env vars aren't propagated.
        subprocess.run(
            ["pactl", "set-default-sink", self.CHROME_OUTPUT_NAME],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["pactl", "set-default-source", monitor_name],
            check=False, capture_output=True,
        )

        # Max out volumes — compensate for WebRTC attenuation.
        self._max_volume(self.SINK_NAME, "sink")
        self._max_volume(monitor_name, "source")

        self._platform = "linux"
        self._device_name = monitor_name
        self._write_target = self.SINK_NAME
        self._module_ids = module_ids
        self._torn_down = False

        return {
            "platform": "linux",
            "device_name": monitor_name,
            "sample_rate": 48000,
            "channels": 2,
            "format": "float32le",
            "module_ids": list(self._module_ids),
            "write_target": self.SINK_NAME,
            "chrome_sink": self.CHROME_OUTPUT_NAME,
        }

    def _setup_darwin(self) -> dict:
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPAudioDataType"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "system_profiler not found (macOS-only command)"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"system_profiler failed: {exc.output}"
            ) from exc

        if "BlackHole" not in out:
            raise RuntimeError(
                "BlackHole virtual audio device not installed. "
                "Install via: brew install blackhole-2ch"
            )

        self._platform = "darwin"
        self._device_name = _BLACKHOLE_DEVICE
        self._write_target = _BLACKHOLE_DEVICE
        self._module_ids = []
        self._torn_down = False

        return {
            "platform": "darwin",
            "device_name": _BLACKHOLE_DEVICE,
            "sample_rate": 48000,
            "channels": 2,
            "format": "float32le",
            "module_ids": [],
            "write_target": _BLACKHOLE_DEVICE,
            "chrome_sink": _BLACKHOLE_DEVICE,
        }

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_module_id(stdout: str) -> int:
        text = (stdout or "").strip()
        if not text:
            raise RuntimeError("pactl load-module returned empty stdout")
        first = text.splitlines()[0].strip()
        token = first.split()[-1]
        try:
            return int(token)
        except ValueError as exc:
            raise RuntimeError(
                f"could not parse pactl module id from: {stdout!r}"
            ) from exc

    @staticmethod
    def _max_volume(device: str, kind: str) -> None:
        """Set volume to 100% (65536) on a sink or source."""
        cmd = ["pactl", f"set-{kind}-volume", device, "65536"]
        subprocess.run(cmd, check=False, capture_output=True)

    @staticmethod
    def _rollback(module_ids: list[int]) -> None:
        """Unload modules in reverse order on setup failure."""
        for mod_id in reversed(module_ids):
            subprocess.run(
                ["pactl", "unload-module", str(mod_id)],
                check=False, capture_output=True,
            )


def chrome_fake_audio_flags(bridge_info: dict) -> list[str]:
    """Return Chrome flags for using the fake audio input.

    The PulseAudio source is selected via PULSE_SOURCE env var AND
    pactl set-default-source (AudioBridge.setup() does both). Callers
    must still set the env var in Chrome's environment before launch:

        env["PULSE_SOURCE"] = bridge_info["device_name"]

    The chrome_output sink is the system default sink, so Chrome
    automatically uses it for speaker output — no env var needed.
    """
    system = platform.system()
    if system == "Linux":
        return ["--use-fake-ui-for-media-stream"]
    if system == "Darwin":
        return ["--use-fake-ui-for-media-stream"]
    if system == "Windows":
        raise RuntimeError("windows not supported in v3")
    raise RuntimeError(f"unsupported platform: {system}")
