#!/usr/bin/env python3
"""OpenAI-compatible local ASR shim backed by Moondream's local transcription engine.

This server speaks the same contract as the generic custom-asr-shim templates:

  - POST multipart/form-data to /audio/transcriptions
  - POST multipart/form-data to /v1/audio/transcriptions
  - Returns JSON {"text": "..."} and optional metadata

It converts uploaded media to a clean 16kHz mono PCM WAV using ffmpeg and then
invokes the same model lifecycle used by app.py:

    md.photon("moondream/parakeet-redux").transcribe(audio=wav_path, timestamps="segment")

The server is intentionally minimal and intentionally matches the contract that
OpenWhispr / OpenAI-compatible clients expect.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import uuid
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import moondream as md
except ImportError:  # pragma: no cover - optional dependency for local backend
    md = None

PORT = int(os.environ.get("MOONDREAM_SHIM_PORT", "8765"))
HOST = os.environ.get("MOONDREAM_SHIM_HOST", "127.0.0.1")
MAX_BODY_BYTES = 25 * 1024 * 1024
MODEL_NAME = os.environ.get("MOONDREAM_MODEL", "moondream/parakeet-redux")
_MODEL_SINGLETON: Any = None
_MODEL_LOCK = threading.Lock()


def configure_hf_runtime() -> None:
    """Configure stable Hugging Face cache paths and default endpoint.

    We default to the official hub unless a mirror is explicitly provided via
    HF_MIRROR or HF_ENDPOINT. That keeps downloads deterministic and avoids the
    common failure mode where a mirror is incomplete or outdated for a specific
    model repository.
    """
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE") or os.path.join(hf_home, "transformers")

    explicit_mirror = os.environ.get("HF_MIRROR") or os.environ.get("HF_ENDPOINT")
    if explicit_mirror:
        os.environ["HF_ENDPOINT"] = explicit_mirror
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")


configure_hf_runtime()


def get_or_create_model() -> Any:
    """Create one model instance per process and reuse it for all requests."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON

    with _MODEL_LOCK:
        if _MODEL_SINGLETON is None:
            if md is None:
                raise RuntimeError(
                    "Moondream is not installed. Install the local dependency or switch "
                    "ASR_BACKEND to 'online' to use a remote OpenAI-compatible service."
                )
            _MODEL_SINGLETON = md.photon(MODEL_NAME)
        return _MODEL_SINGLETON


def warmup_model() -> Any:
    """Start a background warm-up to load the model before the first real request."""
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON

    thread = threading.Thread(target=get_or_create_model, name="moondream-warmup", daemon=True)
    thread.start()
    return _MODEL_SINGLETON


def parse_multipart_form(
    body: bytes, content_type: str
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Parse multipart/form-data into (text fields, uploaded files)."""
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise ValueError("missing multipart boundary in Content-Type")

    boundary = b"--" + match.group(1).strip().encode("utf-8")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}

    for chunk in body.split(boundary):
        if not chunk or chunk.startswith(b"--"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        if b"\r\n\r\n" not in chunk:
            continue

        raw_headers, content = chunk.split(b"\r\n\r\n", 1)
        disposition = ""
        for line in raw_headers.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line
                break

        name_match = re.search(r'name="([^"]*)"', disposition)
        if not name_match:
            continue

        name = name_match.group(1)
        file_match = re.search(r'filename="([^"]*)"', disposition)
        if file_match is not None:
            files[name] = (file_match.group(1), content)
        else:
            fields[name] = content.decode("utf-8", "replace")

    return fields, files


def convert_audio(input_path: str, output_path: str | None = None) -> str:
    """Use ffmpeg to normalize input media to a clean 16kHz mono PCM WAV.

    This mirrors the app.py behavior exactly: a 16kHz mono PCM WAV is the input
    shape that the Moondream transcribe path expects.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode(
            "utf-8", errors="ignore"
        ) or result.stdout.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg conversion failed: {message}")
    return output_path


def _extract_text_from_result(result: Any) -> str:
    """Normalise text extraction across local Moondream results and remote APIs.

    Local Moondream returns a dict with ``text`` or ``segments``. Online OpenAI-
    compatible services often wrap the transcript in ``choices`` or ``messages``
    instead. This helper handles both shapes without depending on a single vendor
    schema.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        text = result.strip()
        return text

    if not isinstance(result, dict):
        return str(result).strip()

    for key in ("text", "transcript", "output_text", "final_text"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    choices = result.get("choices")
    if isinstance(choices, list):
        for item in choices:
            if not isinstance(item, dict):
                continue

            if isinstance(item.get("text"), str) and item["text"].strip():
                return item["text"].strip()

            message = item.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            value = block.get("text") or block.get("content")
                            if isinstance(value, str):
                                text_parts.append(value.strip())
                    joined = " ".join(part for part in text_parts if part)
                    if joined:
                        return joined

            delta = item.get("delta")
            if isinstance(delta, dict):
                value = delta.get("content")
                if isinstance(value, str) and value.strip():
                    return value.strip()

    segments = result.get("segments")
    if isinstance(segments, list):
        parts: list[str] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            value = segment.get("text") or segment.get("transcript") or ""
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    parts.append(cleaned)
        joined = " ".join(parts)
        if joined:
            return joined

    data = result.get("data")
    if isinstance(data, list):
        for item in data:
            text = _extract_text_from_result(item)
            if text:
                return text

    return ""


@lru_cache(maxsize=1)
def load_speech_model() -> Any:
    """Backward-compatible loader that resolves to the process-wide singleton model."""
    return get_or_create_model()


def _multipart_form_data(fields: dict[str, str], files: dict[str, tuple[str, bytes]], boundary: str) -> bytes:
    """Build a multipart/form-data payload for remote OpenAI-compatible services."""
    chunks: list[bytes] = []

    for name, value in fields.items():
        if value is None:
            continue
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")

    for name, (filename, payload) in files.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(
                "utf-8"
            )
        )
        chunks.append(b"Content-Type: application/octet-stream\r\n\r\n")
        chunks.append(payload)
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def transcribe_with_online_service(
    audio_path: str,
    model: str | None = None,
    language: str | None = None,
    prompt: str | None = None,
) -> str:
    """Call an OpenAI-compatible remote transcription service.

    This path is intentionally different from the local Moondream path:
    - the audio is uploaded over HTTP to an upstream service,
    - the upstream may respond with Chat-completion / OpenAI JSON structures,
    - the answer is normalized back to a single plain text string for OpenWhispr.
    """
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("ASR_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    endpoint = f"{base_url}/audio/transcriptions"
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ASR_API_KEY")

    with open(audio_path, "rb") as handle:
        file_bytes = handle.read()

    boundary = uuid.uuid4().hex
    fields = {
        "model": model or os.environ.get("OPENAI_ASR_MODEL") or "whisper-1",
        "language": language or "",
        "prompt": prompt or "",
    }
    body = _multipart_form_data(fields, {"file": (os.path.basename(audio_path), file_bytes)}, boundary)

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib_request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except urllib_error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"Remote ASR service error ({exc.code}): {detail}") from exc

    if not payload:
        return ""

    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return payload.decode("utf-8", errors="replace").strip()

    return _extract_text_from_result(parsed)


def transcribe_audio(
    audio_path: str,
    model: str | None = None,
    language: str | None = None,
    prompt: str | None = None,
) -> str:
    """Choose the correct backend.

    The default is local Moondream, matching the original app.py behavior. A
    remote OpenAI-compatible service can be enabled explicitly with
    ASR_BACKEND=online or ASR_BACKEND=remote.
    """
    backend = (os.environ.get("ASR_BACKEND") or os.environ.get("MOONDREAM_BACKEND") or "local").lower()

    if backend in {"online", "remote", "openai", "cloud"}:
        return transcribe_with_online_service(audio_path, model, language, prompt)

    print("Transcribing audio with local Moondream model:", model or MODEL_NAME)
    speech_model = load_speech_model()
    result = speech_model.transcribe(audio=audio_path, timestamps="segment")
    text = _extract_text_from_result(result)
    print("Transcription result:", repr(text))
    if text:
        return text
    return ""


transcribe = transcribe_audio


class ShimHandler(BaseHTTPRequestHandler):
    """HTTP server that accepts OpenAI-style multipart audio uploads."""

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        try:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            # OpenWhispr may close the socket immediately after the request or when
            # a downstream transcription error occurs. In that case there is nothing
            # left to write to and the server should fail silently instead of
            # crashing the request thread with a stack trace.
            pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("", "/health", "/v1/health"):
            self._send_json(200, {"status": "ok", "service": "moondream-asr"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in (
            "/audio/transcriptions",
            "/v1/audio/transcriptions",
        ):
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return

        if length <= 0:
            self._send_json(400, {"error": "empty body"})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "request body too large"})
            return

        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        print(f"Received POST {self.path} with Content-Length={length} and Content-Type={content_type}")

        try:
            fields, files = parse_multipart_form(body, content_type)
        except ValueError as exc:
            self._send_json(400, {"error": f"bad multipart: {exc}"})
            return

        if "file" not in files:
            self._send_json(400, {"error": "missing 'file' field"})
            return

        filename, file_bytes = files["file"]
        model_name = fields.get("model") or MODEL_NAME
        language = fields.get("language") or None
        prompt = fields.get("prompt") or None

        suffix = os.path.splitext(filename)[1] or ".webm"
        fd, input_path = tempfile.mkstemp(suffix=suffix)
        wav_path = None

        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(file_bytes)

            wav_path = convert_audio(input_path)
            print(f"Converted {filename} to WAV at {wav_path}, invoking model {model_name}")
            text = transcribe_audio(wav_path, model_name, language, prompt)
            payload = {
                "text": text,
                "object": "transcription",
                "model": model_name or MODEL_NAME,
                "language": language or "auto",
            }
            self._send_json(200, payload)
        except Exception as exc:  # pragma: no cover - defensive server layer
            self._send_json(500, {"error": f"transcription failed: {exc}"})
        finally:
            for path in (input_path, wav_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


def main() -> None:
    warmup_model()
    server = ThreadingHTTPServer((HOST, PORT), ShimHandler)
    server.daemon_threads = True
    print(f"Moondream ASR server listening on http://{HOST}:{PORT}")
    print(
        "OpenAI-compatible routes: /v1/audio/transcriptions and /audio/transcriptions"
    )
    print("Point OpenWhispr at it: Settings -> Transcription -> Self-Hosted")
    print(f"Model: {MODEL_NAME}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
