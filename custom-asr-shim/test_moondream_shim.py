#!/usr/bin/env python3
"""Tests for the Moondream OpenAI-compatible ASR shim."""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import types
import unittest
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

fake_moondream = types.ModuleType("moondream")


def _fake_photon(model_name):
    return types.SimpleNamespace(
        transcribe=lambda audio=None, timestamps=None: {"text": "hello world"}
    )


fake_moondream.photon = _fake_photon
sys.modules.setdefault("moondream", fake_moondream)

import moondream_shim  # noqa: E402

CRLF = b"\r\n"


class WarmupModelTests(unittest.TestCase):
    def test_singleton_model_loader_reuses_one_instance(self):
        original_singleton = getattr(moondream_shim, "_MODEL_SINGLETON", None)
        original_md = moondream_shim.md
        moondream_shim._MODEL_SINGLETON = None
        calls = {"count": 0}

        def fake_photon(model_name):
            calls["count"] += 1
            return object()

        moondream_shim.md = types.SimpleNamespace(photon=fake_photon)
        try:
            first = moondream_shim.get_or_create_model()
            second = moondream_shim.get_or_create_model()
            self.assertIs(first, second)
            self.assertEqual(calls["count"], 1)
        finally:
            moondream_shim._MODEL_SINGLETON = original_singleton
            moondream_shim.md = original_md

    def test_configure_hf_runtime_sets_default_cache_and_endpoint(self):
        original_home = os.environ.pop("HF_HOME", None)
        original_endpoint = os.environ.pop("HF_ENDPOINT", None)
        original_hub_cache = os.environ.pop("HF_HUB_CACHE", None)
        original_transformers_cache = os.environ.pop("TRANSFORMERS_CACHE", None)
        try:
            moondream_shim.configure_hf_runtime()
            self.assertTrue(os.environ.get("HF_HOME"))
            self.assertTrue(os.environ.get("HF_ENDPOINT"))
            self.assertTrue(os.environ.get("HF_HUB_CACHE"))
            self.assertTrue(os.environ.get("TRANSFORMERS_CACHE"))
        finally:
            if original_home is not None:
                os.environ["HF_HOME"] = original_home
            if original_endpoint is not None:
                os.environ["HF_ENDPOINT"] = original_endpoint
            if original_hub_cache is not None:
                os.environ["HF_HUB_CACHE"] = original_hub_cache
            if original_transformers_cache is not None:
                os.environ["TRANSFORMERS_CACHE"] = original_transformers_cache




def build_multipart(boundary: bytes, parts) -> bytes:
    out = []
    for name, filename, content, ctype in parts:
        out.append(b"--" + boundary + CRLF)
        disp = b'Content-Disposition: form-data; name="' + name.encode() + b'"'
        if filename is not None:
            disp += b'; filename="' + filename.encode() + b'"'
        out.append(disp + CRLF)
        if ctype is not None:
            out.append(b"Content-Type: " + ctype + CRLF)
        out.append(CRLF)
        out.append(content + CRLF)
    out.append(b"--" + boundary + b"--" + CRLF)
    return b"".join(out)


class MultipartParserTests(unittest.TestCase):
    def test_parse_text_and_file(self):
        body = build_multipart(
            b"B9",
            [
                ("file", "audio.webm", b"\x00\x01\x02", b"audio/webm"),
                ("model", None, b"whisper-1", None),
                ("language", None, b"en", None),
                ("prompt", None, "hello world".encode("utf-8"), None),
            ],
        )
        fields, files = moondream_shim.parse_multipart_form(body, "multipart/form-data; boundary=B9")
        self.assertEqual(fields["model"], "whisper-1")
        self.assertEqual(fields["language"], "en")
        self.assertEqual(fields["prompt"], "hello world")
        self.assertEqual(files["file"][0], "audio.webm")
        self.assertEqual(files["file"][1], b"\x00\x01\x02")


class HandlerHarness:
    def __init__(self):
        self.original_transcribe = moondream_shim.transcribe_audio
        self.original_convert = moondream_shim.convert_audio
        self.captured = {}

    def start(self):
        moondream_shim.convert_audio = lambda path: path

        def fake_transcribe(audio_path, model, language, prompt):
            with open(audio_path, "rb") as f:
                self.captured["bytes"] = f.read()
            self.captured["model"] = model
            self.captured["language"] = language
            self.captured["prompt"] = prompt
            return "hello world"

        moondream_shim.transcribe_audio = fake_transcribe
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), moondream_shim.ShimHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        moondream_shim.convert_audio = self.original_convert
        moondream_shim.transcribe_audio = self.original_transcribe

    def post(self, path, body, content_type):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("POST", path, body, {"Content-Type": content_type})
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()


class ExtractionShapeTests(unittest.TestCase):
    def test_extracts_local_moondream_segment_text(self):
        payload = {
            "segments": [
                {"text": "hello"},
                {"text": "world"},
            ]
        }
        self.assertEqual(moondream_shim._extract_text_from_result(payload), "hello world")

    def test_extracts_online_service_openai_style_text(self):
        payload = {
            "choices": [
                {"text": "remote transcript"},
            ]
        }
        self.assertEqual(moondream_shim._extract_text_from_result(payload), "remote transcript")

    def test_extracts_chat_message_style_text(self):
        payload = {
            "choices": [
                {"message": {"content": "chat transcript"}},
            ]
        }
        self.assertEqual(moondream_shim._extract_text_from_result(payload), "chat transcript")


class HttpHandlerTests(unittest.TestCase):
    def test_valid_transcription_response(self):
        body = build_multipart(
            b"HB",
            [
                ("file", "audio.webm", b"RIFFfake-wav-bytes\r\n\x00", b"audio/webm"),
                ("model", None, b"moondream/parakeet-redux", None),
                ("language", None, b"en", None),
            ],
        )
        h = HandlerHarness()
        h.start()
        try:
            for path in ("/audio/transcriptions", "/v1/audio/transcriptions"):
                status, raw = h.post(path, body, "multipart/form-data; boundary=HB")
                self.assertEqual(status, 200)
                payload = json.loads(raw)
                self.assertEqual(payload["text"], "hello world")
                self.assertEqual(payload["object"], "transcription")
            self.assertEqual(h.captured["bytes"], b"RIFFfake-wav-bytes\r\n\x00")
            self.assertEqual(h.captured["model"], "moondream/parakeet-redux")
            self.assertEqual(h.captured["language"], "en")
        finally:
            h.stop()

    def test_backend_selection_switches_to_online(self):
        original = os.environ.get("ASR_BACKEND")
        try:
            os.environ["ASR_BACKEND"] = "online"
            called = {}

            def fake_online(audio_path, model, language, prompt):
                called["path"] = audio_path
                called["model"] = model
                called["language"] = language
                called["prompt"] = prompt
                return "remote transcript"

            moondream_shim.transcribe_with_online_service = fake_online
            self.assertEqual(
                moondream_shim.transcribe_audio("/tmp/input.wav", "provider-model", "zh", "hello"),
                "remote transcript",
            )
            self.assertEqual(called["model"], "provider-model")
            self.assertEqual(called["language"], "zh")
            self.assertEqual(called["prompt"], "hello")
        finally:
            if original is None:
                os.environ.pop("ASR_BACKEND", None)
            else:
                os.environ["ASR_BACKEND"] = original
            moondream_shim.transcribe_with_online_service = moondream_shim.transcribe_with_online_service

    def test_openwhispr_curl_style_request(self):
        """Equivalent to curl -F file=@sample.webm -F model=... -F language=en.

        This mirrors the exact OpenWhispr / self-hosted request shape: a multipart
        form with a file field and a small set of text metadata fields.
        """
        body = build_multipart(
            b"WHISPR",
            [
                ("file", "audio.webm", b"RIFFfake-wav-bytes\r\n\x00", b"audio/webm"),
                ("model", None, b"moondream/parakeet-redux", None),
                ("language", None, b"en", None),
                ("prompt", None, b"transcribe accurately", None),
            ],
        )
        h = HandlerHarness()
        h.start()
        try:
            status, raw = h.post(
                "/audio/transcriptions",
                body,
                "multipart/form-data; boundary=WHISPR",
            )
            self.assertEqual(status, 200)
            payload = json.loads(raw)
            self.assertEqual(payload["text"], "hello world")
            self.assertEqual(payload["object"], "transcription")
            self.assertEqual(payload["model"], "moondream/parakeet-redux")
            self.assertEqual(payload["language"], "en")
            self.assertEqual(h.captured["bytes"], b"RIFFfake-wav-bytes\r\n\x00")
            self.assertEqual(h.captured["model"], "moondream/parakeet-redux")
            self.assertEqual(h.captured["language"], "en")
            self.assertEqual(h.captured["prompt"], "transcribe accurately")
        finally:
            h.stop()

    def test_wrong_path_404(self):
        body = build_multipart(b"HB", [("file", "audio.webm", b"data", b"audio/webm")])
        h = HandlerHarness()
        h.start()
        try:
            status, _ = h.post("/nope", body, "multipart/form-data; boundary=HB")
            self.assertEqual(status, 404)
        finally:
            h.stop()


if __name__ == "__main__":
    unittest.main()
