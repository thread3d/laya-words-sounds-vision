"""The HTTP surface for a state that carries a recording.

The image path's rules, applied to a sound: the request names a reference, the server owns the
fetch and the transport limits, and the model is handed raw container bytes. These tests are
weight-free and need no checkpoint -- a fake router records the state it was given, which is how the
picture tests assert the same thing.
"""
import io
import os
import struct
import sys
import wave

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.audio import AudioRefused, read_audio_bytes  # noqa: E402
from laya.serve import (  # noqa: E402
    AUDIO_REQUEST_KEYS,
    _resolve_audio_state,
    _state_length,
    _state_text_view,
    create_app,
)


def _wav_bytes(seconds=0.05, rate=8000, channels=1, bits=16):
    n = int(seconds * rate)
    frames = bytearray()
    for i in range(n):
        sample = int(0.5 * ((2 ** (bits - 1)) - 1) * (1 if i % 2 else -1))
        for _ in range(channels):
            frames += sample.to_bytes(bits // 8, "little", signed=True)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(bits // 8)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return buf.getvalue()


class _Router:
    """The smallest thing `create_app` accepts: it records the state it was handed."""

    loaded = ["english"]

    def __init__(self):
        self.states = []

    def predict(self, state, questions, model=None):
        self.states.append(state)
        return {"model": "laya-rl-agent",
                "answers": {"q": {"type": "choice", "choice": "a",
                                  "probabilities": {"a": 1.0, "b": 0.0}, "confidence": 1.0}},
                "usage": {"input_tokens": 4, "output_tokens": 0},
                "routing": {"model": "english", "reason": "English Latin text"}}


def _text_state():
    return {"text": "hello", "vendor": "acme"}


def test_a_state_without_a_sound_is_returned_unchanged(monkeypatch):
    monkeypatch.delenv("LAYA_MAX_AUDIO_BYTES", raising=False)
    state = _text_state()
    assert _resolve_audio_state(state) == state
    assert _resolve_audio_state("plain") == "plain"
    assert _resolve_audio_state(None) is None


def test_a_path_reference_becomes_raw_bytes(monkeypatch, tmp_path):
    monkeypatch.delenv("LAYA_MAX_AUDIO_BYTES", raising=False)
    wav = _wav_bytes()
    path = tmp_path / "clip.wav"
    path.write_bytes(wav)
    out = _resolve_audio_state({"text": "is this a dog?", "audio_path": str(path)})
    assert sorted(out) == ["audio", "text"], out
    assert out["audio"]["bytes"] == wav
    assert "audio_path" not in out, "the reference must not reach the model as state text"


def test_a_base64_reference_decodes_to_the_same_bytes(monkeypatch):
    import base64

    monkeypatch.delenv("LAYA_MAX_AUDIO_BYTES", raising=False)
    wav = _wav_bytes()
    out = _resolve_audio_state({"audio_base64": base64.b64encode(wav).decode()})
    assert out["audio"]["bytes"] == wav


def test_more_than_one_sound_key_is_refused(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.delenv("LAYA_MAX_AUDIO_BYTES", raising=False)
    with pytest.raises(HTTPException) as info:
        _resolve_audio_state({"audio_path": "a.wav", "audio_base64": "AAAA"})
    assert info.value.status_code == 400
    assert "more than one sound key" in info.value.detail


def test_the_byte_cap_is_a_413(monkeypatch, tmp_path):
    from fastapi import HTTPException

    wav = _wav_bytes()
    monkeypatch.setenv("LAYA_MAX_AUDIO_BYTES", str(len(wav) - 1))
    with pytest.raises(HTTPException) as info:
        _resolve_audio_state({"audio": wav})
    assert info.value.status_code == 413
    assert "audio_too_large" in info.value.detail


def test_the_duration_cap_is_a_413(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.delenv("LAYA_MAX_AUDIO_BYTES", raising=False)
    monkeypatch.setenv("LAYA_MAX_AUDIO_SECONDS", "0.001")
    wav = _wav_bytes(seconds=0.5)
    with pytest.raises(HTTPException) as info:
        _resolve_audio_state({"audio": wav})
    assert info.value.status_code == 413
    assert "audio_too_long" in info.value.detail


def test_an_unreadable_or_unsupported_reference_is_a_400(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.delenv("LAYA_MAX_AUDIO_BYTES", raising=False)
    for spec in ({"audio_path": "/nonexistent/x.wav"}, {"audio": b"not a sound, honestly"},
                 {"audio_url": "http://example.com/x.wav"}):
        if "audio" in spec and isinstance(spec["audio"], bytes):
            spec["text"] = "t"
        with pytest.raises(HTTPException) as info:
            _resolve_audio_state(spec)
        assert info.value.status_code == 400, (spec, info.value.detail)


def test_a_bad_limit_in_the_environment_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_AUDIO_SECONDS", "soon")
    with pytest.raises(ValueError) as info:
        _resolve_audio_state({"audio": _wav_bytes()})
    assert "LAYA_MAX_AUDIO_SECONDS" in str(info.value), str(info.value)
    monkeypatch.setenv("LAYA_MAX_AUDIO_SECONDS", "0")
    with pytest.raises(ValueError):
        _resolve_audio_state({"audio": _wav_bytes()})


def test_the_state_length_counts_text_not_recording_bytes(monkeypatch):
    """A recording becomes a block, not characters, so it must not be measured as text."""
    monkeypatch.delenv("LAYA_MAX_AUDIO_BYTES", raising=False)
    wav = _wav_bytes(seconds=0.5)
    resolved = _resolve_audio_state({"text": "hello", "audio": wav})
    assert _state_length(resolved) < _state_length({"text": "hello"}) + 100
    view = _state_text_view(resolved)
    assert view["audio"]["bytes"].startswith("<")


def test_the_endpoint_hands_the_model_raw_bytes(monkeypatch):
    """End to end: the route resolves the reference before the router ever sees the state."""
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.delenv("LAYA_MAX_AUDIO_BYTES", raising=False)
    router = _Router()
    wav = _wav_bytes()
    import base64
    with TestClient(create_app(router=router)) as client:
        response = client.post("/v1/systemone", json={
            "state": {"text": "is this a bark?",
                      "audio_base64": base64.b64encode(wav).decode()},
            "questions": {"q": {"type": "choice", "instructions": "pick",
                                "criteria": ["a", "b"]}},
        })
    assert response.status_code == 200, response.text
    seen = router.states[0]
    assert seen["audio"]["bytes"] == wav, "the model did not receive the container bytes"
    assert "text" in seen


def test_the_refusal_reason_survives_to_the_client(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_MAX_AUDIO_BYTES", "16")
    with TestClient(create_app(router=_Router())) as client:
        import base64
        response = client.post("/v1/systemone", json={
            "state": {"audio_base64": base64.b64encode(_wav_bytes()).decode()},
            "questions": {"q": {"type": "choice", "instructions": "pick",
                                "criteria": ["a", "b"]}},
        })
    assert response.status_code == 413, response.text
    assert "audio_too_large" in response.text


def test_audio_request_keys_are_the_four_a_client_can_send():
    assert AUDIO_REQUEST_KEYS == ("audio", "audio_path", "audio_url", "audio_base64")
    # ... and `read_audio_bytes` refuses a URL unless the deployment opted in, which is what the
    # 400 above is really proving.
    previous = os.environ.pop("LAYA_ALLOW_AUDIO_URLS", None)
    try:
        with pytest.raises(AudioRefused) as info:
            read_audio_bytes({"url": "http://169.254.169.254/x"})
        assert info.value.reason == "audio_url_disabled", info.value.reason
    finally:
        if previous is not None:
            os.environ["LAYA_ALLOW_AUDIO_URLS"] = previous
