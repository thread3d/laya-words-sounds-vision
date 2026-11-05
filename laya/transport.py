"""Reading a reference into raw bytes, with a cap, for every modality that accepts one.

A request names an image or a sound in one of four ways -- raw bytes it already read, a filesystem
path, a base64 string, or (only when a deployment opts in) an ``http(s)`` URL -- and every modality
has to answer the same three questions before it can look at the payload:

* how many bytes arrived, and is that over the cap?
* is this reference even a shape we accept?
* may this process fetch a URL at all?

That is one policy, so it lives in one place. Each modality keeps its own vocabulary for what it
refused (``ImageRefused``, ``AudioRefused``) and its own reason codes, which is why
:func:`read_reference` takes the word for the thing and a factory for the exception rather than
raising a class of its own.

**URLs are off by default and that is not an oversight.** A self-hosted decision engine that fetches
whatever URL a request names is a server-side request forgery primitive pointed at the operator's
own network: ``http://169.254.169.254/`` and ``http://localhost:9200/`` are one request away. A
deployment that wants it opts in -- by setting the modality's environment variable, or by passing
``allow_url=True`` after putting its own egress policy in front -- and even then only ``http`` and
``https`` are fetched.
"""
import os
from typing import Callable, Optional


class Refused(ValueError):
    """A refused payload, carrying the reason a caller should be told.

    ``reason`` is a stable code (``image_too_large``, ``audio_unsupported``); ``detail`` is the
    sentence a person reads. A modality subclasses this to keep its own name for it, so
    ``except ImageRefused`` keeps working while the policy stays shared.
    """

    def __init__(self, kind: str, reason: str, detail: str):
        super().__init__("%s: %s" % (reason, detail))
        self.kind = kind
        self.reason = reason
        self.detail = detail


def looks_base64(spec: str) -> bool:
    """Whether `spec` is plausibly base64 text rather than a path.

    A path wins over a guess, because a file that happens to be named like base64 is still a file.
    The length and alphabet checks are what keep an ordinary sentence from being decoded.
    """
    if not isinstance(spec, str) or os.path.exists(spec):
        return False
    stripped = spec.strip()
    if len(stripped) < 64 or len(stripped) % 4:
        return False
    return all(c.isalnum() or c in "+/=\n\r" for c in stripped)


def read_reference(spec, *, kind: str, max_bytes: int, allow_url: bool = False,
                   url_env: Optional[str] = None, timeout: float = 5.0,
                   refuse: Optional[Callable[[str, str], None]] = None) -> bytes:
    """Resolve a reference into raw bytes, applying the byte cap.

    Accepted forms: raw ``bytes`` (already read), a filesystem ``path``, a base64 string, a mapping
    with one of ``bytes``/``path``/``url``/``base64``, or -- only when `allow_url` is set -- an
    ``http(s)`` URL. `kind` names the payload in every message ("image", "sound"), and `url_env` is
    the variable a deployment sets to opt into URL fetching.

    The returned bytes are *not* validated beyond their size: a caller still checks the header
    (:func:`laya.vision.check_image_bytes`, :func:`laya.audio.check_audio_bytes`) before any decoder
    runs, because "small enough" is not the same as "a picture" or "a sound".
    """
    def refuse_or_raise(reason: str, detail: str):
        if refuse is not None:
            refuse(reason, detail)
        raise Refused(kind, reason, detail)

    limit = int(max_bytes)
    if isinstance(spec, (bytes, bytearray)):
        data = bytes(spec)
    elif isinstance(spec, dict):
        if "bytes" in spec:
            data = spec["bytes"]
        elif "path" in spec:
            data = _read_path(spec["path"], limit, kind, refuse_or_raise)
        elif "url" in spec:
            data = _read_url(spec["url"], limit, allow_url, url_env, timeout, kind, refuse_or_raise)
        elif "base64" in spec:
            data = _decode_base64(spec["base64"], limit, kind, refuse_or_raise)
        else:
            refuse_or_raise(kind, "a mapping must carry one of bytes, path, url or base64; got %s"
                            % ", ".join(sorted(spec)))
    elif isinstance(spec, str):
        if spec.startswith(("http://", "https://")):
            data = _read_url(spec, limit, allow_url, url_env, timeout, kind, refuse_or_raise)
        elif os.path.exists(spec):
            data = _read_path(spec, limit, kind, refuse_or_raise)
        else:
            data = _decode_base64(spec, limit, kind, refuse_or_raise)
    else:
        refuse_or_raise(kind, "expected bytes, a path, a URL or base64; got %s"
                        % type(spec).__name__)
    if len(data) > limit:
        refuse_or_raise("%s_too_large" % kind,
                        "%d bytes exceeds the %d-byte limit" % (len(data), limit))
    return data


def _read_path(path, limit: int, kind: str, refuse) -> bytes:
    if not isinstance(path, str):
        refuse(kind, "path must be a string, got %s" % type(path).__name__)
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        refuse("%s_unreadable" % kind, "%s: %s" % (path, exc.strerror or exc))
    if size > limit:
        refuse("%s_too_large" % kind,
               "%s is %d bytes, over the %d-byte limit" % (path, size, limit))
    try:
        with open(path, "rb") as handle:
            return handle.read(limit + 1)
    except OSError as exc:
        refuse("%s_unreadable" % kind, "%s: %s" % (path, exc.strerror or exc))


def _decode_base64(text, limit: int, kind: str, refuse) -> bytes:
    import base64
    if not isinstance(text, str):
        refuse(kind, "base64 must be a string, got %s" % type(text).__name__)
    # A base64 string is 4/3 the size of what it decodes to, so the encoded length is already a
    # lower bound on the work; refusing here avoids materialising a huge buffer to measure it.
    if len(text) > (limit // 3 + 1) * 4:
        refuse("%s_too_large" % kind,
               "the encoded %s is longer than a %d-byte payload can be" % (kind, limit))
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:                                     # noqa: BLE001 - any decode error
        refuse("%s_invalid" % kind, "base64 did not decode: %s" % exc)


def _read_url(url, limit: int, allow_url: bool, url_env: Optional[str], timeout: float,
              kind: str, refuse) -> bytes:
    from urllib.parse import urlparse

    if not allow_url:
        hint = ("set %s=1 to enable it" % url_env) if url_env else "pass allow_url=True to enable it"
        refuse("%s_url_disabled" % kind,
               "network fetching is off by default; %s, or send the bytes" % (hint,))
    if not isinstance(url, str):
        refuse(kind, "url must be a string, got %s" % type(url).__name__)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        refuse("%s_url_scheme" % kind,
               "only http and https are fetched, got %r" % (parsed.scheme,))
    if not parsed.hostname:
        refuse("%s_url_invalid" % kind, "no host in %r" % (url,))
    import urllib.request
    request = urllib.request.Request(url, headers={"User-Agent": "laya-%s-fetch" % kind})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:   # noqa: S310 - opt-in
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > limit:
                refuse("%s_too_large" % kind,
                       "the server declares %s bytes, over the %d-byte limit" % (declared, limit))
            data = response.read(limit + 1)
    except Refused:
        raise
    except Exception as exc:                                     # noqa: BLE001 - network errors
        refuse("%s_unreachable" % kind, "%s: %s" % (url, exc))
    if len(data) > limit:
        refuse("%s_too_large" % kind, "%s returned more than %d bytes" % (url, limit))
    return data
