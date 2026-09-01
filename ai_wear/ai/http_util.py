"""HTTP helpers built on urllib (stdlib) so the addon needs no pip installs.

Runs fine from worker threads. Does NOT touch bpy.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
import gzip
import zlib
from typing import Dict, List, Optional, Tuple


class NetworkError(Exception):
    def __init__(self, message: str, kind: str = "NETWORK", status: int = 0):
        super().__init__(message)
        self.kind = kind
        self.status = status


def _decode_body(body: bytes, encoding: Optional[str]) -> bytes:
    """Decompress a response body per its Content-Encoding.

    urllib does NOT auto-decompress (unlike `requests`), so a gzip/deflate/brotli
    response comes back as raw compressed bytes and breaks json.loads with e.g.
    "utf-8 codec can't decode byte 0x80". We handle gzip/deflate (stdlib) and give
    a clear error for brotli (no stdlib decoder).
    """
    enc = (encoding or "").lower().strip()
    if not enc or "identity" in enc:
        return body
    try:
        if "gzip" in enc:
            return gzip.decompress(body)
        if "deflate" in enc:
            # zlib-wrapped deflate first, then raw deflate
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
        if "br" in enc:
            raise NetworkError(
                "Response is brotli-compressed; the addon is stdlib-only and "
                "cannot decode brotli. The request already advertises "
                "'Accept-Encoding: gzip, deflate' to avoid this, but the server/"
                "gateway returned brotli anyway. Disable brotli on the endpoint "
                "or use the OpenAI/Gemini/ComfyUI provider.", kind="API")
    except NetworkError:
        raise
    except Exception as e:
        raise NetworkError(f"Failed to decompress response ({enc}): {e}", kind="API")
    return body


def _content_encoding(resp) -> Optional[str]:
    """Read Content-Encoding from a urllib response/HTTPError, tolerant of API."""
    try:
        return resp.headers.get("Content-Encoding")
    except Exception:
        pass
    try:
        return resp.info().get("Content-Encoding")
    except Exception:
        return None


def _request(url: str, data: bytes, headers: Dict[str, str], timeout: float,
             method: str = "POST") -> Tuple[int, bytes]:
    # Advertise only the encodings we can decode (gzip/deflate). This steers
    # servers/gateways away from brotli, which stdlib can't decompress.
    h = dict(headers)
    if not any(k.lower() == "accept-encoding" for k in h):
        h["Accept-Encoding"] = "gzip, deflate"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            body = _decode_body(body, _content_encoding(r))
            return r.getcode(), body
    except urllib.error.HTTPError as e:
        body = b""
        try:
            raw = e.read()
            body = _decode_body(raw, _content_encoding(e)) if raw else b""
        except NetworkError:
            # decompression error on an error body — keep the original message
            raise
        except Exception:
            pass
        txt = body.decode("utf-8", "replace")[:1000]
        raise NetworkError(f"HTTP {e.code} {e.reason}: {txt}", kind="API", status=e.code)
    except urllib.error.URLError as e:
        raise NetworkError(f"Network error: {e.reason}", kind="NETWORK")
    except Exception as e:
        raise NetworkError(f"Request failed: {e}", kind="NETWORK")


def post_json(url: str, body, headers: Optional[Dict[str, str]] = None,
              timeout: float = 120.0) -> Tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8")
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        h.update(headers)
    return _request(url, data, h, timeout, "POST")


def get_bytes(url: str, headers: Optional[Dict[str, str]] = None,
              timeout: float = 120.0) -> Tuple[int, bytes]:
    h = {"Accept": "*/*"}
    if headers:
        h.update(headers)
    return _request(url, None, h, timeout, "GET")


def download(url: str, dest_path: str, headers: Optional[Dict[str, str]] = None,
             timeout: float = 120.0) -> str:
    code, body = get_bytes(url, headers, timeout)
    if code != 200 or not body:
        raise NetworkError(f"Download failed ({code}) from {url}", kind="NETWORK")
    with open(dest_path, "wb") as f:
        f.write(body)
    return dest_path


def post_multipart(url: str, fields: Dict[str, str],
                   files: Dict[str, Tuple[str, bytes, str]],
                   headers: Optional[Dict[str, str]] = None,
                   timeout: float = 120.0) -> Tuple[int, bytes]:
    """files: name -> (filename, bytes, content_type)."""
    boundary = "----aiwear" + uuid.uuid4().hex
    lines: List[bytes] = []
    for k, v in fields.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        lines.append(v.encode("utf-8"))
        lines.append(b"\r\n")
    for k, (fname, fbytes, ctype) in files.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'.encode())
        lines.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        lines.append(fbytes)
        lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)
    h = {"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"}
    if headers:
        h.update(headers)
    return _request(url, body, h, timeout, "POST")
