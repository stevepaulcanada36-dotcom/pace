"""Minimal WSGI core: routing, request parsing, JSON/HTML responses and error mapping.

Kept dependency-free on purpose; any WSGI server (wsgiref, gunicorn, waitress) can run it.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs

from ..domain import ValidationError, parse_time
from ..study import Conflict, NotFound

log = logging.getLogger("pacesense.web")
MAX_BODY = 1_000_000
REASONS = {200: "OK", 201: "Created", 204: "No Content", 304: "Not Modified", 400: "Bad Request", 401: "Unauthorized",
           403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed", 409: "Conflict",
           413: "Payload Too Large", 422: "Unprocessable Entity", 500: "Internal Server Error"}


class HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, str]
    body: dict[str, Any]
    headers: dict[str, str]


@dataclass
class Response:
    status: int
    body: Any                      # dict/list -> JSON; str -> text with content_type
    content_type: str = "application/json"
    headers: list[tuple[str, str]] = field(default_factory=list)


def time_field(value: Any, name: str) -> float:
    t = parse_time(value)
    if t is None:
        raise ValidationError(f"{name} must be a time like 3:31:20 or a number of seconds")
    return t


Handler = Callable[..., Any]


class Router:
    def __init__(self) -> None:
        self.routes: list[tuple[str, re.Pattern, Handler]] = []

    def add(self, method: str, pattern: str, handler: Handler) -> None:
        self.routes.append((method, re.compile(f"^{pattern}$"), handler))

    def resolve(self, method: str, path: str) -> tuple[Handler, dict[str, str]]:
        allowed = False
        for m, pat, h in self.routes:
            match = pat.match(path)
            if match:
                allowed = True
                if m == method:
                    return h, match.groupdict()
        raise HTTPError(405 if allowed else 404, "method not allowed" if allowed else "not found")


def parse_request(environ: dict) -> Request:
    method = environ["REQUEST_METHOD"].upper()
    query = {k: v[0] for k, v in parse_qs(environ.get("QUERY_STRING", "")).items()}
    headers = {k[5:].replace("_", "-").lower(): v for k, v in environ.items() if k.startswith("HTTP_")}
    body: dict[str, Any] = {}
    if method in ("POST", "PUT"):
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY:
            raise HTTPError(413, "request body too large")
        raw = environ["wsgi.input"].read(length) if length else b""
        if raw:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                raise HTTPError(400, "request body must be JSON") from None
            if not isinstance(body, dict):
                raise HTTPError(400, "request body must be a JSON object")
    return Request(method, environ.get("PATH_INFO", "/") or "/", query, body, headers)


class WSGIApp:
    """Turns a Router into a WSGI callable with consistent errors and security headers."""

    def __init__(self, router: Router, cors_origin: str | None = None) -> None:
        self.router, self.cors_origin = router, cors_origin

    def _headers(self, own: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
        h = [("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "same-origin")]
        if not any(k.lower() == "cache-control" for k, _ in (own or [])):
            h.append(("Cache-Control", "no-store"))
        if self.cors_origin:
            h += [("Access-Control-Allow-Origin", self.cors_origin),
                  ("Access-Control-Allow-Headers", "Content-Type, Authorization"),
                  ("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")]
        return h

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        try:
            if environ["REQUEST_METHOD"].upper() == "OPTIONS":
                start_response("204 No Content", self._headers())
                return [b""]
            req = parse_request(environ)
            handler, params = self.router.resolve(req.method, req.path)
            out = handler(req, **params)
            resp = out if isinstance(out, Response) else Response(*out) if isinstance(out, tuple) else Response(200, out)
        except HTTPError as e:
            resp = Response(e.status, {"error": e.message})
        except NotFound as e:
            resp = Response(404, {"error": str(e)})
        except Conflict as e:
            resp = Response(409, {"error": str(e)})
        except ValidationError as e:
            resp = Response(422, {"error": str(e)})
        except Exception:  # pragma: no cover - logged, never leaked
            log.error("unhandled error\n%s", traceback.format_exc())
            resp = Response(500, {"error": "internal error"})
        if isinstance(resp.body, (dict, list)):
            data = json.dumps(_clean(resp.body), default=_json_default, allow_nan=False).encode("utf-8")
            ctype = "application/json"
        else:
            data, ctype = str(resp.body).encode("utf-8"), resp.content_type
        if resp.status == 304:
            data = b""
        start_response(f"{resp.status} {REASONS.get(resp.status, '')}",
                       [("Content-Type", ctype), ("Content-Length", str(len(data))), *self._headers(resp.headers),
                        *resp.headers])
        return [data]


def _clean(o: Any) -> Any:
    """Replace NaN/inf with null so responses are valid JSON for browsers."""
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    try:
        import numpy as np
        if isinstance(o, np.floating):
            return _clean(float(o))
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return _clean(o.tolist())
    except ImportError:  # pragma: no cover
        pass
    return o


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return float(o)
