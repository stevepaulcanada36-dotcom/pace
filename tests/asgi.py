"""A tiny synchronous ASGI test client (Starlette's TestClient needs httpx, which the app does not)."""
from __future__ import annotations

import asyncio
import json as _json
from dataclasses import dataclass


@dataclass
class Resp:
    status: int
    headers: dict
    body: bytes

    def json(self):
        return _json.loads(self.body)

    @property
    def text(self) -> str:
        return self.body.decode()


class Client:
    def __init__(self, app) -> None:
        self.app = app

    def request(self, method: str, path: str, json=None, headers: dict | None = None,
                data: bytes | None = None, content_type: str = "application/json") -> Resp:
        path, _, query = path.partition("?")
        body = data if data is not None else (b"" if json is None else _json.dumps(json).encode())
        hdrs = [(b"host", b"test"), (b"content-type", content_type.encode()), (b"content-length", str(len(body)).encode())]
        hdrs += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
                 "path": path, "raw_path": path.encode(), "query_string": query.encode(), "root_path": "",
                 "headers": hdrs, "client": ("127.0.0.1", 1), "server": ("test", 80)}
        out = {"status": 0, "headers": {}, "body": b""}
        sent = [False]

        async def receive():
            if not sent[0]:
                sent[0] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(msg):
            if msg["type"] == "http.response.start":
                out["status"] = msg["status"]
                out["headers"] = {k.decode(): v.decode() for k, v in msg["headers"]}
            elif msg["type"] == "http.response.body":
                out["body"] += msg.get("body", b"")

        asyncio.run(self.app(scope, receive, send))
        return Resp(out["status"], out["headers"], out["body"])

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, json=None, **kw):
        return self.request("POST", path, json=json if json is not None else {}, **kw)

    def put(self, path, json=None, **kw):
        return self.request("PUT", path, json=json, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)
