"""PaceSense Studio: the ASGI application (Starlette + Uvicorn).

    pacesense studio                       # development: http://127.0.0.1:8000
    uvicorn --factory pacesense.studio.app:create_app_from_env --host 0.0.0.0 --port 8000 --workers 1

One worker process (sessions live in memory); requests run in a thread pool, and each session has its own lock.

Routes (all JSON under /api/v1; errors are RFC 9457 problem documents with a request id):
    GET    /                                   the Studio interface
    GET    /healthz  /readyz  /metrics         liveness, readiness (data loaded), Prometheus text metrics
    GET    /api/v1/meta                        version, data source, limits
    GET    /api/v1/openapi.json                this API, described
    GET    /api/v1/runners?q=                  runner picker
    POST   /api/v1/sessions                    CreateSession            -> 201 {id, state}
    GET    /api/v1/sessions/{sid}                                       -> state
    DELETE /api/v1/sessions/{sid}                                       -> 204
    POST   /api/v1/sessions/{sid}/commands     Command                  -> {reply, card, actions, simulation, state}
    POST   /api/v1/sessions/{sid}/transcribe   audio bytes              -> {text, available, source, latency_s}
    PUT    /api/v1/sessions/{sid}/weights      SetWeight                -> {card, state}
    POST   /api/v1/sessions/{sid}/undo                                  -> {card, state}
    POST   /api/v1/sessions/{sid}/estimate     Estimate                 -> as commands
    POST   /api/v1/sessions/{sid}/simulations  Simulate                 -> simulation
    POST   /api/v1/sessions/{sid}/reveal                                -> state
    POST   /api/v1/sessions/{sid}/reactions    React                    -> reaction counts
    GET    /api/v1/sessions/{sid}/export                                -> transcript + final state (replayable)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import __version__
from ..domain import ValidationError as DomainError
from ..study import NotFound
from . import schemas
from .logs import configure_logging
from .service import StudioService
from .settings import Settings
from .store import SessionNotFound

log = logging.getLogger("pacesense.studio")
access = logging.getLogger("pacesense.studio.access")
MAX_BODY = 64 * 1024
MAX_AUDIO_BODY = 4 * 1024 * 1024       # one spoken utterance: 16 kHz mono WAV is about 2 MB a minute
AUDIO_TYPES = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/webm": "webm",
               "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/mpeg": "mp3", "audio/flac": "flac"}

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
       "base-uri 'none'; form-action 'none'")


# ---- JSON + errors ------------------------------------------------------------------------------
def _clean(o: Any) -> Any:
    """NaN/inf are not JSON; send null instead."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class JSON(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(_clean(content), ensure_ascii=False, separators=(",", ":"), default=str).encode()


class Problem(Exception):
    def __init__(self, status: int, title: str, detail: str = "", errors: list | None = None) -> None:
        self.status, self.title, self.detail, self.errors = status, title, detail, errors


def problem(request: Request, status: int, title: str, detail: str = "", errors: list | None = None) -> JSON:
    body = {"type": "about:blank", "title": title, "status": status, "detail": detail,
            "request_id": getattr(request.state, "request_id", None)}
    if errors:
        body["errors"] = errors
    return JSON(body, status, media_type="application/problem+json")


BodyT = TypeVar("BodyT", bound=BaseModel)


async def body_of(request: Request, model: type[BodyT]) -> BodyT:
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise Problem(413, "Request too large", f"bodies are limited to {MAX_BODY // 1024} KB")
    try:
        data = json.loads(raw or b"{}")
    except json.JSONDecodeError as e:
        raise Problem(400, "Malformed JSON", str(e)) from None
    try:
        return model.model_validate(data)
    except PydanticError as e:
        errs = [{"field": ".".join(map(str, x["loc"])), "message": x["msg"]} for x in e.errors()]
        raise Problem(422, "Invalid request", "; ".join(f"{x['field']}: {x['message']}" for x in errs), errs) from None


# ---- middleware: request id, access log, metrics, security headers ------------------------------
class Metrics:
    BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

    def __init__(self) -> None:
        self.routes: list = []
        self.requests: dict[tuple[str, str, int], int] = {}
        self.hist: dict[str, list[int]] = {}
        self.sum: dict[str, float] = {}

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        k = (method, route, status)
        self.requests[k] = self.requests.get(k, 0) + 1
        h = self.hist.setdefault(route, [0] * (len(self.BUCKETS) + 1))
        for i, b in enumerate(self.BUCKETS):
            if seconds <= b:
                h[i] += 1
        h[-1] += 1
        self.sum[route] = self.sum.get(route, 0.0) + seconds

    def template(self, scope) -> str:
        """The route pattern (/api/v1/sessions/{sid}) rather than the raw path, so labels stay bounded."""
        from starlette.routing import Match
        for r in self.routes:
            if r.matches(scope)[0] == Match.FULL:
                return getattr(r, "path", "unmatched") or "unmatched"
        return "unmatched"

    def render(self, service: StudioService) -> str:
        out = ["# TYPE pacesense_http_requests_total counter"]
        out += [f'pacesense_http_requests_total{{method="{m}",route="{r}",status="{s}"}} {n}'
                for (m, r, s), n in sorted(self.requests.items())]
        out.append("# TYPE pacesense_http_request_seconds histogram")
        for r, h in sorted(self.hist.items()):
            out += [f'pacesense_http_request_seconds_bucket{{route="{r}",le="{b}"}} {h[i]}' for i, b in enumerate(self.BUCKETS)]
            out += [f'pacesense_http_request_seconds_bucket{{route="{r}",le="+Inf"}} {h[-1]}',
                    f'pacesense_http_request_seconds_sum{{route="{r}"}} {self.sum[r]:.6f}',
                    f'pacesense_http_request_seconds_count{{route="{r}"}} {h[-1]}']
        st = service.sessions.stats()
        out += ["# TYPE pacesense_sessions_active gauge", f"pacesense_sessions_active {st['active']}",
                "# TYPE pacesense_sessions_created_total counter", f"pacesense_sessions_created_total {st['created_total']}",
                "# TYPE pacesense_sessions_expired_total counter", f"pacesense_sessions_expired_total {st['expired_total']}"]
        return "\n".join(out) + "\n"


class Observe:
    """Pure ASGI middleware (no buffering, so it never breaks streaming)."""

    def __init__(self, app, metrics: Metrics) -> None:
        self.app, self.metrics = app, metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = dict(scope["headers"]).get(b"x-request-id", b"").decode()[:64] or uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = rid
        t0, status = time.perf_counter(), [500]

        async def wrapped(msg):
            if msg["type"] == "http.response.start":
                status[0] = msg["status"]
                h = list(msg.get("headers", []))
                h += [(b"x-request-id", rid.encode()), (b"x-content-type-options", b"nosniff"),
                      (b"referrer-policy", b"no-referrer"), (b"x-frame-options", b"DENY")]
                if not any(k == b"content-security-policy" for k, _ in h):
                    h.append((b"content-security-policy", CSP.encode()))
                msg["headers"] = h
            await send(msg)

        try:
            await self.app(scope, receive, wrapped)
        finally:
            dt = time.perf_counter() - t0
            route = self.metrics.template(scope)
            if route not in ("/metrics",):
                self.metrics.observe(scope["method"], route, status[0], dt)
            if route != "/static" and scope["path"] != "/favicon.ico" and not route.startswith("/static") and route not in ("/healthz",):
                access.info("%s %s %s %.1fms", scope["method"], scope["path"], status[0], dt * 1000,
                            extra={"request_id": rid, "status": status[0], "ms": round(dt * 1000, 1), "route": route})




class BasicAuth:
    """Optional HTTP Basic authentication for a public single-user deployment."""

    def __init__(self, app, username: str, password: str) -> None:
        self.app, self.username, self.password = app, username, password

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") not in ("/healthz", "/readyz"):
            headers = dict(scope.get("headers", []))
            raw = headers.get(b"authorization", b"")
            valid = False
            if raw.startswith(b"Basic "):
                try:
                    decoded = base64.b64decode(raw[6:], validate=True).decode("utf-8")
                    username, password = decoded.split(":", 1)
                    valid = (hmac.compare_digest(username, self.username)
                             and hmac.compare_digest(password, self.password))
                except (ValueError, UnicodeDecodeError):
                    valid = False
            if not valid:
                body = b"Authentication required"
                await send({"type": "http.response.start", "status": 401, "headers": [
                    (b"www-authenticate", b'Basic realm="PaceSense", charset="UTF-8"'),
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", b"no-store"),
                ]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


# ---- the app ------------------------------------------------------------------------------------
def create_app(service: StudioService, settings: Settings | None = None) -> Starlette:
    settings = settings or service.settings
    metrics = Metrics()
    static_dir = resources.files("pacesense.studio").joinpath("static")
    # The script address changes whenever a static file changes, so a browser never runs an old script
    # against a new server (the release number alone stays the same between deployments).
    digest = hashlib.sha256()
    for name in ("studio.js", "adaptive.js", "studio.css"):
        asset = static_dir.joinpath(name)
        if asset.is_file():
            digest.update(asset.read_bytes())
    asset_version = f"{__version__}-{digest.hexdigest()[:10]}"
    index_html = static_dir.joinpath("index.html").read_text(encoding="utf-8").replace("{{version}}", asset_version)

    async def call(fn, *a):
        return await run_in_threadpool(fn, *a)

    # pages and operations
    async def index(request: Request):
        return Response(index_html, media_type="text/html; charset=utf-8", headers={"cache-control": "no-cache"})

    async def healthz(request: Request):
        return JSON(service.health())

    async def readyz(request: Request):
        r = service.ready()
        return JSON(r, 200 if r["ready"] else 503)

    async def metrics_view(request: Request):
        return PlainTextResponse(metrics.render(service), media_type="text/plain; version=0.0.4")

    async def meta(request: Request):
        return JSON({"name": "PaceSense Studio", "version": __version__, "data": service.info.__dict__,
                     "limits": {"session_ttl_s": settings.session_ttl_s, "max_sessions": settings.max_sessions,
                                "max_body_bytes": MAX_BODY}, "ai": service.ai_factory is not None})

    async def openapi(request: Request):
        return JSON(openapi_document())

    # runners and sessions
    async def runners(request: Request):
        q = request.query_params.get("q", "")[:64]
        gender = request.query_params.get("gender") or None
        if gender not in (None, "F", "M"):
            raise Problem(422, "Invalid request", "gender must be F or M")
        raw_min_races = request.query_params.get("min_races", "0")
        try:
            min_races = int(raw_min_races)
        except ValueError:
            raise Problem(422, "Invalid request", "min_races must be a whole number") from None
        if not 0 <= min_races <= 20:
            raise Problem(422, "Invalid request", "min_races must be between 0 and 20")
        return JSON(await call(service.runners, q, 50, gender, min_races))

    async def create(request: Request):
        b = await body_of(request, schemas.CreateSession)
        sid, state = await call(service.create, b.runner_id, b.holdout, b.coach_id, b.condition)
        return JSON({"id": sid, "state": state}, 201, headers={"location": f"/api/v1/sessions/{sid}"})

    async def get_state(request: Request):
        return JSON(await call(service.state, request.path_params["sid"]))

    async def delete(request: Request):
        await call(service.close, request.path_params["sid"])
        return Response(status_code=204)

    async def command(request: Request):
        b = await body_of(request, schemas.Command)
        voice = b.voice.model_dump(exclude_none=True) if b.voice else None
        return JSON(await call(service.command, request.path_params["sid"], b.text, b.source, voice))

    async def transcribe(request: Request):
        """Raw utterance audio (WAV, WebM, Ogg or MP4) -> Boson's transcript, for the browser voice control."""
        raw = await request.body()
        if len(raw) > MAX_AUDIO_BODY:
            raise Problem(413, "Recording too large", f"utterances are limited to {MAX_AUDIO_BODY // (1024 * 1024)} MB")
        if not raw:
            raise Problem(400, "Empty recording", "send the recorded audio as the request body")
        kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
        fmt = AUDIO_TYPES.get(kind)
        if fmt is None:
            raise Problem(415, "Unsupported audio type", f"send one of: {', '.join(sorted(AUDIO_TYPES))}")
        return JSON(await call(service.transcribe, request.path_params["sid"], raw, fmt))

    async def weights(request: Request):
        b = await body_of(request, schemas.SetWeight)
        return JSON(await call(service.set_weight, request.path_params["sid"], b.index, b.weight))

    async def undo(request: Request):
        return JSON(await call(service.undo, request.path_params["sid"]))

    async def estimate(request: Request):
        b = await body_of(request, schemas.Estimate)
        return JSON(await call(service.estimate, request.path_params["sid"], b.finish))

    async def simulate(request: Request):
        b = await body_of(request, schemas.Simulate)
        return JSON(await call(service.simulate, request.path_params["sid"], b.temp_c, b.wind_kmh, b.profile))

    async def reveal(request: Request):
        return JSON(await call(service.reveal, request.path_params["sid"]))

    async def react(request: Request):
        b = await body_of(request, schemas.React)
        return JSON(await call(service.react, request.path_params["sid"], b.turn, b.value, b.note))

    async def export(request: Request):
        sid = request.path_params["sid"]
        data = await call(service.export, sid)
        return JSON(data, headers={"content-disposition": f'attachment; filename="pacesense-session-{sid[:8]}.json"'})

    # adaptive layer
    async def session_event(request: Request):
        b = await body_of(request, schemas.SessionEvent)
        return JSON(await call(service.event, request.path_params["sid"], b.kind, b.panel))

    async def condition(request: Request):
        b = await body_of(request, schemas.Condition)
        return JSON(await call(service.set_condition, request.path_params["sid"], b.condition))

    async def choose(request: Request):
        b = await body_of(request, schemas.Choose)
        return JSON(await call(service.choose, request.path_params["sid"], b.index))

    async def interview_start(request: Request):
        return JSON(await call(service.interview_start, request.path_params["sid"]))

    async def interview_answer(request: Request):
        b = await body_of(request, schemas.InterviewAnswer)
        return JSON(await call(service.interview_answer, request.path_params["sid"], b.question, b.choice))

    def _coach_id(request: Request) -> str:
        cid = request.path_params["coach_id"]
        if not re.fullmatch(schemas.COACH_ID[1:-1], cid):
            raise Problem(422, "Invalid request", "coach id must be 3-64 letters, digits, - or _")
        return cid

    async def get_profile(request: Request):
        return JSON(await call(service.profile, _coach_id(request)))

    async def put_profile(request: Request):
        cid = _coach_id(request)
        b = await body_of(request, schemas.ProfileEdit)
        changes = b.model_dump(exclude_unset=True)
        sid = changes.pop("session_id", None)
        if not changes.get("reset"):
            changes.pop("reset", None)
        return JSON(await call(service.edit_profile, cid, changes, sid))

    async def delete_profile(request: Request):
        await call(service.forget_coach, _coach_id(request))
        return Response(status_code=204)

    async def loops(request: Request):
        from ..adaptive import default_loops
        return JSON([{k: v for k, v in lp.status(None).items() if k not in ("fired", "last")}  # type: ignore[arg-type]
                     for lp in default_loops()])

    S = "/api/v1/sessions/{sid:str}"
    C = "/api/v1/coaches/{coach_id:str}/profile"
    routes = [
        Route("/", index), Route("/healthz", healthz), Route("/readyz", readyz), Route("/metrics", metrics_view),
        Route("/api/v1/meta", meta), Route("/api/v1/openapi.json", openapi),
        Route("/api/v1/runners", runners),
        Route("/api/v1/sessions", create, methods=["POST"]),
        Route(S, get_state, methods=["GET"]), Route(S, delete, methods=["DELETE"]),
        Route(S + "/commands", command, methods=["POST"]), Route(S + "/weights", weights, methods=["PUT"]),
        Route(S + "/transcribe", transcribe, methods=["POST"]),
        Route(S + "/undo", undo, methods=["POST"]), Route(S + "/estimate", estimate, methods=["POST"]),
        Route(S + "/simulations", simulate, methods=["POST"]), Route(S + "/reveal", reveal, methods=["POST"]),
        Route(S + "/reactions", react, methods=["POST"]), Route(S + "/export", export, methods=["GET"]),
        Route(S + "/events", session_event, methods=["POST"]),
        Route(S + "/condition", condition, methods=["PUT"]), Route(S + "/choice", choose, methods=["POST"]),
        Route(S + "/interview/start", interview_start, methods=["POST"]),
        Route(S + "/interview", interview_answer, methods=["POST"]),
        Route(C, get_profile, methods=["GET"]), Route(C, put_profile, methods=["PUT"]),
        Route(C, delete_profile, methods=["DELETE"]), Route("/api/v1/loops", loops),
        Mount("/static", StaticFiles(directory=str(static_dir)), name="static"),
    ]

    async def on_problem(request: Request, exc: Problem):
        return problem(request, exc.status, exc.title, exc.detail, exc.errors)

    async def on_missing_session(request: Request, exc: SessionNotFound):
        return problem(request, 404, "Session not found", "It may have expired after a period of inactivity; "
                                                          "open the runner again.")

    async def on_not_found(request: Request, exc: NotFound):
        return problem(request, 404, "Not found", str(exc))

    async def on_domain(request: Request, exc: DomainError):
        return problem(request, 422, "Invalid request", str(exc))

    async def on_http(request: Request, exc):
        return problem(request, exc.status_code, {404: "Not found", 405: "Method not allowed"}.get(exc.status_code, "Error"),
                       str(exc.detail) if exc.detail else "")

    async def on_error(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"request_id": getattr(request.state, "request_id", None)})
        return problem(request, 500, "Internal error", "Something went wrong on our side; the request id identifies it in the logs.")

    from starlette.exceptions import HTTPException
    middleware = [Middleware(Observe, metrics=metrics), Middleware(GZipMiddleware, minimum_size=1024)]
    if settings.access_user and settings.access_password:
        middleware.append(Middleware(BasicAuth, username=settings.access_user, password=settings.access_password))
    if settings.cors_origins:
        middleware.append(Middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                                     allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["content-type", "x-request-id"]))
    handlers: dict[Any, Any] = {
        Problem: on_problem, SessionNotFound: on_missing_session, NotFound: on_not_found, DomainError: on_domain,
        HTTPException: on_http, Exception: on_error}
    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service.coach.repo.close()      # close SQLite cleanly when the server stops

    app = Starlette(routes=routes, middleware=middleware, exception_handlers=handlers, lifespan=lifespan)
    app.state.service = service
    metrics.routes = routes
    return app


def openapi_document() -> dict:
    comps = {}
    for name, model in schemas.BODIES.items():
        sch = model.model_json_schema(ref_template="#/components/schemas/{model}")
        comps.update(sch.pop("$defs", {}))
        comps[name] = sch

    def op(summary, body=None, ok="200", params=True):
        o = {"summary": summary, "responses": {ok: {"description": "OK"},
                                               "422": {"description": "Invalid request (problem+json)"}}}
        if params:
            o["parameters"] = [{"name": "sid", "in": "path", "required": True, "schema": {"type": "string"}}]
            o["responses"]["404"] = {"description": "Session not found or expired"}
        if body:
            o["requestBody"] = {"required": True, "content": {"application/json": {
                "schema": {"$ref": f"#/components/schemas/{body}"}}}}
        return o

    S = "/api/v1/sessions/{sid}"
    return {"openapi": "3.1.0",
            "info": {"title": "PaceSense Studio API", "version": __version__,
                     "description": "Case-based marathon prediction you can steer by talking (after Muijlwijk et al., IUI '24)."},
            "paths": {
                "/api/v1/runners": {"get": {"summary": "Find and filter existing runners", "parameters": [
                    {"name": "q", "in": "query", "schema": {"type": "string", "maxLength": 64},
                     "description": "Optional runner name or ID search."},
                    {"name": "gender", "in": "query", "schema": {"type": "string", "enum": ["F", "M"]},
                     "description": "Optional gender filter."},
                    {"name": "min_races", "in": "query", "schema": {"type": "integer", "minimum": 0, "maximum": 20},
                     "description": "Minimum number of previous marathons."}],
                    "responses": {"200": {"description": "Matching existing runner records"},
                                  "422": {"description": "Invalid filter (problem+json)"}}}},
                "/api/v1/sessions": {"post": op("Open a runner", "CreateSession", "201", params=False)},
                S: {"get": op("Session state"), "delete": op("Close a session", ok="204")},
                S + "/commands": {"post": op("Say or type something", "Command")},
                S + "/weights": {"put": op("Move one race's weight", "SetWeight")},
                S + "/undo": {"post": op("Undo the last change")},
                S + "/estimate": {"post": op("Record the coach's own estimate", "Estimate")},
                S + "/simulations": {"post": op("Simulate race day", "Simulate")},
                S + "/reveal": {"post": op("Reveal the hidden race (demo)")},
                S + "/reactions": {"post": op("React to a feedback card", "React")},
                S + "/export": {"get": op("Export a replayable transcript")},
                S + "/events": {"post": op("Tell the loops about voice mode or a panel", "SessionEvent")},
                S + "/condition": {"put": op("Known or unknown runner (turns loops on and off)", "Condition")},
                S + "/choice": {"post": op("Pick a choice of the open question (or just say it)", "Choose")},
                S + "/interview/start": {"post": op("Start the guided interview")},
                S + "/interview": {"post": op("Answer an interview question", "InterviewAnswer")},
                "/api/v1/coaches/{coach_id}/profile": {
                    "get": {"summary": "Coach persona and track record", "responses": {"200": {"description": "OK"}}},
                    "put": {"summary": "Correct the persona or lock a layout", "requestBody": {"required": True, "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/ProfileEdit"}}}},
                        "responses": {"200": {"description": "OK"}}},
                    "delete": {"summary": "Forget everything learned about this coach", "responses": {"204": {"description": "Deleted"}}}},
                "/api/v1/loops": {"get": {"summary": "The feedback loops and what each one answers",
                                          "responses": {"200": {"description": "OK"}}}},
            },
            "components": {"schemas": comps}}


def create_app_from_env() -> Starlette:
    settings = Settings.from_env()
    configure_logging(settings.log_level, settings.log_format)
    return create_app(StudioService.from_settings(settings), settings)


__all__ = ["create_app", "create_app_from_env", "openapi_document"]
