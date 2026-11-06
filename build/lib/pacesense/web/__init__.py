"""HTTP layer: a JSON API over the same services the Streamlit dashboard uses, for other programs.

Development:  ``pacesense serve``  (stdlib wsgiref)
Production:   ``gunicorn 'pacesense.web:app_from_env()' -w 1 --threads 8 -b 0.0.0.0:8000``

Environment for ``app_from_env``: PACESENSE_DB (SQLite path), PACESENSE_DATA (race CSV;
synthetic demo data if unset), PACESENSE_ADMIN_TOKEN (required for /api/admin/*),
PACESENSE_ASSIGNMENT (random | interactive | noninteractive), PACESENSE_CORS_ORIGIN.
Use one worker process per SQLite file; scale with threads.

People use the Streamlit dashboard (`pacesense dashboard`); this API is for other programs.
`GET /` lists the endpoints.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Iterable

from ..tuning import configured_cbr
from ..cbr import CaseBase, MarathonModel
from ..coach import CoachService
from ..data import load_csv, synthetic_runners
from ..storage import Repository
from ..study import StudyService
from . import coach_api, hud_api, steer_api, study_api
from .core import Router, WSGIApp


class App:
    """Composes the study and coach route groups over shared model, data and storage."""

    def __init__(self, service: StudyService, coach: CoachService | None = None, admin_token: str | None = None,
                 cors_origin: str | None = None, default_assignment: str = "random", ai_factory=None) -> None:
        self.service = service
        self.coach = coach or CoachService(service.model, service.runners, service.repo)
        router = Router()
        router.add("GET", "/", lambda req: {"name": "PaceSense API", "dashboard": "run `pacesense dashboard`", "live": "/live", "command_center": "/hud",
                                            "endpoints": sorted({p.pattern.strip("^$") for _, p, _ in router.routes})})
        router.add("GET", "/health", lambda req: {"status": "ok", "runners": len(service.runners),
                                                  "cases": len(service.model.case_base)})
        study_api.register(router, service, admin_token, default_assignment)
        coach_api.register(router, self.coach)
        steer_api.register(router, self.coach, ai_factory or self._default_ai)
        hud_api.register(router, self.coach, ai_factory or self._default_ai)
        self._wsgi = WSGIApp(router, cors_origin)

    def _default_ai(self):
        """AIService for PaceSense Live's optional AI reader (offline unless an OpenRouter key is configured)."""
        from ..llm.service import AIService
        return AIService(self.service.repo)

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        return self._wsgi(environ, start_response)


def build_service(db_path: str, data_path: str | None = None, seed: int | None = None) -> StudyService:
    if data_path:
        runners, _ = load_csv(data_path)
        source = os.path.basename(data_path)
    else:
        runners, source = synthetic_runners(), "synthetic-demo"
    return StudyService(MarathonModel(CaseBase(runners, configured_cbr())), runners, Repository(db_path), data_source=source, seed=seed)


def app_from_env() -> App:
    logging.basicConfig(level=os.environ.get("PACESENSE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = build_service(os.environ.get("PACESENSE_DB", "pacesense.sqlite3"), os.environ.get("PACESENSE_DATA"))
    return App(service, admin_token=os.environ.get("PACESENSE_ADMIN_TOKEN"),
               cors_origin=os.environ.get("PACESENSE_CORS_ORIGIN"),
               default_assignment=os.environ.get("PACESENSE_ASSIGNMENT", "random"))


__all__ = ["App", "build_service", "app_from_env"]
