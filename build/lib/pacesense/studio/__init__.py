"""PaceSense Studio: the production web app (Starlette API + a clean dashboard).

    pacesense studio            or            python -m pacesense.studio
"""
from .service import StudioService
from .settings import Settings

__all__ = ["Settings", "StudioService", "main"]


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    from .app import create_app
    from .logs import configure_logging

    p = argparse.ArgumentParser(prog="pacesense studio", description="Run PaceSense Studio.")
    p.add_argument("--host"), p.add_argument("--port", type=int)
    p.add_argument("--data", help="race CSV (default: synthetic demo data)"), p.add_argument("--db")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    a = p.parse_args(argv)
    settings = Settings.from_env(host=a.host, port=a.port, data_path=a.data, db_path=a.db)
    configure_logging(settings.log_level, settings.log_format)
    app = create_app(StudioService.from_settings(settings), settings)
    url = f"http://{'localhost' if settings.host in ('127.0.0.1', '0.0.0.0') else settings.host}:{settings.port}/"
    print(f"PaceSense Studio is running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower(),
                access_log=False, proxy_headers=True, timeout_graceful_shutdown=10)
    return 0
