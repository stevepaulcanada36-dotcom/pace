"""Command-line interface: ``pacesense <command>``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .analysis import analyze_repository
from .cbr import CaseBase, MarathonModel, validate_model
from .data import load_csv, runners_to_frame, synthetic_runners
from .domain import SEGMENT_LABELS, ValidationError, format_time, pace_per_km
from .repro import configured_seed
from .storage import Repository
from .study import StudyService


def _runners(args):
    if args.data:
        runners, rep = load_csv(args.data)
        print(f"Loaded {rep.runners:,} runners, {rep.races:,} races from {args.data} ({rep.rows_skipped} rows skipped)")
        return runners, Path(args.data).name
    runners = synthetic_runners(args.synthetic_n, args.seed)
    print(f"Using {len(runners):,} synthetic demo runners (seed {args.seed})")
    return runners, "synthetic-demo"


def cmd_generate(args) -> int:
    runners = synthetic_runners(args.n, args.seed)
    df = runners_to_frame(runners)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} races for {len(runners):,} runners to {args.out}")
    return 0


def cmd_validate(args) -> int:
    runners, _ = _runners(args)
    model = MarathonModel(CaseBase(runners))
    v = validate_model(model, runners, n=args.n, seed=args.seed)
    print(f"Case base: {len(model.case_base):,} cases")
    print(f"Leave-best-out validation on {v.n} runners (equal weights):")
    print(f"  mean error {v.mean_error_pct:.2f}%   median {v.median_error_pct:.2f}%   90th percentile {v.p90_error_pct:.2f}%")
    print(f"  likely range (80%) contained the actual result for {v.range_coverage_pct:.0f}% of runners")
    print("  (the paper's initial model averaged 3.14% on unfamiliar runners)")
    return 0


def cmd_predict(args) -> int:
    runners, _ = _runners(args)
    model = MarathonModel(CaseBase(runners))
    by_id = {r.runner_id: r for r in runners}
    if args.runner not in by_id:
        print(f"Runner {args.runner} not found", file=sys.stderr)
        return 2
    r = by_id[args.runner]
    races = r.recent(5)
    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    pred = model.predict(r, races, weights)
    print(f"{r.display_name} ({r.gender}, age {r.age})")
    for race, w, pr in zip(races, pred.weights, pred.per_race):
        print(f"  {race.event:<10} {race.year}  {format_time(race.finish_s)}  weight {w:.1f}  "
              f"-> {format_time(pr.finish_s) if pr else 'no similar cases'}")
    print(f"Predicted finish: {format_time(pred.finish_s)}")
    paces = pace_per_km(pred.finish_s, pred.profile)
    print("Pacing plan (min/km): " + "  ".join(f"{lab} {int(round(p)) // 60}:{int(round(p)) % 60:02d}" for lab, p in zip(SEGMENT_LABELS, paces)))
    return 0


def cmd_simulate(args) -> int:
    from .simulate import simulate_study
    runners, source = _runners(args)
    service = StudyService(MarathonModel(CaseBase(runners)), runners, Repository(args.db), data_source=source, seed=args.seed)
    ids = simulate_study(service, n_coaches=args.coaches, seed=args.seed)
    print(f"Simulated {len(ids)} coaches into {args.db} (simulated data - not evidence about real coaches)")
    return 0


def cmd_analyze(args) -> int:
    report = analyze_repository(Repository(args.db))
    text = report.to_markdown()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote report to {args.out}")
    else:
        print(text)
    return 0


def cmd_export(args) -> int:
    repo = Repository(args.db)
    repo.trials_frame(completed_only=False).to_csv(args.trials, index=False)
    repo.sessions_frame().to_csv(args.sessions, index=False)
    print(f"Wrote {args.trials} and {args.sessions}")
    return 0


def cmd_dashboard(args) -> int:
    """Start the Streamlit dashboard (needs: pip install "pacesense[dashboard]")."""
    import importlib.util
    import os
    import subprocess
    if importlib.util.find_spec("streamlit") is None:
        print('Streamlit is not installed. Install the dashboard extras first:\n\n    pip install -e ".[dashboard]"\n',
              file=sys.stderr)
        return 2
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    env = dict(os.environ, PACESENSE_DB=args.db)
    if args.data:
        env["PACESENSE_DATA"] = args.data
    if args.admin_token:
        env["PACESENSE_ADMIN_TOKEN"] = args.admin_token
    if args.offline_ai:
        env["PACESENSE_AI"] = "offline"
    if args.offline_ai or not env.get("OPENROUTER_API_KEY"):
        print("AI text features: offline rules (no OPENROUTER_API_KEY set; voice transcription uses local Whisper if installed)")
    else:
        print("AI features: OpenRouter (choose a model per task in Settings > AI models)")
    print(f"PaceSense dashboard: open http://localhost:{args.port}  (Ctrl+C to stop)")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


def cmd_serve(args) -> int:
    from wsgiref.simple_server import make_server, WSGIServer
    from socketserver import ThreadingMixIn
    from .web import App, build_service

    class ThreadingServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    app = App(build_service(args.db, args.data, args.seed), admin_token=args.admin_token, cors_origin=args.cors)
    with make_server(args.host, args.port, app, server_class=ThreadingServer) as httpd:
        print(f"PaceSense on http://{args.host}:{args.port}  (coach dashboard: /coach, study: /study; Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def cmd_live(args) -> int:
    """PaceSense Live: talk to the model in the browser; the sliders follow (needs no extra packages)."""
    import webbrowser
    from wsgiref.simple_server import make_server, WSGIServer
    from socketserver import ThreadingMixIn
    from .web import App, build_service

    class ThreadingServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    app = App(build_service(args.db, args.data, args.seed))
    url = f"http://{args.host}:{args.port}/live"
    with make_server(args.host, args.port, app, server_class=ThreadingServer) as httpd:
        print(f"PaceSense Live: open {url}  (Chrome or Edge for continuous listening; Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def cmd_reproduce(args) -> int:
    from .repro import Params, check, run_pipeline
    if args.check:
        return check(Path(args.check))
    seed = args.seed if args.seed is not None else configured_seed()
    p = Params.quick(seed) if args.quick else Params(seed)
    p.data = args.data
    run_pipeline(Path(args.out), p)
    print(f"Check it later (or on another computer) with:  pacesense reproduce --check {Path(args.out) / 'manifest.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pacesense", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def data_opts(sp):
        sp.add_argument("--data", help="race CSV (default: synthetic demo data)")
        sp.add_argument("--synthetic-n", type=int, default=8000)
        sp.add_argument("--seed", type=int, default=seed, help=f"random seed (default {seed}, from config)")

    seed = configured_seed()
    g = sub.add_parser("generate-data", help="write a synthetic race CSV")
    g.add_argument("--n", type=int, default=8000); g.add_argument("--seed", type=int, default=seed)
    g.add_argument("--out", default="races.csv"); g.set_defaults(fn=cmd_generate)

    v = sub.add_parser("validate", help="leave-best-out accuracy of the model"); data_opts(v)
    v.add_argument("--n", type=int, default=400); v.set_defaults(fn=cmd_validate)

    pr = sub.add_parser("predict", help="predict a runner's finish time"); data_opts(pr)
    pr.add_argument("runner"); pr.add_argument("--weights", help="comma-separated, one per race, 0-1")
    pr.set_defaults(fn=cmd_predict)

    s = sub.add_parser("simulate", help="run simulated coaches through the study"); data_opts(s)
    s.add_argument("--db", default="study.sqlite3"); s.add_argument("--coaches", type=int, default=71)
    s.set_defaults(fn=cmd_simulate)

    a = sub.add_parser("analyze", help="analyse a study database"); a.add_argument("--db", default="study.sqlite3")
    a.add_argument("--out"); a.set_defaults(fn=cmd_analyze)

    e = sub.add_parser("export", help="export trials and sessions as CSV"); e.add_argument("--db", default="study.sqlite3")
    e.add_argument("--trials", default="trials.csv"); e.add_argument("--sessions", default="sessions.csv")
    e.set_defaults(fn=cmd_export)

    db = sub.add_parser("dashboard", help="start the Streamlit dashboard")
    db.add_argument("--data", help="race CSV (default: synthetic demo data)")
    db.add_argument("--db", default="pacesense.sqlite3")
    db.add_argument("--port", type=int, default=8501)
    db.add_argument("--admin-token", help="unlocks the Study results page")
    db.add_argument("--offline-ai", action="store_true", help="never call OpenRouter; use the offline rules")
    db.set_defaults(fn=cmd_dashboard)

    sv = sub.add_parser("serve", help="run the study API (development server)"); data_opts(sv)
    sv.add_argument("--db", default="study.sqlite3"); sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000); sv.add_argument("--admin-token")
    sv.add_argument("--cors"); sv.set_defaults(fn=cmd_serve)

    lv = sub.add_parser("live", help="PaceSense Live: steer the model by talking (opens the browser)"); data_opts(lv)
    lv.add_argument("--db", default="live.sqlite3"); lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=8000); lv.add_argument("--no-browser", action="store_true")
    lv.set_defaults(fn=cmd_live)

    rp = sub.add_parser("reproduce", help="rebuild every reported number and write a manifest (or --check one)")
    rp.add_argument("--out", default="results"); rp.add_argument("--data", help="race CSV (default: synthetic)")
    rp.add_argument("--seed", type=int, default=None, help="default: reproducibility.seed in config/pacesense.yaml")
    rp.add_argument("--quick", action="store_true", help="smaller run (about 15 s) for CI and quick checks")
    rp.add_argument("--check", metavar="MANIFEST", help="re-run and compare with an existing manifest.json")
    rp.set_defaults(fn=cmd_reproduce)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return args.fn(args)
    except ValidationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
