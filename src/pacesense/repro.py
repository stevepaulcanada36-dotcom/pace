"""One command that rebuilds every number PaceSense reports, and proves it rebuilt the same numbers.

    pacesense reproduce --out results                 # run the pipeline, write results/ and results/manifest.json
    pacesense reproduce --check results/manifest.json # run it again in a temporary folder and compare hashes

Steps (all seeded from `reproducibility.seed` in config/pacesense.yaml unless --seed is given):

    1. data         synthetic race data (or your --data CSV), saved and hashed
    2. validation   leave-best-out accuracy of the equal-weights model (paper Sec. 4.2.2)
    3. recency      the "trust recent races more" half-life fitted with the same protocol
    4. benchmark    PaceSense Live's interpreter on resources/steer_benchmark.yaml (dev and held-out test)
    5. study        simulated coaches through the real study code, analysed like the paper's Section 5
                    (simulated behaviour: tests the pipeline, says nothing about real coaches)

The manifest records the inputs (seed, parameters, SHA-256 of the config, lexicon, benchmark and data), the
environment (Python, platform, package versions, git commit) and a SHA-256 for every output. Numbers in the JSON
outputs are rounded to 6 significant decimals before hashing, so last-bit floating-point differences between
machines do not break the comparison; any real change does.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Callable

DEFAULT_SEED = 20240318
MANIFEST_VERSION = 1
PACKAGES = ("numpy", "pandas", "scipy", "PyYAML", "matplotlib", "streamlit", "faster-whisper")


def configured_seed() -> int:
    """`reproducibility.seed` from config/pacesense.yaml, else the default."""
    try:
        from .llm.config import load_config
        seed = (load_config().get("reproducibility") or {}).get("seed")
        return int(seed) if seed is not None else DEFAULT_SEED
    except (ValueError, TypeError, OSError):
        return DEFAULT_SEED


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _round(o: Any, digits: int = 6) -> Any:
    if isinstance(o, float):
        if not math.isfinite(o):
            return None
        return float(f"{o:.{digits}g}") if o != 0 else 0.0
    if isinstance(o, dict):
        return {str(k): _round(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, digits) for v in o]
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return _round(o.item(), digits)
    except ImportError:  # pragma: no cover
        pass
    return o


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_round(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment() -> dict:
    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    try:
        version = metadata.version("pacesense")
    except metadata.PackageNotFoundError:
        from . import __version__ as version  # type: ignore[attr-defined]
    commit = None
    try:
        root = Path(__file__).resolve().parents[2]
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                                   timeout=5).stdout.strip()
            commit = r.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return {"pacesense": version, "python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "packages": pkgs, "git_commit": commit}


def _resource_hash(name: str) -> str:
    return sha256_bytes(resources.files("pacesense.resources").joinpath(name).read_bytes())


def _config_hash() -> str | None:
    from .llm.config import config_dir
    p = config_dir() / "pacesense.yaml"
    return sha256_file(p) if p.is_file() else None


@dataclass
class Params:
    seed: int
    n_runners: int = 8000
    validate_n: int = 400
    recency_n: int = 300
    coaches: int = 71
    data: str | None = None

    @classmethod
    def quick(cls, seed: int) -> "Params":
        return cls(seed, n_runners=2500, validate_n=120, recency_n=80, coaches=24)


def run_pipeline(out: Path, p: Params, log: Callable[[str], None] = print) -> dict:
    """Run every step into `out` and return the manifest (also written to out/manifest.json)."""
    from .analysis import analyze_repository
    from .cbr import CaseBase, MarathonModel, validate_model
    from .data import load_csv, runners_to_frame, synthetic_runners
    from .simulate import simulate_study
    from .steer.benchmark import run as run_benchmark
    from .steer.recency import fit_half_life
    from .storage import Repository
    from .study import StudyService

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    outputs: dict[str, str] = {}

    # 1. data
    data_path = out / "races.csv"
    if p.data:
        shutil.copyfile(p.data, data_path)
        runners, rep = load_csv(str(data_path))
        source = Path(p.data).name
        log(f"1/5 data: {rep.runners:,} runners from {p.data}")
    else:
        runners = synthetic_runners(p.n_runners, p.seed)
        runners_to_frame(runners).to_csv(data_path, index=False, float_format="%.1f")
        runners, _ = load_csv(str(data_path))       # read back, so every later step sees exactly the saved file
        source = "synthetic"
        log(f"1/5 data: {len(runners):,} synthetic runners (seed {p.seed})")
    data_hash = sha256_file(data_path)
    model = MarathonModel(CaseBase(runners))

    # 2. validation
    v = validate_model(model, runners, n=p.validate_n, seed=p.seed)
    validation = {"n": v.n, "mean_error_pct": v.mean_error_pct, "median_error_pct": v.median_error_pct,
                  "p90_error_pct": v.p90_error_pct, "range_coverage_pct": v.range_coverage_pct,
                  "cases": len(model.case_base), "paper_initial_model_error_pct": 3.14}
    write_json(out / "validation.json", validation)
    log(f"2/5 validation: mean error {v.mean_error_pct:.2f}% on {v.n} runners, range coverage {v.range_coverage_pct:.0f}%")

    # 3. recency
    rf = fit_half_life(model, runners, n=p.recency_n, seed=p.seed)
    write_json(out / "recency_fit.json", rf.to_dict())
    best = "equal weights" if math.isinf(rf.best_half_life) else f"{rf.best_half_life:g} years"
    log(f"3/5 recency: best half-life {best} ({rf.errors[rf.best_half_life]:.2f}% vs {rf.equal_weights_error:.2f}% "
        f"with equal weights, n={rf.n})")

    # 4. benchmark
    b = run_benchmark().to_dict()
    write_json(out / "steer_benchmark.json", b)
    log(f"4/5 interpreter: dev {b['dev']['correct']}/{b['dev']['cases']}, held-out test "
        f"{b['test']['correct']}/{b['test']['cases']} (race precision {b['test']['race_precision']:.2f}, "
        f"recall {b['test']['race_recall']:.2f})")

    # 5. simulated study + analysis
    db = out / "study.sqlite3"
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    repo = Repository(str(db))
    service = StudyService(model, runners, repo, data_source=source, seed=p.seed)
    simulate_study(service, n_coaches=p.coaches, seed=p.seed)
    report = analyze_repository(repo).to_markdown()
    (out / "study_report.md").write_text(report, encoding="utf-8")
    repo.close() if hasattr(repo, "close") else None
    log(f"5/5 study: {p.coaches} simulated coaches analysed (simulated data, not evidence about real coaches)")

    for name in ("validation.json", "recency_fit.json", "steer_benchmark.json", "study_report.md"):
        outputs[name] = sha256_file(out / name)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "command": "pacesense reproduce",
        "parameters": {"seed": p.seed, "n_runners": p.n_runners, "validate_n": p.validate_n,
                       "recency_n": p.recency_n, "coaches": p.coaches, "data": Path(p.data).name if p.data else None},
        "inputs": {"data_sha256": data_hash, "config_sha256": _config_hash(),
                   "lexicon_sha256": _resource_hash("steer_lexicon.yaml"),
                   "benchmark_sha256": _resource_hash("steer_benchmark.yaml"),
                   "questionnaire_sha256": _resource_hash("questionnaire.json")},
        "outputs": outputs,
        "summary": {"validation": validation, "recency": rf.to_dict(),
                    "benchmark": {k: {x: y for x, y in v.items() if x != "failures"} for k, v in b.items()}},
        "environment": environment(),
        "not_compared": {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
                         "note": "study.sqlite3 holds random session ids and clock times, so it is not hashed; "
                                 "its analysis (study_report.md) is."},
    }
    (out / "manifest.json").write_text(json.dumps(_round(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"Done in {time.time() - t0:.0f} s. Manifest: {out / 'manifest.json'}")
    return manifest


def compare(old: dict, new: dict) -> list[str]:
    """Differences that matter: inputs, parameters and output hashes."""
    diffs = []
    for section in ("parameters", "inputs", "outputs"):
        a, b = old.get(section, {}), new.get(section, {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diffs.append(f"{section}.{k}: {a.get(k)} -> {b.get(k)}")
    return diffs


def check(manifest_path: Path, log: Callable[[str], None] = print) -> int:
    old = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    prm = old["parameters"]
    data = prm.get("data")
    if data:
        candidate = Path(manifest_path).parent / "races.csv"
        data = str(candidate) if candidate.is_file() else data
    p = Params(int(prm["seed"]), int(prm["n_runners"]), int(prm["validate_n"]), int(prm["recency_n"]),
               int(prm["coaches"]), data)
    with tempfile.TemporaryDirectory(prefix="pacesense-check-") as tmp:
        new = run_pipeline(Path(tmp), p, log=lambda m: log("   " + m))
    diffs = compare(old, new)
    env_old, env_new = old.get("environment", {}), new.get("environment", {})
    if diffs:
        log("NOT REPRODUCED. Differences:")
        for d in diffs:
            log("  - " + d)
        changed = [k for k in ("python", "platform") if env_old.get(k) != env_new.get(k)]
        pk = {k: (env_old.get("packages", {}).get(k), v) for k, v in env_new.get("packages", {}).items()
              if env_old.get("packages", {}).get(k) != v}
        if changed or pk:
            log("  Environment also differs: " + ", ".join(changed + [f"{k} {a}->{b}" for k, (a, b) in pk.items()]))
        return 1
    log(f"REPRODUCED: all {len(new['outputs'])} outputs match {manifest_path} (seed {prm['seed']}).")
    return 0


__all__ = ["DEFAULT_SEED", "configured_seed", "Params", "run_pipeline", "check", "compare", "environment"]
