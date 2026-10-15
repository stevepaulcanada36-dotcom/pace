"""One-click PaceSense demo. Run:  python run_demo.py   (add --dashboard to open the dashboard afterwards)

Works without installing the package. Needs numpy, pandas and scipy in the SAME Python
that runs this script; if they are missing it prints the exact command to install them.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

missing = []
for mod in ("numpy", "pandas", "scipy"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"This Python ({sys.executable}) is missing: {', '.join(missing)}")
    print("Install them for this exact Python with:\n")
    print(f'    "{sys.executable}" -m pip install numpy pandas scipy\n')
    print("then run this script again.")
    sys.exit(1)

from pacesense.cli import main  # noqa: E402

out = HERE / "output"
out.mkdir(exist_ok=True)
data = HERE / "data" / "races.csv"
db = out / "study.sqlite3"
if db.exists():
    db.unlink()
for extra in ("-wal", "-shm"):
    p = Path(str(db) + extra)
    if p.exists():
        p.unlink()


def step(title, args):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    code = main(args)
    if code:
        sys.exit(code)


if not data.exists():
    data.parent.mkdir(exist_ok=True)
    step("Generating the demo race database", ["generate-data", "--out", str(data)])
step("1. Model accuracy (hide each runner's best race, predict it)", ["validate", "--data", str(data)])
step("2. Prediction for runner R00042, equal weights", ["predict", "--data", str(data), "R00042"])
step("3. Same runner, coach down-weights the older race", ["predict", "--data", str(data), "R00042", "--weights", "0.1,1.0"])
step("4. Simulating 71 coaches through the study", ["simulate", "--data", str(data), "--coaches", "71", "--db", str(db)])
step("5. Analysis report", ["analyze", "--db", str(db), "--out", str(out / "report.md")])
step("6. Export", ["export", "--db", str(db), "--trials", str(out / "trials.csv"), "--sessions", str(out / "sessions.csv")])
print(f"\nDone. Open the '{out.name}' folder: report.md, trials.csv, sessions.csv")

if "--dashboard" in sys.argv:
    sys.exit(main(["dashboard", "--data", str(data), "--db", str(out / "dashboard.sqlite3"), "--admin-token", "mysecret"]))
print('Next, open the dashboard:  pacesense dashboard --data data/races.csv   (or: python run_demo.py --dashboard)')
