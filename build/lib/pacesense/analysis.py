"""Analysis of a completed study, following Muijlwijk et al. (2024), Section 5.

* RQ1-3: Acceptance ~ interactivity * familiar + sequence + covariates,
  linear mixed model with a random intercept per coach (Table 2).
* RQ1: Perceived Competence and Willingness to Depend ~ interactivity + covariates, OLS.
* Section 5.7: mean Acceptance per ModelAdaptation class (Table 4).
* RQ4: paired t-test of model error, initial vs adapted (unfamiliar runners, interactive).
* Section 5.9: paired t-test of coaches' own error before vs after seeing the model.
* Section 5.3: number of interactions ~ runner sequence (mixed model, interactive condition).
* Section 5.7: Acceptance ~ ModelAdaptation class (mixed model, baseline non-interactive).
* Section 5.8: final slider position ~ year of the race (linear regression).
* Section 5.10: Acceptance ~ runner motivation (familiar runners).
* Table 2 note: the acceptance model refitted on binary (rounded at .5) acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .measures import ADAPTATION_CLASSES
from .stats import FitResult, TTest, cronbach_alpha, first_component, mixed_lm, ols, paired_t, welch_t
from .storage import Repository
from .questionnaire import Questionnaire
from .study import RUNNER_MOTIVATIONS

COVARIATES = ["comp_eff", "se_finish", "coaching_experience", "data_exp", "female",
              "edu_vocational", "edu_highschool", "edu_graduate"]
EXPERIENCE_COLS = ["q_years_coach", "q_years_run", "q_runners_coached", "q_marathons"]


def prepare(trials: pd.DataFrame, sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trial-level, coach-level) analysis frames with coded predictors."""
    s = sessions.copy()
    s = s[s["completed_at"].notna()] if "completed_at" in s else s
    s["interactive"] = (s["condition"] == "interactive").astype(int)
    for col in ["q_comp_eff", "q_se_finish", "q_data_exp", *EXPERIENCE_COLS,
                "q_perceived_competence", "q_willingness_to_depend"]:
        if col not in s:
            s[col] = np.nan
    s["comp_eff"], s["se_finish"], s["data_exp"] = s["q_comp_eff"], s["q_se_finish"], s["q_data_exp"]
    have = [c for c in EXPERIENCE_COLS if s[c].notna().any()]
    s["coaching_experience"] = first_component(s, have) if len(have) >= 2 and len(s) > 2 else np.nan
    gender = s.get("q_gender", pd.Series(index=s.index, dtype=object))
    s["female"] = np.where(gender == "Female", 1.0, np.where(gender == "Male", 0.0, np.nan))
    edu = s.get("q_education", pd.Series(index=s.index, dtype=object))
    for col, level in [("edu_vocational", "Vocational"), ("edu_highschool", "High school"), ("edu_graduate", "Graduate")]:
        s[col] = np.where(edu.isna(), np.nan, (edu == level).astype(float))
    keep = ["id", "interactive", "condition", "q_perceived_competence", "q_willingness_to_depend", *COVARIATES,
            *[c for c in s.columns if c.startswith("q_") and c[2:] in Questionnaire.load().item_scores]]
    coach = s[[c for c in keep if c in s]].rename(columns={"id": "session_id"})
    t = trials.merge(coach.drop(columns=["condition"]), on="session_id", how="inner")
    t["familiar"] = t["familiar"].astype(int)
    t["interactive_x_familiar"] = t["interactive"] * t["familiar"]
    return t, coach


@dataclass
class Report:
    n_sessions: int
    n_trials: int
    observations: pd.DataFrame
    acceptance_cells: pd.DataFrame
    acceptance_model: FitResult | None
    competence_model: FitResult | None
    depend_model: FitResult | None
    adaptation_table: pd.DataFrame
    rq4: TTest | None
    coach_accuracy: TTest | None
    acceptance_welch: TTest | None
    interactions: pd.DataFrame
    alphas: dict[str, float]
    class_model: FitResult | None = None
    interactions_model: FitResult | None = None
    slider_year_model: FitResult | None = None
    motivation_table: pd.DataFrame | None = None
    binary_model: FitResult | None = None
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        def fit(f: FitResult | None) -> str:
            return f"```\n{f.summary()}\n```" if f else "_Not enough data to fit this model._"

        def tt(t: TTest | None, a: str, b: str) -> str:
            return (f"{a} {t.mean_a:.2f} vs {b} {t.mean_b:.2f}; {t}" if t else "_Not enough data._")

        L = ["# PaceSense study report", "",
             f"{self.n_sessions} completed coaches, {self.n_trials} coach x runner observations.", "",
             "## Observations (cf. Table 1)", md_table(self.observations, 0), "",
             "## RQ1-RQ3: Acceptance", "Mean acceptance by condition and runner familiarity:", "",
             md_table(self.acceptance_cells), "",
             "Acceptance, interactive vs non-interactive (Welch): " + tt(self.acceptance_welch, "interactive", "non-interactive"), "",
             "Multilevel model (cf. Table 2, Acceptance column):", fit(self.acceptance_model), "",
             "## RQ1: Trust (coach level, OLS)",
             f"Cronbach's alpha: perceived competence {self.alphas.get('perceived_competence', float('nan')):.3f}, "
             f"willingness to depend {self.alphas.get('willingness_to_depend', float('nan')):.3f}", "",
             "Perceived competence:", fit(self.competence_model), "", "Willingness to depend:", fit(self.depend_model), "",
             "## Model adaptation and acceptance (cf. Table 4)", md_table(self.adaptation_table), "",
             "## RQ4: Model accuracy (unfamiliar runners, interactive condition)",
             tt(self.rq4, "initial model error %", "adapted model error %"), "",
             "## Coaches' own accuracy (unfamiliar runners, both conditions)",
             tt(self.coach_accuracy, "coach before %", "coach after %"), "",
             "## Slider interactions per runner position (interactive, cf. Sec. 5.3)", md_table(self.interactions, 1), "",
             "Number of interactions ~ runner sequence (mixed model):", fit(self.interactions_model), "",
             "## Acceptance by adaptation class (cf. Sec. 5.7; baseline: non-interactive)", fit(self.class_model), "",
             "## Slider position ~ year of the race (cf. Sec. 5.8)", fit(self.slider_year_model), "",
             "## Acceptance ~ runner motivation, familiar runners (cf. Sec. 5.10)",
             "_One mixed model per motivation (indicator vs. not indicated), random intercept per coach._", "",
             md_table(self.motivation_table, 3) if self.motivation_table is not None else "_Not enough data._", "",
             "## Robustness: binary acceptance (rounded at .5), same predictors (cf. Table 2 note)",
             "_Linear probability mixed model; the paper used a multilevel logistic model._", fit(self.binary_model)]
        if self.notes:
            L += ["", "## Notes", *[f"- {n}" for n in self.notes]]
        return "\n".join(L) + "\n"


def md_table(df: pd.DataFrame, digits: int = 2) -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    if df is None or df.empty:
        return "_No data._"
    def cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    head = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx, *values in df.itertuples(name=None):
        lines.append("| " + " | ".join([str(idx)] + [cell(v) for v in values]) + " |")
    return "\n".join(lines)


def _try(fn, notes: list[str], label: str):
    try:
        return fn()
    except (ValueError, np.linalg.LinAlgError) as e:
        notes.append(f"{label}: {e}")
        return None


def analyze(trials: pd.DataFrame, sessions: pd.DataFrame) -> Report:
    if trials.empty or sessions.empty or "condition" not in sessions:
        raise ValueError("no completed observations to analyse")
    notes: list[str] = []
    t, coach = prepare(trials, sessions)
    if t.empty:
        raise ValueError("no completed observations to analyse")
    cond_label = t["interactive"].map({1: "Interactive", 0: "Non-interactive"}).rename("condition")
    fam_label = t["familiar"].map({1: "Familiar runner", 0: "Unfamiliar runner"}).rename("runner")
    obs = pd.crosstab(cond_label, fam_label, margins=True, margins_name="Total")
    cells = t.pivot_table(index=cond_label, columns=fam_label, values="acceptance", aggfunc="mean")

    covs = [c for c in COVARIATES if c in t and t[c].notna().mean() > 0.8]
    dropped = sorted(set(COVARIATES) - set(covs))
    if dropped:
        notes.append(f"covariates left out (missing for >20% of coaches): {', '.join(dropped)}")
    acc_x = ["interactive", "familiar", "interactive_x_familiar", "seq", *covs]
    acc_model = _try(lambda: mixed_lm(t, "acceptance", acc_x, group="session_id"), notes, "acceptance model")
    trust_x = ["interactive", *[c for c in covs]]
    comp = _try(lambda: ols(coach, "q_perceived_competence", trust_x), notes, "competence model")
    dep = _try(lambda: ols(coach, "q_willingness_to_depend", trust_x), notes, "willingness model")
    item_scores = Questionnaire.load().item_scores
    alphas = {f: cronbach_alpha(coach, [f"q_{i}" for i, ff in item_scores.items() if ff == f and f"q_{i}" in coach])
              for f in ("perceived_competence", "willingness_to_depend")}

    rows = []
    for k, label in ADAPTATION_CLASSES.items():
        sub = t[t["adaptation_class"] == k]
        rows.append({"class": f"{k}. {label}", "runners": len(sub), "mean_acceptance": sub["acceptance"].mean()})
    adapt = pd.DataFrame(rows).set_index("class")

    i_unf = t[(t["interactive"] == 1) & (t["familiar"] == 0)]
    rq4 = paired_t(i_unf["err_model_initial"], i_unf["err_model_adapted"])
    unf = t[t["familiar"] == 0]
    coach_acc = paired_t(unf["err_coach_pre"], unf["err_coach_post"])
    welch = welch_t(t.loc[t["interactive"] == 1, "acceptance"], t.loc[t["interactive"] == 0, "acceptance"])
    inter = (t[t["interactive"] == 1].groupby("seq")["interactions"].agg(["mean", "median", "max"])
             if (t["interactive"] == 1).any() else pd.DataFrame())
    # Sec. 5.7: adaptation classes as predictors
    for k in range(1, 5):
        t[f"class_{k}"] = (t["adaptation_class"] == k).astype(float)
    class_model = _try(lambda: mixed_lm(t[t["adaptation_class"].notna()], "acceptance",
                                        [f"class_{k}" for k in range(1, 5)], group="session_id"), notes, "class model")
    # Sec. 5.3: interactions ~ sequence
    ti = t[t["interactive"] == 1]
    inter_model = _try(lambda: mixed_lm(ti, "interactions", ["seq", "familiar", *covs], group="session_id"),
                       notes, "interactions model")
    # Sec. 5.8: slider position ~ race year
    rows = []
    for r in ti.itertuples():
        if r.weights and isinstance(r.race_years, str) and r.race_years:
            for w, y in zip(str(r.weights).split("|"), r.race_years.split("|")):
                rows.append({"slider": float(w), "year": float(y), "session_id": r.session_id})
    sy = pd.DataFrame(rows)
    slider_model = _try(lambda: ols(sy[sy["year"] > 0], "slider", ["year"]), notes, "slider-year model") if len(sy) else None
    # Sec. 5.10: runner motivations (familiar runners only)
    fam = t[t["familiar"] == 1].copy()
    mot_cols = []
    for m in RUNNER_MOTIVATIONS:
        col = "mot_" + m.lower().replace(" ", "_").replace("-", "_")
        fam[col] = fam["runner_motivations"].fillna("").str.split("|").map(lambda xs, m=m: float(m in xs))
        if fam[col].sum() >= 3:
            mot_cols.append(col)
    mot_rows = []
    for col in mot_cols:
        f = _try(lambda col=col: mixed_lm(fam, "acceptance", [col], group="session_id"), notes, f"motivation {col}")
        if f is not None and col in f.params.index:
            mot_rows.append({"motivation": col[4:].replace("_", " "), "runners": int(fam[col].sum()),
                             "coef": f.params.loc[col, "coef"], "se": f.params.loc[col, "se"], "p": f.params.loc[col, "p"]})
    mot_table = pd.DataFrame(mot_rows).set_index("motivation") if mot_rows else None
    binary = _try(lambda: mixed_lm(t, "acceptance_binary", acc_x, group="session_id"), notes, "binary model")
    return Report(len(coach), len(t), obs, cells, acc_model, comp, dep, adapt, rq4, coach_acc, welch, inter, alphas,
                  class_model, inter_model, slider_model, mot_table, binary, notes)


def analyze_repository(repo: Repository) -> Report:
    return analyze(repo.trials_frame(completed_only=True), repo.sessions_frame())
