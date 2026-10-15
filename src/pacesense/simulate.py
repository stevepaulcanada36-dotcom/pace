"""Simulated coaches that run through the real study service.

Used for load testing, pipeline checks and power analysis. The behavioural
parameters are assumptions loosely shaped after the effect directions reported
in the paper; simulated results are NOT evidence about real coaches.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .cbr import holdout_best
from .domain import Runner, ValidationError
from .study import RUNNER_MOTIVATIONS, StudyService

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoachBehaviour:
    noise_unfamiliar: float = 0.055   # SD of a coach's own estimate, as share of finish time
    noise_familiar: float = 0.025
    conservative_bias: float = 0.01   # coaches tend to suggest slightly slow targets
    base_accept: float = 0.40
    interactive_accept: float = 0.15
    familiar_accept: float = -0.15
    interactive_x_familiar: float = 0.12
    experience_accept: float = -0.05
    steer_probability: float = 0.25   # chance an interactive coach drags the model to their own view


def _round5(s: float) -> float:
    return round(s / 300) * 300


def simulate_study(service: StudyService, n_coaches: int = 71, seed: int = 1,
                   behaviour: CoachBehaviour | None = None, own_runner_rate: float = 0.7) -> list[str]:
    b = behaviour or CoachBehaviour()
    rng = np.random.default_rng(seed)
    pool = [r for r in service.runners if len(r.races) >= 3]
    sessions = []
    for c in range(n_coaches):
        exp_years = float(rng.gamma(1.5, 3.5))
        exp_z = (exp_years - 5) / 5
        sess = service.create_session("random", coach_label=f"sim-{c + 1:03d}")
        sid, interactive = sess["session_id"], sess["condition"] == "interactive"
        own = rng.random() < own_runner_rate
        familiar_ids = [pool[i].runner_id for i in rng.choice(len(pool), 2, replace=False)] if own else []
        service.plan_runners(sid, familiar_ids, unfamiliar_only=not own)
        for seq in range(1, 5):
            trial = service.repo.get_trial(sid, seq)
            familiar = bool(trial["familiar"])
            runner: Runner = service.runner(trial["runner_id"])
            truth = trial["truth_s"] or holdout_best(runner)[0].finish_s  # coaches know their own runners' form
            noise = b.noise_familiar if familiar else b.noise_unfamiliar
            pre = _round5(truth * (1 + b.conservative_bias + rng.normal(0, noise)))
            pred = service.submit_pre(sid, seq, pre, pace_pre=float(np.round(rng.uniform(-.3, .3), 1)),
                                      familiarity=int(rng.integers(3, 6)) if familiar else None,
                                      runner_motivations=list(rng.choice(RUNNER_MOTIVATIONS, 2, replace=False)) if familiar else ())
            model = pred["finish_s"]
            if interactive:
                weights = [0.5] * len(pred["weights"])
                times = [r["finish_s"] for r in service.trial_view(sid, seq)["races"]]
                med = float(np.median(times))
                steer = rng.random() < b.steer_probability * (1.5 if familiar else 1.0)
                for i, t in enumerate(times):   # domain knowledge: discount off-days, favour recent races
                    if t > med * 1.04:
                        weights[i] = 0.2
                    elif i == len(times) - 1:
                        weights[i] = 0.8
                    try:
                        model = service.set_weights(sid, seq, weights)["finish_s"]
                    except ValidationError:
                        weights[i] = 0.5
                if steer:   # greedy search towards the coach's own estimate, then commit the final setting
                    trial_row = service.repo.get_trial(sid, seq)
                    for _ in range(6):
                        best = None
                        for i in range(len(weights)):
                            for d in (-0.3, 0.3):
                                w = weights.copy()
                                w[i] = round(float(np.clip(w[i] + d, 0, 1)), 1)
                                try:
                                    cand = service._predict(trial_row, w).finish_s
                                except ValidationError:
                                    continue
                                if best is None or abs(cand - pre) < abs(best[1] - pre):
                                    best = (w, cand)
                        if best is None or abs(best[1] - pre) >= abs(model - pre):
                            break
                        weights, model = best
                        model = service.set_weights(sid, seq, weights)["finish_s"]
            p_accept = (b.base_accept + b.interactive_accept * interactive + b.familiar_accept * familiar
                        + b.interactive_x_familiar * interactive * familiar + b.experience_accept * exp_z)
            p_accept = float(np.clip(p_accept + rng.normal(0, .08), .02, .98))
            u = rng.random()
            share = 1.0 if u < p_accept else (0.0 if u > p_accept + .25 else rng.uniform(.2, .8))
            post = _round5(pre + share * (model - pre))
            service.submit_post(sid, seq, post, pace_post=0.0, motivation="simulated")
        latent = {"perceived_competence": rng.normal(4.2 + 0.8 * interactive, 0.9),
                  "willingness_to_depend": rng.normal(4.0 + 0.4 * interactive, 1.0),
                  "se_finish": rng.normal(5, 1), "se_input": rng.normal(4.5, 1),
                  "comp_eff": rng.normal(7, 1.5), "data_exp": rng.normal(3, 1.2)}
        answers: dict = {}
        for scale in service.questionnaire.scales_for(sess["condition"]):
            for item in scale["items"]:
                answers[item["id"]] = int(np.clip(round(latent[item["score"]] + rng.normal(0, 0.5)), 1, scale["points"]))
        answers.update(age=int(rng.integers(19, 70)), years_coach=round(exp_years, 1), years_run=round(exp_years + rng.gamma(2, 4), 1),
                       runners_coached=int(rng.integers(1, 60)), marathons=int(rng.integers(0, 40)),
                       gender=str(rng.choice(["Female", "Male"], p=[.4, .6])),
                       education=str(rng.choice(["High school", "Vocational", "Undergraduate", "Graduate"], p=[.1, .25, .35, .3])))
        service.submit_questionnaire(sid, answers)
        sessions.append(sid)
    log.info("simulated %d coaches", len(sessions))
    return sessions
