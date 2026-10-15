import io
import json
import unittest
from wsgiref.util import setup_testing_defaults

from pacesense.analysis import analyze_repository
from pacesense.simulate import simulate_study
from pacesense.questionnaire import Questionnaire
from pacesense.study import Conflict
from pacesense.domain import ValidationError
from pacesense.web import App

from .helpers import runners, service

ANSWERS = {**{i["id"]: 5 for sc in Questionnaire.load().spec["scales"] for i in sc["items"]},
           "age": 44, "gender": "Female", "education": "Graduate", "years_coach": 6, "years_run": 15,
           "runners_coached": 12, "marathons": 9}


class StudyServiceTests(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.fam = [r.runner_id for r in runners() if len(r.races) >= 3][:2]

    def run_session(self, condition):
        sid = self.svc.create_session(condition)["session_id"]
        views = self.svc.plan_runners(sid, self.fam)
        self.assertEqual(len(views), 4)
        self.assertEqual(sum(v["familiar"] for v in views), 2)
        for seq in range(1, 5):
            v = self.svc.trial_view(sid, seq)
            pred = self.svc.submit_pre(sid, seq, 12600, familiarity=4 if v["familiar"] else None)
            if condition == "interactive":
                w = [0.5] * len(pred["weights"]); w[0] = 0.0 if len(w) > 1 else 1.0
                pred = self.svc.set_weights(sid, seq, w)
            out = self.svc.submit_post(sid, seq, pred["finish_s"])
            self.assertEqual(out["measures"]["acceptance"], 1.0)
        return sid

    def test_full_session_both_conditions(self):
        for cond in ("interactive", "noninteractive"):
            sid = self.run_session(cond)
            res = self.svc.submit_questionnaire(sid, ANSWERS)
            self.assertEqual(res["perceived_competence"], 5)
            self.assertTrue(self.svc.session_summary(sid)["completed"])
        df = self.svc.repo.trials_frame()
        self.assertEqual(len(df), 8)
        unf = df[df.familiar == 0]
        self.assertTrue(unf["truth_s"].notna().all())
        self.assertTrue(df[df.familiar == 1]["truth_s"].isna().all())

    def test_hidden_best_race_not_shown(self):
        sid = self.svc.create_session("interactive")["session_id"]
        self.svc.plan_runners(sid, unfamiliar_only=True)
        for seq in range(1, 5):
            t = self.svc.repo.get_trial(sid, seq)
            shown = [r["finish_s"] for r in self.svc.trial_view(sid, seq)["races"]]
            self.assertTrue(all(t["truth_s"] <= x for x in shown))
            self.assertNotIn("truth_s", self.svc.trial_view(sid, seq))

    def test_rules_are_enforced(self):
        sid = self.svc.create_session("noninteractive")["session_id"]
        with self.assertRaises(ValidationError):
            self.svc.plan_runners(sid, self.fam[:1])
        self.svc.plan_runners(sid, unfamiliar_only=True)
        with self.assertRaises(Conflict):
            self.svc.plan_runners(sid, unfamiliar_only=True)
        with self.assertRaises(Conflict):
            self.svc.submit_pre(sid, 2, 12600)              # runner 1 first
        with self.assertRaises(Conflict):
            self.svc.submit_post(sid, 1, 12600)             # estimate first
        p = self.svc.submit_pre(sid, 1, 12600)
        with self.assertRaises(Conflict):
            self.svc.set_weights(sid, 1, [1.0] * len(p["weights"]))  # non-interactive
        with self.assertRaises(ValidationError):
            self.svc.submit_post(sid, 1, 500)
        with self.assertRaises(Conflict):
            self.svc.submit_questionnaire(sid, ANSWERS)    # runners not finished

    def test_questionnaire_validation(self):
        sid = self.run_session("interactive")
        bad = dict(ANSWERS); bad["comp1"] = 9
        with self.assertRaises(ValidationError):
            self.svc.submit_questionnaire(sid, bad)
        missing = dict(ANSWERS); del missing["sei1"]
        with self.assertRaises(ValidationError):
            self.svc.submit_questionnaire(sid, missing)

    def test_unfamiliar_runners_reuse_previous_selections(self):
        """Sec. 4.1: unfamiliar runners are the runners previous coaches selected as their own,
        matched on the number of shown marathons."""
        first = self.svc.create_session("interactive")["session_id"]
        own = [r for r in runners() if len(r.recent(5)) == 4][:2]
        self.svc.plan_runners(first, own)
        second = self.svc.create_session("interactive")["session_id"]
        mine = [r for r in runners() if len(r.recent(5)) == 3 and r not in own][:2]
        self.svc.plan_runners(second, mine)
        unf = {t["runner_id"] for t in self.svc.repo.list_trials(second) if not t["familiar"]}
        self.assertEqual(unf, {r.runner_id for r in own})
        for t in self.svc.repo.list_trials(second):
            self.assertEqual(len(t["runner"]["races"]), 3)

    def test_questionnaire_scores(self):
        sid = self.run_session("noninteractive")
        a = dict(ANSWERS); a["comp1"] = 7; a["comp2"] = 1
        del a["sei1"], a["sei2"]                      # not asked in the non-interactive condition
        self.svc.submit_questionnaire(sid, a)
        q = self.svc.repo.get_session(sid)["questionnaire"]
        self.assertEqual(q["perceived_competence"], (7 + 1 + 5 + 5) / 4)
        self.assertNotIn("se_input", q)

    def test_analysis_of_empty_study_says_so(self):
        with self.assertRaisesRegex(ValueError, "no completed observations"):
            analyze_repository(service().repo)

    def test_simulation_and_analysis(self):
        svc = service(seed=9)
        simulate_study(svc, n_coaches=40, seed=5)
        rep = analyze_repository(svc.repo)
        self.assertEqual(rep.n_sessions, 40)
        self.assertEqual(rep.n_trials, 160)
        self.assertIsNotNone(rep.acceptance_model)
        self.assertIn("interactive", rep.acceptance_model.params.index)
        md = rep.to_markdown()
        self.assertIn("RQ4", md)
        self.assertIsNotNone(rep.class_model)
        self.assertIsNotNone(rep.slider_year_model)


def call(app, method, path, body=None, token=None):
    env = {}
    setup_testing_defaults(env)
    raw = json.dumps(body).encode() if body is not None else b""
    env.update(REQUEST_METHOD=method, PATH_INFO=path.split("?")[0], QUERY_STRING=path.partition("?")[2],
               CONTENT_LENGTH=str(len(raw)), CONTENT_TYPE="application/json")
    env["wsgi.input"] = io.BytesIO(raw)
    if token:
        env["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    out = {}
    def start(status, headers):
        out["status"], out["headers"] = int(status.split()[0]), dict(headers)
    data = b"".join(app(env, start))
    ctype = out["headers"]["Content-Type"]
    return out["status"], (json.loads(data) if ctype == "application/json" else data.decode())


class WebTests(unittest.TestCase):
    def setUp(self):
        self.app = App(service(), admin_token="secret")

    def test_api_flow(self):
        self.assertEqual(call(self.app, "GET", "/health")[0], 200)
        st, res = call(self.app, "GET", "/api/runners?q=R000")
        self.assertEqual(st, 200)
        self.assertTrue(res["results"])
        st, s = call(self.app, "POST", "/api/sessions", {"assignment": "interactive"})
        self.assertEqual(st, 201)
        sid = s["session_id"]
        hand = {"name": "Ann", "gender": "F", "age": 41, "races": [
            {"event": "Amsterdam", "year": 2017, "finish": "3:37:49"},
            {"event": "Amsterdam", "year": 2018, "finish": "3:50:55"}]}
        st, plan = call(self.app, "POST", f"/api/sessions/{sid}/runners",
                        {"familiar_ids": [res["results"][0]["runner_id"]], "familiar_runners": [hand]})
        self.assertEqual(st, 201, plan)
        for seq in range(1, 5):
            st, v = call(self.app, "GET", f"/api/sessions/{sid}/runners/{seq}")
            body = {"coach_pre": "3:40:00", "familiarity": 5} if v["familiar"] else {"coach_pre": "3:40:00"}
            st, p = call(self.app, "POST", f"/api/sessions/{sid}/runners/{seq}/estimate", body)
            self.assertEqual(st, 200, p)
            st, p2 = call(self.app, "PUT", f"/api/sessions/{sid}/runners/{seq}/weights", {"weights": [1.0] * len(p["weights"])})
            self.assertEqual(st, 200, p2)
            st, a = call(self.app, "POST", f"/api/sessions/{sid}/runners/{seq}/advice", {"coach_post": "3:38:00"})
            self.assertEqual(st, 200, a)
        st, q = call(self.app, "POST", f"/api/sessions/{sid}/questionnaire", ANSWERS)
        self.assertEqual(st, 200, q)
        st, summary = call(self.app, "GET", f"/api/sessions/{sid}")
        self.assertTrue(summary["completed"])

    def test_survey_page_and_questionnaire(self):
        self.assertEqual(call(self.app, "GET", "/study")[0], 404)     # the survey moved to Streamlit
        st, s = call(self.app, "POST", "/api/sessions", {"assignment": "noninteractive"})
        st, q = call(self.app, "GET", f"/api/sessions/{s['session_id']}/questionnaire")
        self.assertEqual(st, 200)
        self.assertNotIn("self_efficacy_input", [x["id"] for x in q["scales"]])

    def test_errors_and_auth(self):
        self.assertEqual(call(self.app, "GET", "/nope")[0], 404)
        self.assertEqual(call(self.app, "DELETE", "/api/sessions")[0], 405)
        self.assertEqual(call(self.app, "GET", "/api/sessions/" + "0" * 32)[0], 404)
        st, s = call(self.app, "POST", "/api/sessions", {"assignment": "bogus"})
        self.assertEqual(st, 422)
        self.assertEqual(call(self.app, "GET", "/api/admin/trials.csv")[0], 401)
        self.assertEqual(call(self.app, "GET", "/api/admin/trials.csv", token="wrong")[0], 401)
        st, csv = call(self.app, "GET", "/api/admin/trials.csv", token="secret")
        self.assertEqual(st, 200)
        self.assertEqual(call(App(service()), "GET", "/api/admin/report")[0], 403)


if __name__ == "__main__":
    unittest.main()
