"""PaceSense Studio: settings, session store, the HTTP API (validation, errors, headers, metrics) and replay."""
import threading
import unittest

from pacesense.coach import CoachService
from pacesense.planning import PlanSession
from pacesense.steer import replay
from pacesense.storage import Repository
from pacesense.studio import Settings, StudioService
from pacesense.studio.app import create_app, openapi_document
from pacesense.studio.store import SessionNotFound, SessionStore
from tests import helpers
from tests.asgi import Client


class SettingsTests(unittest.TestCase):
    def test_defaults_and_environment(self):
        s = Settings.from_env({})
        self.assertEqual((s.host, s.port, s.log_level), ("127.0.0.1", 8000, "INFO"))
        s = Settings.from_env({"PACESENSE_PORT": "9001", "PACESENSE_LOG_LEVEL": "debug",
                               "PACESENSE_CORS_ORIGINS": "https://a.example, https://b.example", "PACESENSE_AI": "1"})
        self.assertEqual(s.port, 9001)
        self.assertEqual(s.log_level, "DEBUG")
        self.assertEqual(s.cors_origins, ["https://a.example", "https://b.example"])
        self.assertTrue(s.ai_enabled)

    def test_bad_settings_fail_fast_with_a_readable_message(self):
        with self.assertRaises(SystemExit) as cm:
            Settings.from_env({"PACESENSE_PORT": "99999"})
        self.assertIn("port", str(cm.exception))
        with self.assertRaises(SystemExit):
            Settings.from_env({"PACESENSE_DATA": "/no/such/file.csv"})


class StoreTests(unittest.TestCase):
    def test_idle_sessions_expire_and_the_size_is_capped(self):
        now = [0.0]
        st = SessionStore(ttl_s=10, max_size=2, clock=lambda: now[0])
        a = st.add("a")
        now[0] = 5
        self.assertEqual(st.entry(a).value, "a")         # touching keeps it alive
        now[0] = 14
        self.assertEqual(st.entry(a).value, "a")
        now[0] = 30
        with self.assertRaises(SessionNotFound):
            st.entry(a)
        b = st.add("b")
        st.add("c")
        st.add("d")
        self.assertEqual(len(st), 2)
        with self.assertRaises(SessionNotFound):
            st.entry(b)                                   # the oldest made room
        self.assertEqual(st.stats()["expired_total"], 2)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        cls.service = StudioService(cls.coach, settings=Settings(max_sessions=50))
        cls.c = Client(create_app(cls.service))

    def open_demo(self):
        r = self.c.post("/api/v1/sessions", {})
        self.assertEqual(r.status, 201)
        return r.json()["id"], r.json()["state"]

    def test_operations_endpoints(self):
        self.assertEqual(self.c.get("/healthz").json()["status"], "ok")
        r = self.c.get("/readyz").json()
        self.assertTrue(r["ready"])
        self.assertGreater(r["data"]["cases"], 0)
        self.assertEqual(self.c.get("/api/v1/meta").json()["name"], "PaceSense Studio")
        doc = self.c.get("/api/v1/openapi.json").json()
        self.assertEqual(doc["openapi"], "3.1.0")
        self.assertIn("Command", doc["components"]["schemas"])
        self.assertEqual(doc, openapi_document())

    def test_the_page_and_its_security_headers(self):
        r = self.c.get("/")
        self.assertEqual(r.status, 200)
        self.assertIn("PaceSense Studio", r.text)
        self.assertNotIn("{{version}}", r.text)
        self.assertIn("default-src 'self'", r.headers["content-security-policy"])
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")
        self.assertTrue(r.headers["x-request-id"])
        self.assertEqual(self.c.get("/static/studio.js").status, 200)
        self.assertEqual(self.c.get("/", headers={"x-request-id": "trace-123"}).headers["x-request-id"], "trace-123")

    def test_runner_picker_starts_with_the_demo(self):
        rows = self.c.get("/api/v1/runners").json()
        self.assertEqual(rows[0]["runner_id"], "demo")
        self.assertTrue(all(r["n_races"] >= 1 for r in rows))
        rid = rows[1]["runner_id"]
        self.assertIn(rid, [r["runner_id"] for r in self.c.get(f"/api/v1/runners?q={rid}").json()])

    def test_a_full_coaching_session(self):
        sid, st = self.open_demo()
        self.assertEqual(st["mode"], "demo")
        self.assertIsNone(st["holdout"]["truth_s"])
        self.assertFalse(st["session"]["can_undo"])

        slow = next(r for r in st["races"] if r["evidence"]["direction"] > 0)
        r = self.c.post(f"/api/v1/sessions/{sid}/commands", {"text": f"{slow['year']} was a hot day and she was injured"}).json()
        self.assertEqual(r["card"]["items"][0]["new"], 0.0)
        self.assertEqual(r["card"]["items"][0]["verdict"], "supported")
        self.assertIn(r["card"]["accuracy"]["text"], ("closer to her real result", "further from her real result",
                                                      "no closer to her real result"))
        self.assertTrue(r["state"]["session"]["can_undo"])

        r = self.c.put(f"/api/v1/sessions/{sid}/weights", {"index": 0, "weight": 0.9}).json()
        self.assertEqual(r["state"]["races"][0]["weight"], 0.9)
        r = self.c.post(f"/api/v1/sessions/{sid}/undo").json()
        self.assertEqual(r["state"]["races"][0]["weight"], st["races"][0]["weight"])

        r = self.c.post(f"/api/v1/sessions/{sid}/estimate", {"finish": "3:25"}).json()
        self.assertEqual(r["state"]["steering"]["coach_estimate_s"], 3 * 3600 + 25 * 60)

        sim = self.c.post(f"/api/v1/sessions/{sid}/simulations", {"temp_c": 26, "wind_kmh": 10, "profile": "fast_start"}).json()
        self.assertGreater(sim["summary"]["sim_finish_s"], sim["summary"]["plan_finish_s"])

        r = self.c.post(f"/api/v1/sessions/{sid}/commands", {"text": "simulate 28 degrees"}).json()
        self.assertEqual(r["simulation"]["summary"]["temp_c"], 28.0)

        self.assertEqual(self.c.post(f"/api/v1/sessions/{sid}/reactions", {"turn": 1, "value": "agree"}).json()["agree"], 1)
        st = self.c.post(f"/api/v1/sessions/{sid}/reveal").json()
        self.assertIsNotNone(st["holdout"]["truth_s"])
        self.assertIsNotNone(st["holdout"]["error_pct"])

        # the export replays to exactly the same weights (sliders included)
        ex = self.c.get(f"/api/v1/sessions/{sid}/export").json()
        self.assertEqual([t["source"] for t in ex["transcript"]][:3], ["typed", "manual", "undo"])
        again = replay(PlanSession(self.coach, ex["runner"]["id"]), ex["transcript"])
        self.assertEqual([round(w, 6) for w in again.plan.weights], [round(w, 6) for w in ex["final"]["weights"]])

        self.assertEqual(self.c.delete(f"/api/v1/sessions/{sid}").status, 204)
        self.assertEqual(self.c.get(f"/api/v1/sessions/{sid}").status, 404)

    def test_invalid_requests_get_problem_documents(self):
        sid, _ = self.open_demo()
        cases = [
            ("POST", f"/api/v1/sessions/{sid}/commands", {"text": ""}, 422, "text"),
            ("POST", f"/api/v1/sessions/{sid}/commands", {"text": "x" * 1001}, 422, "text"),
            ("POST", f"/api/v1/sessions/{sid}/commands", {"text": "hi", "admin": True}, 422, "admin"),
            ("PUT", f"/api/v1/sessions/{sid}/weights", {"index": 0, "weight": 1.5}, 422, "weight"),
            ("PUT", f"/api/v1/sessions/{sid}/weights", {"index": 15, "weight": 0.5}, 422, "out of range"),
            ("POST", f"/api/v1/sessions/{sid}/simulations", {"temp_c": 60}, 422, "temp_c"),
            ("POST", f"/api/v1/sessions/{sid}/simulations", {"profile": "sprint"}, 422, "profile"),
            ("POST", f"/api/v1/sessions/{sid}/estimate", {"finish": "fast"}, 422, "finish"),
            ("POST", f"/api/v1/sessions/{sid}/estimate", {"finish": "9:30"}, 422, "between"),
            ("POST", f"/api/v1/sessions/{sid}/reactions", {"turn": 1, "value": "maybe"}, 422, "value"),
            ("POST", "/api/v1/sessions", {"runner_id": "nobody"}, 404, "nobody"),
            ("GET", "/api/v1/sessions/no-such-session", None, 404, "expired"),
            ("GET", "/api/v1/nothing-here", None, 404, ""),
            ("DELETE", "/api/v1/sessions", None, 405, ""),
        ]
        for method, path, body, status, needle in cases:
            with self.subTest(path=path, body=body):
                r = self.c.request(method, path, json=body)
                self.assertEqual(r.status, status, r.body)
                self.assertTrue(r.headers["content-type"].startswith("application/problem+json"))
                p = r.json()
                self.assertEqual(p["status"], status)
                self.assertIn(needle, p["detail"] + " ".join(e["field"] for e in p.get("errors", [])))
                self.assertTrue(p["request_id"])
        r = self.c.request("POST", f"/api/v1/sessions/{sid}/commands", json=None)
        self.assertEqual(r.status, 422)

    def test_malformed_json(self):
        sid, _ = self.open_demo()
        import asyncio
        import json
        scope_body = b"{not json"
        out = {}

        async def run():
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
                     "path": f"/api/v1/sessions/{sid}/commands", "raw_path": b"", "query_string": b"", "root_path": "",
                     "headers": [(b"content-type", b"application/json")], "client": ("t", 1), "server": ("t", 80)}
            sent = []

            async def receive():
                if not sent:
                    sent.append(1)
                    return {"type": "http.request", "body": scope_body, "more_body": False}
                return {"type": "http.disconnect"}

            async def send(m):
                if m["type"] == "http.response.start":
                    out["status"] = m["status"]
                elif m["type"] == "http.response.body":
                    out["body"] = out.get("body", b"") + m.get("body", b"")
            await self.c.app(scope, receive, send)
        asyncio.run(run())
        self.assertEqual(out["status"], 400)
        self.assertEqual(json.loads(out["body"])["title"], "Malformed JSON")

    def test_metrics_use_route_templates(self):
        sid, _ = self.open_demo()
        self.c.get(f"/api/v1/sessions/{sid}")
        text = self.c.get("/metrics").text
        self.assertIn('route="/api/v1/sessions/{sid:str}"', text)
        self.assertNotIn(sid, text)
        self.assertIn("pacesense_sessions_active", text)

    def test_concurrent_requests_on_one_session_are_serialised(self):
        sid, st = self.open_demo()
        errors = []

        def move(i):
            try:
                r = self.c.put(f"/api/v1/sessions/{sid}/weights", {"index": i % len(st["races"]), "weight": (i % 10) / 10})
                if r.status != 200:
                    errors.append(r.status)
            except (OSError, RuntimeError, ValueError) as e:      # pragma: no cover
                errors.append(e)
        threads = [threading.Thread(target=move, args=(i,)) for i in range(24)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        ex = self.c.get(f"/api/v1/sessions/{sid}/export").json()
        self.assertEqual(len(ex["transcript"]), 24)
        again = replay(PlanSession(self.coach, ex["runner"]["id"]), ex["transcript"])
        self.assertEqual([round(w, 6) for w in again.plan.weights], [round(w, 6) for w in ex["final"]["weights"]])


if __name__ == "__main__":
    unittest.main()
