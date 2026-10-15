"""PaceSense Live over HTTP: /live and /api/steer/*."""
import io
import json
import unittest
from wsgiref.util import setup_testing_defaults

from pacesense.web import App

from .helpers import service


def call(app, method, path, body=None):
    env = {}
    setup_testing_defaults(env)
    raw = json.dumps(body).encode() if body is not None else b""
    env.update(REQUEST_METHOD=method, PATH_INFO=path, CONTENT_LENGTH=str(len(raw)), wsgi_input=None)
    env["wsgi.input"] = io.BytesIO(raw)
    status = {}
    out = b"".join(app(env, lambda s, h: status.update(code=int(s.split()[0]), headers=dict(h))))
    ctype = status["headers"]["Content-Type"]
    return status["code"], (json.loads(out) if ctype.startswith("application/json") else out.decode())


class SteerWebTests(unittest.TestCase):
    def setUp(self):
        self.app = App(service(), ai_factory=lambda: None)

    def test_live_page_is_served(self):
        code, html = call(self.app, "GET", "/live")
        self.assertEqual(code, 200)
        self.assertIn("PaceSense", html)
        self.assertIn("webkitSpeechRecognition", html)

    def test_session_round_trip(self):
        code, d = call(self.app, "GET", "/api/steer/demo-runner")
        self.assertEqual(code, 200)
        code, d = call(self.app, "POST", "/api/steer/sessions", {"runner_id": d["runner_id"]})
        self.assertEqual(code, 201)
        sid, races = d["id"], d["state"]["races"]
        slow = next((r for r in races if r["evidence"]["direction"] > 0), races[0])
        code, d = call(self.app, "POST", f"/api/steer/sessions/{sid}/utterances",
                       {"text": f"{slow['year']} was a hot day and she was injured", "source": "voice"})
        self.assertEqual(code, 200)
        self.assertEqual(d["turn"]["applied"][0]["verdict"], "supported")
        self.assertEqual(d["state"]["races"][slow["index"]]["weight"], 0.0)
        code, d = call(self.app, "POST", f"/api/steer/sessions/{sid}/manual", {"index": 0, "weight": 0.9})
        self.assertEqual(code, 200)
        self.assertLess(d["state"]["reasoned_share"], 1.0)
        code, d = call(self.app, "POST", f"/api/steer/sessions/{sid}/undo")
        self.assertEqual(d["state"]["races"][0]["weight"], 0.5)
        code, d = call(self.app, "GET", f"/api/steer/sessions/{sid}/transcript")
        self.assertEqual([t["source"] for t in d["transcript"]], ["voice", "manual", "undo"])

    def test_errors(self):
        self.assertEqual(call(self.app, "GET", "/api/steer/sessions/doesnotexist1")[0], 404)
        self.assertEqual(call(self.app, "POST", "/api/steer/sessions", {})[0], 422)
        _, d = call(self.app, "GET", "/api/steer/demo-runner")
        _, d = call(self.app, "POST", "/api/steer/sessions", {"runner_id": d["runner_id"]})
        self.assertEqual(call(self.app, "POST", f"/api/steer/sessions/{d['id']}/utterances", {"text": " "})[0], 422)
        self.assertEqual(call(self.app, "POST", f"/api/steer/sessions/{d['id']}/audio", {"audio_b64": "%%%"})[0], 422)


if __name__ == "__main__":
    unittest.main()
