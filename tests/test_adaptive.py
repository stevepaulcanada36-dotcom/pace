"""The adaptive layer: persona inference, each feedback loop, the engine, and the Studio routes that expose it."""
import unittest

from pacesense.adaptive import AdaptiveEngine, Event, Persona, default_loops, questions
from pacesense.adaptive.explain import likely_range
from pacesense.adaptive.loops import cue_category
from pacesense.coach import CoachService
from pacesense.storage import Repository
from pacesense.studio import Settings, StudioService
from pacesense.studio.app import create_app
from tests import helpers
from tests.asgi import Client


def utter(text, source="typed", applied=None, **data):
    return Event("utterance", text, source, {"applied": applied or [], **data})


def applied(race="Amsterdam 2010", index=1, old=0.5, new=0.0, verdict="supported", cue="injured"):
    return {"index": index, "race": race, "old": old, "new": new, "verdict": verdict, "cue": cue,
            "evidence": "The times agree."}


STATE = {"runner": {"gender": "F"},
         "races": [{"index": 0, "event": "Amsterdam", "year": 2009, "finish_s": 13580, "finish": "3:46:20", "weight": 0.5,
                    "evidence": {"direction": 0, "z": 0.2, "others_median_s": 13800}},
                   {"index": 1, "event": "Amsterdam", "year": 2010, "finish_s": 14637, "finish": "4:03:57", "weight": 0.5,
                    "evidence": {"direction": 1, "z": 2.4, "others_median_s": 13560}},
                   {"index": 2, "event": "Rotterdam", "year": 2014, "finish_s": 13431, "finish": "3:43:51", "weight": 0.5,
                    "evidence": {"direction": -1, "z": -1.1, "others_median_s": 13850}}],
         "prediction": {"finish_s": 12923, "low_s": 12675, "high_s": 13212},
         "steering": {"index": None, "coach_estimate_s": None}}


class ExplanationTests(unittest.TestCase):
    def test_likely_range_explains_percentiles_and_validated_coverage(self):
        explanation = likely_range(STATE)
        self.assertIn("20th to 80th percentiles", explanation)
        self.assertIn("about 80% of runners", explanation)
        self.assertIn("not a guarantee", explanation)


class PersonaTests(unittest.TestCase):
    def test_starts_neutral_and_assumes_nothing(self):
        p = Persona()
        self.assertEqual(p.archetype()["key"], "unknown")
        self.assertTrue(all(abs(d["value"] - 0.5) < 1e-9 for d in p.summary()["dimensions"]))

    def test_intuitive_coach(self):
        p = Persona()
        for t in ["I know her, she was definitely injured", "trust me she always fades", "my gut says 3:20, for sure",
                  "I know this runner, she never runs well in heat"]:
            p.observe(utter(t, "voice"))
        p.observe(Event("utterance", "x", "typed", {"applied": [applied(verdict="against_data")]}))
        self.assertLess(p.value("reliance"), 0.4)
        self.assertGreater(p.value("certainty"), 0.6)
        self.assertGreater(p.value("voice"), 0.6)
        self.assertEqual(p.archetype()["key"], "intuitive")
        self.assertTrue(any("trust me" in c or "I know" in c for c in p.dims["reliance"].cues))

    def test_analyst_coach(self):
        p = Persona()
        for t in ["why is the likely range so wide compared to similar runners?", "explain what the data says",
                  "show me the median of similar runners", "how does the model weigh the typical races and why"]:
            p.observe(utter(t))
        p.observe(Event("reaction", data={"value": "agree", "verdict": "supported"}))
        p.observe(Event("panel", data={"panel": "similar"}))
        self.assertGreater(p.value("reliance"), 0.6)
        self.assertGreater(p.value("detail"), 0.7)
        self.assertEqual(p.archetype()["key"], "analyst")

    def test_overrides_win_and_reset_clears(self):
        p = Persona()
        p.edit({"detail": 0.9})
        self.assertEqual(p.value("detail"), 0.9)
        for _ in range(5):
            p.observe(utter("ok"))                      # short answers push detail down…
        self.assertEqual(p.value("detail"), 0.9)        # …but the coach's own setting wins
        p.edit({"detail": None})
        self.assertLess(p.value("detail"), 0.5)
        p.edit({"reset": True})
        self.assertEqual(p.signals, 0)
        with self.assertRaises(ValueError):
            p.edit({"height": 1})

    def test_round_trip(self):
        p = Persona()
        p.observe(utter("why? explain it", "voice"))
        q = Persona.from_dict(p.to_dict())
        self.assertEqual(q.summary()["dimensions"], p.summary()["dimensions"])


class LoopTests(unittest.TestCase):
    def engine(self, repo=None, coach="c-test"):
        e = AdaptiveEngine(repo or Repository(":memory:"), coach)
        e.start(dict(STATE))
        return e

    def kinds(self, res, loop):
        return [a["kind"] for a in res["adaptations"] if a["loop"] == loop]

    def test_there_are_n_independent_loops_at_four_time_scales(self):
        loops = default_loops()
        self.assertGreaterEqual(len(loops), 12)
        self.assertEqual({lp.timescale for lp in loops}, {"instant", "session", "cross-session", "population"})
        self.assertEqual(len({lp.key for lp in loops}), len(loops))
        self.assertTrue(all(lp.answers for lp in loops))

    def test_data_check_raises_a_contradiction(self):
        e = self.engine()
        r = e.process(utter("2014 was a disaster", applied=[applied("Rotterdam 2014", 2, verdict="against_data")], turn=1))
        self.assertIn("prompt", self.kinds(r, "evidence"))
        self.assertEqual({c.get("action") for c in r["prompt"]["data"]["choices"]}, {"keep", "undo", "tab:similar"})

    def test_clarify_asks_which_race(self):
        e = self.engine()
        r = e.process(utter("it was a hot day"))
        self.assertEqual(r["prompt"]["loop"], "clarify")
        self.assertEqual(len(r["prompt"]["data"]["choices"]), 4)               # three races + none
        self.assertIn("Amsterdam 2010", r["prompt"]["data"]["choices"][1]["say"])
        # …but not when the interpreter did find the race
        r = e.process(utter("it was a hot day", notes=["Amsterdam 2010 already counts for nothing"]))
        self.assertNotIn("prompt", self.kinds(r, "clarify"))

    def test_bias_guard_fires_once_then_rearms(self):
        e = self.engine()
        st = dict(STATE, steering={"index": 0.92, "coach_estimate_s": 12000})
        r = e.process(utter("she is better than that", applied=[applied()]), st)
        self.assertIn("prompt", self.kinds(r, "bias"))
        r = e.process(utter("and again", applied=[applied()]), st)
        self.assertNotIn("prompt", self.kinds(r, "bias"))                     # not nagging
        e.process(utter("back", applied=[applied()]), dict(STATE, steering={"index": 0.3, "coach_estimate_s": 12000}))
        r = e.process(utter("again", applied=[applied()]), st)
        self.assertIn("prompt", self.kinds(r, "bias"))                         # re-armed

    def test_bias_guard_is_stricter_for_familiar_runners(self):
        e = self.engine()
        base = next(lp for lp in e.loops if lp.key == "bias").threshold(e.ctx)
        e.ctx.persona.runner["familiarity"].override = 0.9
        self.assertLess(next(lp for lp in e.loops if lp.key == "bias").threshold(e.ctx), base)

    def test_explanations_follow_reactions(self):
        e = self.engine()
        r = e.process(Event("reaction", data={"value": "unsure", "verdict": "supported"}))
        self.assertEqual(r["ui"]["explain"], "full")
        for _ in range(3):
            r = e.process(Event("reaction", data={"value": "agree", "verdict": "supported"}))
        self.assertEqual(r["ui"]["explain"], "brief")

    def test_voice_mode_changes_the_layout_and_back(self):
        e = self.engine()
        r = e.process(Event("voice_on"))
        self.assertEqual(r["ui"]["layout"], "voice")
        self.assertTrue(r["ui"]["speak"])
        r = e.process(Event("voice_off"))
        self.assertNotEqual(r["ui"]["layout"], "voice")

    def test_spotlight_follows_the_race_talked_about(self):
        e = self.engine()
        r = e.process(utter("2010 was hot", applied=[applied(index=1)]))
        self.assertEqual(r["ui"]["spotlight"], [1])

    def test_layout_follows_persona_unless_locked(self):
        repo = Repository(":memory:")
        e = self.engine(repo)
        for t in ["why is that? explain the data", "show me similar runners and the median, why", "explain the model"]:
            r = e.process(utter(t))
        self.assertEqual(r["persona"]["archetype"]["key"], "analyst")
        self.assertEqual(r["ui"]["layout"], "analysis")
        r = e.edit_persona({"layout_lock": "summary"})
        self.assertEqual(r["ui"]["layout"], "summary")
        with self.assertRaises(ValueError):
            e.edit_persona({"layout_lock": "sideways"})

    def test_track_record_and_shared_learning_persist_across_sessions_and_coaches(self):
        repo = Repository(":memory:")
        for coach, imp in [("c-a", 1.0), ("c-b", 0.8), ("c-c", -0.5)]:
            e = self.engine(repo, coach)
            e.process(utter("she was injured in 2010", applied=[applied(cue="injured")]))
            r = e.process(Event("reveal", data={"error_pct": 2.0 - imp, "equal_weights_error_pct": 2.0}))
            self.assertIn("insight", self.kinds(r, "record"))
            e.process(Event("reveal", data={"error_pct": 0.1, "equal_weights_error_pct": 2.0}))  # counted once
        pop = repo.get_setting("adaptive:population")
        self.assertEqual(pop["cues"]["illness/injury"], {"n": 3, "helped": 2, "sum": 1.3})
        # a fourth coach sees the pooled evidence when making the same kind of change
        e = self.engine(repo, "c-d")
        r = e.process(utter("she was sick", applied=[applied(cue="sick")]))
        msg = [a["message"] for a in r["adaptations"] if a["loop"] == "shared"]
        self.assertTrue(msg and "67%" in msg[0])
        # coach a's next session opens with their track record
        e = AdaptiveEngine(repo, "c-a")
        r = e.start(dict(STATE))
        self.assertTrue(any(a["loop"] == "record" and "1 of 1" in a["message"] for a in r["adaptations"]))
        e.forget()
        self.assertEqual(repo.get_setting("adaptive:coach:c-a"), {})

    def test_interview_walks_through_questions_built_from_the_data(self):
        qs = questions(STATE)
        self.assertEqual([q.id for q in qs], ["familiarity", "slow", "best", "trend", "estimate"])
        self.assertIn("Amsterdam 2010", qs[1].text)
        self.assertIn("18 min slower", qs[1].text)
        self.assertEqual(qs[1].choice("hot").say, "Amsterdam 2010 was a hot day")
        e = self.engine()
        r = e.start_interview()
        self.assertEqual(r["prompt"]["data"]["question"]["id"], "familiarity")
        self.assertEqual(e.interview_choice("familiarity", "coach").familiarity, 0.9)
        seen = ["familiarity"]
        for _ in range(4):
            r = e.process(Event("interview_answer", data={"choice": "skip"}))
            if r["prompt"] and r["prompt"]["data"].get("question"):
                seen.append(r["prompt"]["data"]["question"]["id"])
        self.assertEqual(seen, ["familiarity", "slow", "best", "trend", "estimate"])
        r = e.process(Event("interview_answer", data={"choice": "skip"}))
        self.assertTrue(any("interview done" in a["message"] for a in r["adaptations"]))

    def test_speech_is_short_or_full_by_persona(self):
        e = self.engine()
        card = {"items": [{"race": "Amsterdam 2010", "new": 0.0, "verdict": "supported"}], "target": "3:35:23",
                "range": ["3:31:15", "3:40:12"], "accuracy": {"text": "closer to her real result"}}
        brief = e.speak(card)
        e.edit_persona({"detail": 1.0})
        full = e.speak(card)
        self.assertNotIn("Likely", brief)
        self.assertIn("Likely between 3:31:15 and 3:40:12", full)
        self.assertIn("closer to her real result", full)

    def test_cue_categories(self):
        self.assertEqual(cue_category("hot day"), "heat")
        self.assertEqual(cue_category("injured + sick"), "illness/injury")
        self.assertEqual(cue_category("recency"), "recency")
        self.assertEqual(cue_category("xyz"), "other")


class AdaptiveApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        cls.c = Client(create_app(StudioService(coach, settings=Settings())))

    def test_session_carries_the_adaptive_state(self):
        r = self.c.post("/api/v1/sessions", {"coach_id": "coach-api-1"}).json()
        sid, a = r["id"], r["state"]["adapt"]
        self.assertEqual(len(a["loops"]), 12)
        self.assertIn("layout", a["ui"])
        self.assertEqual(self.c.get(f"/api/v1/sessions/{sid}").json()["adapt"]["persona"]["archetype"]["key"], "unknown")

        r = self.c.post(f"/api/v1/sessions/{sid}/commands", {"text": "trust the recent races more", "source": "voice"}).json()
        self.assertIn("adapt", r)
        # A voice command that changes the prediction is spoken by the browser only after it has applied the
        # change, so the server defers the confirmation instead of speaking it optimistically.
        self.assertTrue(r["voice_action_pending"])
        self.assertTrue(r["reply"])
        self.assertEqual(r["adapt"]["speak"], "")

        r = self.c.post(f"/api/v1/sessions/{sid}/events", {"kind": "voice_on"}).json()
        self.assertEqual(r["adapt"]["ui"]["layout"], "voice")
        r = self.c.post(f"/api/v1/sessions/{sid}/events", {"kind": "voice_off"}).json()
        self.assertNotEqual(r["adapt"]["ui"]["layout"], "voice")
        self.assertEqual(self.c.post(f"/api/v1/sessions/{sid}/events", {"kind": "teleport"}).status, 422)

        r = self.c.post(f"/api/v1/sessions/{sid}/interview/start").json()
        q = r["adapt"]["prompt"]["data"]["question"]
        self.assertEqual(q["id"], "familiarity")
        r = self.c.post(f"/api/v1/sessions/{sid}/interview", {"question": "familiarity", "choice": "coach"}).json()
        self.assertEqual(r["adapt"]["persona"]["runner"][0]["value"], 0.9)
        self.assertEqual(r["adapt"]["prompt"]["data"]["question"]["id"], "slow")
        r = self.c.post(f"/api/v1/sessions/{sid}/interview", {"question": "slow", "choice": "hot"}).json()
        self.assertTrue(r["card"]["items"])                                    # the answer became a checked change
        self.assertEqual(self.c.post(f"/api/v1/sessions/{sid}/interview", {"question": "slow", "choice": "nope"}).status, 422)

        r = self.c.post(f"/api/v1/sessions/{sid}/reactions", {"turn": 1, "value": "unsure"}).json()
        self.assertEqual(r["adapt"]["ui"]["explain"], "full")

        st = self.c.post(f"/api/v1/sessions/{sid}/reveal").json()
        self.assertTrue(any(a["loop"] == "record" for a in st["adapt"]["adaptations"]))

        ex = self.c.get(f"/api/v1/sessions/{sid}/export").json()
        self.assertTrue(ex["adaptive"]["adaptations"])

    def test_profile_routes(self):
        cid = "coach-api-2"
        sid = self.c.post("/api/v1/sessions", {"coach_id": cid}).json()["id"]
        p = self.c.get(f"/api/v1/coaches/{cid}/profile").json()
        self.assertEqual(p["persona"]["archetype"]["key"], "unknown")
        r = self.c.put(f"/api/v1/coaches/{cid}/profile", {"detail": 0.95, "reliance": 0.9, "certainty": 0.5,
                                                          "session_id": sid}).json()
        self.assertEqual(r["adapt"]["persona"]["archetype"]["key"], "analyst")
        self.assertEqual(r["adapt"]["ui"]["layout"], "analysis")
        r = self.c.put(f"/api/v1/coaches/{cid}/profile", {"layout_lock": "summary"}).json()
        self.assertEqual(r["layout_lock"], "summary")
        self.assertEqual(self.c.put(f"/api/v1/coaches/{cid}/profile", {"detail": 3}).status, 422)
        self.assertEqual(self.c.put("/api/v1/coaches/other-coach/profile", {"detail": 0.1, "session_id": sid}).status, 422)
        self.assertEqual(self.c.get("/api/v1/coaches/x!/profile").status, 422)
        self.assertEqual(self.c.delete(f"/api/v1/coaches/{cid}/profile").status, 204)
        self.assertEqual(self.c.get(f"/api/v1/coaches/{cid}/profile").json()["persona"]["signals"], 0)
        self.assertEqual(self.c.post("/api/v1/sessions", {"coach_id": "x"}).status, 422)

    def test_loops_are_described(self):
        rows = self.c.get("/api/v1/loops").json()
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["answers"] for r in rows))


if __name__ == "__main__":
    unittest.main()
