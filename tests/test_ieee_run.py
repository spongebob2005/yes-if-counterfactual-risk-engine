import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from risk_engine import GUIDANCE, REPO_ROOT, RiskEngine
from src.api.main import create_app
from src.api.store import SQLiteStore
from src.data.core import IEEE_FEATURES

IEEE_RUN = REPO_ROOT / "runs" / "ieee"


@unittest.skipUnless((IEEE_RUN / "models" / "engine.json").exists(), "no IEEE-CIS run committed")
class IeeeRunTests(unittest.TestCase):
    """Runs against the committed runs/ieee artifacts (python -m src.experiments.run_all --source ieee --out runs/ieee)."""

    @classmethod
    def setUpClass(cls):
        cls.engine = RiskEngine.load(IEEE_RUN / "models")
        cls.samples = {s["key"]: s for s in json.loads((IEEE_RUN / "models" / "samples.json").read_text())}
        cls.results = json.loads((IEEE_RUN / "reports" / "results.json").read_text())

    def test_artifacts_match_the_ieee_feature_set(self):
        self.assertEqual(self.engine.meta["data_source"], "ieee")
        self.assertEqual(tuple(self.engine.features), IEEE_FEATURES)

    def test_friction_stays_within_the_configured_cap(self):
        actions = self.results["E5_policy"]["policies"]["Learned friction (ours)"]["actions"]
        self.assertGreaterEqual(actions["ALLOW"] / sum(actions.values()), 0.85)

    def test_ring_example_stays_small(self):
        ring = self.results["ring_example"]
        self.assertLessEqual(len(ring["edges"]), 60)
        self.assertLessEqual(ring["totals"]["cards"], 50)

    def test_every_checkout_persona_has_a_sample(self):
        from src.api.main import PERSONAS

        self.assertTrue(set(PERSONAS) <= set(self.samples))

    def test_api_accepts_ieee_fields(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = create_app(engine=self.engine, store=SQLiteStore(Path(tmp.name) / "kv.sqlite3"), serve_static=False)
        http = TestClient(app)
        tx = {**self.samples["clean"]["transaction"], "C1": 7.0, "M4": "M2"}
        body = http.post("/api/decide?view=analyst", json=tx).json()
        self.assertIn(body["action"], GUIDANCE)
        self.assertEqual((body["transaction"]["C1"], body["transaction"]["M4"]), (7.0, "M2"))
        self.assertEqual(http.post("/api/decide", json={**tx, "D1": 5000}).status_code, 422)
        self.assertEqual(http.post("/api/decide", json={**tx, "R_emaildomain": "x" * 65}).status_code, 422)


class ArtifactDirTests(unittest.TestCase):
    def test_risk_engine_dir_selects_the_run_to_serve(self):
        if not (IEEE_RUN / "models" / "engine.json").exists():
            self.skipTest("no IEEE-CIS run committed")
        code = "from risk_engine import RiskEngine; print(RiskEngine.load().meta['data_source'])"
        env = {**os.environ, "RISK_ENGINE_DIR": "runs/ieee"}
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "ieee")
        env.pop("RISK_ENGINE_DIR")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "synthetic")


if __name__ == "__main__":
    unittest.main()
