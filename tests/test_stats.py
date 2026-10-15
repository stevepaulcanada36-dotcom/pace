import unittest

import numpy as np
import pandas as pd
from scipy import stats as st

from pacesense.stats import cronbach_alpha, mixed_lm, ols, paired_t


class StatsTests(unittest.TestCase):
    def test_ols_matches_lstsq(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=200)})
        df["y"] = 1 + 2 * df["x"] + rng.normal(size=200)
        f = ols(df, "y", ["x"])
        self.assertAlmostEqual(f.params.loc["x", "coef"], 2, delta=0.2)
        lr = st.linregress(df["x"], df["y"])
        self.assertAlmostEqual(f.params.loc["x", "coef"], lr.slope, places=8)
        self.assertAlmostEqual(f.params.loc["x", "se"], lr.stderr, places=8)

    def test_mixed_model_recovers_parameters(self):
        rng = np.random.default_rng(1)
        G, n = 300, 4
        g = np.repeat(np.arange(G), n)
        treat = np.repeat(rng.integers(0, 2, G), n)
        x = rng.normal(size=G * n)
        u = rng.normal(0, 0.5, G)[g]
        y = 0.3 + 0.25 * treat - 0.1 * x + u + rng.normal(0, 0.4, G * n)
        df = pd.DataFrame({"y": y, "treat": treat, "x": x, "g": g})
        f = mixed_lm(df, "y", ["treat", "x"], "g")
        self.assertAlmostEqual(f.params.loc["treat", "coef"], 0.25, delta=0.12)
        self.assertAlmostEqual(f.params.loc["x", "coef"], -0.1, delta=0.03)
        self.assertAlmostEqual(f.sigma2_group, 0.25, delta=0.07)
        self.assertAlmostEqual(f.sigma2_resid, 0.16, delta=0.03)
        # a between-group effect has a larger SE under the mixed model than naive OLS
        self.assertGreater(f.params.loc["treat", "se"], ols(df, "y", ["treat", "x"]).params.loc["treat", "se"])

    def test_mixed_model_without_group_variance_equals_ols(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": rng.normal(size=400), "g": np.repeat(np.arange(100), 4)})
        df["y"] = 1 + df["x"] + rng.normal(size=400)
        m, o = mixed_lm(df, "y", ["x"], "g"), ols(df, "y", ["x"])
        self.assertAlmostEqual(m.params.loc["x", "coef"], o.params.loc["x", "coef"], delta=0.01)

    def test_collinear_design_is_rejected(self):
        df = pd.DataFrame({"y": [1., 2, 3, 4, 5, 6], "a": [1., 0, 1, 0, 1, 0], "b": [0., 1, 0, 1, 0, 1]})
        with self.assertRaises(ValueError):
            ols(df, "y", ["a", "b"])

    def test_paired_t_matches_scipy(self):
        a, b = np.array([3.1, 2.9, 4.0, 3.3, 2.5]), np.array([2.4, 2.6, 3.1, 3.0, 2.2])
        t = paired_t(a, b)
        ref = st.ttest_rel(a, b)
        self.assertAlmostEqual(t.statistic, ref.statistic)
        self.assertAlmostEqual(t.p, ref.pvalue)

    def test_cronbach(self):
        rng = np.random.default_rng(3)
        latent = rng.normal(size=300)
        df = pd.DataFrame({f"i{k}": latent + rng.normal(0, .5, 300) for k in range(4)})
        self.assertGreater(cronbach_alpha(df, list(df)), 0.85)


if __name__ == "__main__":
    unittest.main()
