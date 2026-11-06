"""Statistical models used in the analysis.

``mixed_lm`` fits a linear mixed model with a random intercept per group
(y_ij = x_ij'b + u_i + e_ij), as used in the paper for Acceptance with a random
intercept per coach. It is estimated by (RE)ML, profiling out the fixed effects
and residual variance and optimising the variance ratio numerically, which keeps
the package free of a statsmodels dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass(frozen=True)
class FitResult:
    params: pd.DataFrame          # coef, se, stat, p
    n_obs: int
    n_groups: int | None
    r2: float                     # adjusted R2 (OLS) or marginal R2 (mixed)
    r2_label: str
    sigma2_resid: float
    sigma2_group: float | None = None
    method: str = ""

    def summary(self) -> str:
        head = f"{self.method}: n = {self.n_obs}" + (f", groups = {self.n_groups}" if self.n_groups else "")
        head += f", {self.r2_label} = {self.r2:.3f}"
        if self.sigma2_group is not None:
            head += f", group variance = {self.sigma2_group:.4f}, residual variance = {self.sigma2_resid:.4f}"
        table = self.params.copy()
        table["sig"] = table["p"].map(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
        return head + "\n" + table.to_string(float_format=lambda v: f"{v:.3f}")


def _design(df: pd.DataFrame, y: str, x: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], pd.Index]:
    data = df[[y, *x]].apply(pd.to_numeric, errors="coerce").dropna()
    names = ["Intercept"] + [c for c in x if data[c].nunique() > 1]
    X = np.column_stack([np.ones(len(data))] + [data[c].to_numpy(float) for c in names[1:]])
    if len(data) and np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("predictors are perfectly collinear; drop one of: " + ", ".join(names[1:]))
    return data[y].to_numpy(float), X, names, data.index


def ols(df: pd.DataFrame, y: str, x: list[str]) -> FitResult:
    yv, X, names, _ = _design(df, y, x)
    n, p = X.shape
    if n <= p:
        raise ValueError(f"not enough observations ({n}) for {p} parameters")
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    s2 = resid @ resid / (n - p)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), n - p)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - p)
    params = pd.DataFrame({"coef": beta, "se": se, "t": t, "p": pv}, index=names)
    return FitResult(params, n, None, float(adj), "adj. R2", float(s2), method="OLS")


def mixed_lm(df: pd.DataFrame, y: str, x: list[str], group: str, reml: bool = True) -> FitResult:
    """Linear mixed model with a random intercept for ``group``."""
    yv, X, names, idx = _design(df, y, x)
    groups = df.loc[idx, group].to_numpy()
    codes, uniq = pd.factorize(groups)
    n, p = X.shape
    G = len(uniq)
    if n <= p or G < 2:
        raise ValueError("not enough observations or groups for a mixed model")
    sizes = np.bincount(codes, minlength=G).astype(float)
    # group sums for the closed-form inverse V_i^-1 = I - c_i J, c_i = lam / (1 + n_i lam)
    Xs = np.zeros((G, p))
    np.add.at(Xs, codes, X)
    ys = np.bincount(codes, weights=yv, minlength=G)
    XtX, Xty, yty = X.T @ X, X.T @ yv, yv @ yv

    def pieces(lam: float):
        c = lam / (1 + sizes * lam)
        A = XtX - (Xs * c[:, None]).T @ Xs           # X' V^-1 X
        b = Xty - (Xs * c[:, None]).T @ ys           # X' V^-1 y
        beta = np.linalg.solve(A, b)
        quad = yty - np.sum(c * ys ** 2) - 2 * beta @ b + beta @ A @ beta  # (y-Xb)' V^-1 (y-Xb)
        logdetV = np.sum(np.log1p(sizes * lam))
        return beta, A, quad, logdetV

    def neg_ll(log_lam: float) -> float:
        lam = np.exp(log_lam)
        beta, A, quad, logdetV = pieces(lam)
        dof = n - p if reml else n
        s2 = quad / dof
        ll = -0.5 * (dof * np.log(s2) + logdetV + dof)
        if reml:
            ll -= 0.5 * np.linalg.slogdet(A)[1]
        return -ll

    res = optimize.minimize_scalar(neg_ll, bounds=(-12, 6), method="bounded", options={"xatol": 1e-6})
    lam = float(np.exp(res.x))
    # boundary check: compare with lam -> 0 (no group variance)
    if neg_ll(-30.0) <= res.fun:
        lam = float(np.exp(-30.0))
    beta, A, quad, _ = pieces(lam)
    s2 = quad / (n - p if reml else n)
    cov = s2 * np.linalg.inv(A)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    s2u = lam * s2
    var_fixed = np.var(X @ beta)
    r2m = var_fixed / (var_fixed + s2u + s2)
    params = pd.DataFrame({"coef": beta, "se": se, "z": z, "p": pv}, index=names)
    return FitResult(params, n, G, float(r2m), "marginal R2", float(s2), float(s2u),
                     method=f"Linear mixed model, random intercept per {group} ({'REML' if reml else 'ML'})")


@dataclass(frozen=True)
class TTest:
    statistic: float
    df: float
    p: float
    n: int
    mean_a: float
    mean_b: float

    def __str__(self) -> str:
        p = "p < .001" if self.p < .001 else f"p = {self.p:.3f}"
        return f"t({self.df:.1f}) = {self.statistic:.2f}, {p}, n = {self.n}"


def paired_t(a, b) -> TTest | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.std(a - b) == 0:
        return None
    r = stats.ttest_rel(a, b)
    return TTest(float(r.statistic), float(len(a) - 1), float(r.pvalue), len(a), float(a.mean()), float(b.mean()))


def welch_t(a, b) -> TTest | None:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    r = stats.ttest_ind(a, b, equal_var=False)
    df = getattr(r, "df", np.nan)
    return TTest(float(r.statistic), float(df), float(r.pvalue), len(a) + len(b), float(a.mean()), float(b.mean()))


def first_component(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Score on the first principal component of standardised columns (factor-score proxy).

    Oriented so that higher scores mean higher values on the first column."""
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    z = (data - data.mean()) / data.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)
    _, _, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    v = vt[0]
    if v[0] < 0:
        v = -v
    score = z.to_numpy() @ v
    return pd.Series((score - score.mean()) / (score.std() or 1.0), index=df.index)


def cronbach_alpha(df: pd.DataFrame, cols: list[str]) -> float:
    items = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    k = len(cols)
    if k < 2 or len(items) < 2:
        return float("nan")
    var_items = items.var(ddof=1).sum()
    var_total = items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - var_items / var_total)) if var_total > 0 else float("nan")
