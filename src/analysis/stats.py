"""Statistical comparison utilities: Diebold-Mariano, Wilcoxon, block bootstrap."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _loss(err: np.ndarray, kind: str = "abs") -> np.ndarray:
    return np.abs(err) if kind == "abs" else err ** 2


def dm_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
            h: int = 1, kind: str = "abs") -> dict:
    """Diebold-Mariano test with Newey-West (Bartlett) long-run variance.

    H0: the two forecasts have equal expected loss.  Negative statistic means
    model 1 has the lower loss.
    """
    d = _loss(y - p1, kind) - _loss(y - p2, kind)
    n = d.size
    dbar = float(d.mean())
    gamma0 = float(np.var(d, ddof=0))
    lrv = gamma0
    for k in range(1, h):
        gk = float(np.cov(d[:-k], d[k:], ddof=0)[0, 1])
        lrv += 2.0 * (1.0 - k / h) * gk
    lrv = max(lrv, 1e-12)
    stat = dbar / np.sqrt(lrv / n)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
    return {"dm_stat": float(stat), "p_value": p, "mean_loss_diff": dbar,
            "n": int(n), "loss": kind,
            "favours": "model1" if stat < 0 else "model2"}


def wilcoxon_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                  kind: str = "abs") -> dict:
    a, b = _loss(y - p1, kind), _loss(y - p2, kind)
    if np.allclose(a, b):
        return {"stat": float("nan"), "p_value": 1.0, "n": int(a.size)}
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(res.statistic), "p_value": float(res.pvalue), "n": int(a.size)}


def block_bootstrap_ci(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                       block: int = 24, n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> dict:
    """Block bootstrap CI for the MAE difference (model2 - model1) and for each MAE."""
    rng = np.random.default_rng(seed)
    n = y.size
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)

    mae1 = np.abs(y - p1)
    mae2 = np.abs(y - p2)
    d1, d2, dd = [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([
            np.arange(s, s + block) for s in rng.choice(starts_pool, n_blocks)
        ])[:n]
        d1.append(mae1[idx].mean())
        d2.append(mae2[idx].mean())
        dd.append(d2[-1] - d1[-1])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "mae1": float(mae1.mean()), "mae2": float(mae2.mean()),
        "mae1_ci": [float(np.percentile(d1, lo)), float(np.percentile(d1, hi))],
        "mae2_ci": [float(np.percentile(d2, lo)), float(np.percentile(d2, hi))],
        "diff_ci": [float(np.percentile(dd, lo)), float(np.percentile(dd, hi))],
        "p_two_sided": float(2 * min((np.array(dd) <= 0).mean(), (np.array(dd) >= 0).mean())),
        "block": block, "n_boot": n_boot,
    }


def paired_summary(y: np.ndarray, preds: dict[str, np.ndarray],
                   ref: str, h: int = 1) -> dict:
    """DM + Wilcoxon + bootstrap for every model against a reference model."""
    out = {}
    for name, p in preds.items():
        if name == ref:
            continue
        out[name] = {
            "dm_abs": dm_test(y, preds[ref], p, h=h, kind="abs"),
            "dm_sq": dm_test(y, preds[ref], p, h=h, kind="sq"),
            "wilcoxon_abs": wilcoxon_test(y, preds[ref], p, kind="abs"),
            "bootstrap": block_bootstrap_ci(y, preds[ref], p, block=24),
        }
    return out
