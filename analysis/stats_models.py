"""
stats_models.py
---------------
Correct within-subject statistics for the 2Inspire per-piece tables.

This respects the design: repeated-measures structure (each participant supplies
all 3 pieces) is modelled with Friedman / paired / mixed models rather than
independent tests; one observation per participant x piece (no per-time-bin
pseudoreplication); multiple-comparison correction within each test family.

What this module does instead:
  * One observation = one participant x piece. HRV outcomes use only
    ``hrv_qc_ok`` rows (drops the artifact-corrupted device-RR sessions, e.g. P2).
  * CONDITION CONTRAST: self_selected (piece 3) vs sight_reading (mean of pieces
    1 & 2 per participant) — PAIRED Wilcoxon signed-rank + rank-biserial effect
    size, with a within-participant permutation null. Plus Friedman across the 3
    pieces when piece-1/2 are kept separate.
  * MUSIC -> PHYSIOLOGY: linear mixed model ``outcome ~ predictor + duration_s +
    (1|participant)`` (statsmodels MixedLM). Run on piece-3-only (where music
    actually varies across participants — pieces 1/2 are the same standardized
    score) and on all pieces with duration as a covariate.
  * Benjamini-Hochberg FDR across each family of tests; effect sizes always
    reported alongside p-values.

Output: results/analysis/stats_results.csv  (+ console summary)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("stats")

_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = _ROOT / "results" / "analysis"

# Outcomes that require a clean device-RR HRV (gated on hrv_qc_ok)
HRV_OUTCOMES = {"rmssd_ms", "mean_hr_bpm", "pnn50", "RSA_P2T_Mean", "RSA_PorgesBohrer"}


# ---------------------------------------------------------------------------
def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size for Wilcoxon signed-rank."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp = r[d > 0].sum()
    rm = r[d < 0].sum()
    return float((rp - rm) / r.sum())


def condition_contrast(df: pd.DataFrame, metric: str, n_perm: int = 10000) -> dict:
    """Paired self_selected (piece 3) vs sight_reading (mean of pieces 1,2)."""
    d = df.copy()
    if metric in HRV_OUTCOMES and "hrv_qc_ok" in d.columns:
        d = d[d["hrv_qc_ok"]]
    sr = (d[d["condition"] == "sight_reading"]
          .groupby("participant_id")[metric].mean())
    ss = (d[d["condition"] == "self_selected"]
          .set_index("participant_id")[metric])
    common = sr.index.intersection(ss.index)
    a = ss.loc[common].to_numpy(dtype=float)   # self-selected
    b = sr.loc[common].to_numpy(dtype=float)   # sight-reading
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    res = {"test": "condition_contrast", "metric": metric, "n_pairs": n,
           "median_self": np.nan, "median_read": np.nan,
           "effect_rank_biserial": np.nan, "stat": np.nan, "p_value": np.nan}
    if n < 6:
        return res
    res["median_self"] = float(np.median(a))
    res["median_read"] = float(np.median(b))
    res["effect_rank_biserial"] = rank_biserial(a, b)
    try:
        w, p = stats.wilcoxon(a, b)
        res["stat"], res["p_value"] = float(w), float(p)
    except ValueError:
        pass
    # within-pair sign-flip permutation null on the mean difference
    diff = a - b
    obs = np.abs(diff.mean())
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = np.abs((signs * diff).mean(axis=1))
    res["p_perm"] = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return res


def friedman_across_pieces(df: pd.DataFrame, metric: str) -> dict:
    """Friedman test across pieces 1/2/3 (non-parametric repeated measures)."""
    d = df.copy()
    if metric in HRV_OUTCOMES and "hrv_qc_ok" in d.columns:
        d = d[d["hrv_qc_ok"]]
    wide = d.pivot_table(index="participant_id", columns="piece", values=metric)
    wide = wide.dropna(subset=[1, 2, 3]) if set([1, 2, 3]).issubset(wide.columns) else wide.iloc[0:0]
    res = {"test": "friedman_pieces", "metric": metric, "n_pairs": len(wide),
           "stat": np.nan, "p_value": np.nan}
    if len(wide) >= 6:
        s, p = stats.friedmanchisquare(wide[1], wide[2], wide[3])
        res["stat"], res["p_value"] = float(s), float(p)
    return res


def music_physio_model(df: pd.DataFrame, outcome: str, predictors: list[str],
                       piece3_only: bool, covariates: tuple[str, ...] = ()) -> dict:
    """Relate music feature(s) to a physiological outcome.

    piece3_only=True  -> ONE row per participant (the self-selected piece), so
                         there is no within-subject nesting: plain OLS. (A mixed
                         model with 1 obs/group is degenerate.)
    piece3_only=False -> all pieces, repeated measures: MixedLM with a
                         participant random intercept (+ duration covariate).

    The first predictor is the one of interest; remaining predictors + covariates
    are adjustment terms. Reports the standardized beta and p for predictors[0].
    """
    d = df.copy()
    if outcome in HRV_OUTCOMES and "hrv_qc_ok" in d.columns:
        d = d[d["hrv_qc_ok"]]
    if piece3_only:
        d = d[d["piece"] == 3]
    terms = list(dict.fromkeys(predictors + list(covariates)))
    add_dur = (not piece3_only) and "duration_s" in d.columns
    cols = [outcome, "participant_id"] + terms + (["duration_s"] if add_dur else [])
    d = d[[c for c in cols if c in d.columns]].replace([np.inf, -np.inf], np.nan).dropna()

    tag = "p3_ols" if piece3_only else "all_lmm"
    adj = "+".join(predictors[1:] + list(covariates)) or "-"
    res = {"test": tag, "outcome": outcome, "predictor": predictors[0],
           "adjusted_for": adj, "n_obs": len(d),
           "n_groups": d["participant_id"].nunique(),
           "beta": np.nan, "p_value": np.nan}
    if len(d) < 8 or (not piece3_only and res["n_groups"] < 6):
        return res

    d = d.copy()
    rhs = []
    for i, p in enumerate(terms):
        d[f"_x{i}"] = (d[p] - d[p].mean()) / (d[p].std() + 1e-9)
        rhs.append(f"_x{i}")
    if add_dur:
        rhs.append("duration_s")
    formula = f"{outcome} ~ " + " + ".join(rhs)
    try:
        if piece3_only:
            m = smf.ols(formula, d).fit()
        else:
            m = smf.mixedlm(formula, d, groups=d["participant_id"]).fit(reml=False)
        res["beta"] = float(m.params["_x0"])
        res["p_value"] = float(m.pvalues["_x0"])
    except Exception as exc:  # noqa: BLE001
        log.warning("model failed (%s ~ %s, %s): %s", outcome, predictors[0], tag, exc)
    return res


def condition_lmm(df: pd.DataFrame, outcome: str, adjust: tuple[str, ...] = ()) -> dict:
    """outcome ~ condition (+ adjust) + (1|participant) across all pieces.

    Reports the self_selected-vs-sight_reading coefficient. Used to test whether
    a condition effect survives adjusting for hand-motion energy (motor confound).
    """
    d = df.copy()
    if outcome in HRV_OUTCOMES and "hrv_qc_ok" in d.columns:
        d = d[d["hrv_qc_ok"]]
    cols = [outcome, "participant_id", "condition"] + list(adjust)
    d = d[[c for c in cols if c in d.columns]].replace([np.inf, -np.inf], np.nan).dropna()
    res = {"test": "condition_lmm", "outcome": outcome,
           "adjusted_for": "+".join(adjust) or "-", "n_obs": len(d),
           "n_groups": d["participant_id"].nunique(), "beta": np.nan, "p_value": np.nan}
    if len(d) < 10 or res["n_groups"] < 6:
        return res
    d = d.copy()
    d["cond_self"] = (d["condition"] == "self_selected").astype(float)
    rhs = ["cond_self"]
    for i, a in enumerate(adjust):
        d[f"_a{i}"] = (d[a] - d[a].mean()) / (d[a].std() + 1e-9)
        rhs.append(f"_a{i}")
    try:
        m = smf.mixedlm(f"{outcome} ~ " + " + ".join(rhs), d,
                        groups=d["participant_id"]).fit(reml=False)
        res["beta"] = float(m.params["cond_self"])     # self - reading
        res["p_value"] = float(m.pvalues["cond_self"])
    except Exception as exc:  # noqa: BLE001
        log.warning("condition_lmm failed (%s, adj=%s): %s", outcome, adjust, exc)
    return res


def fdr(results: list[dict], family_key: str) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["family"] = family_key
    mask = df["p_value"].notna()
    df["p_fdr"] = np.nan
    if mask.any():
        df.loc[mask, "p_fdr"] = multipletests(
            df.loc[mask, "p_value"], method="fdr_bh")[1]
    return df


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Within-subject stats for 2Inspire.")
    ap.add_argument("--features", default=str(ANALYSIS / "features_by_piece.csv"))
    ap.add_argument("--audio", default=str(ANALYSIS / "audio_features_by_piece.csv"))
    ap.add_argument("--coupling", default=str(ANALYSIS / "coupling_by_piece.csv"))
    ap.add_argument("--motion", default=str(ANALYSIS / "motion_by_piece.csv"))
    ap.add_argument("--out", default=str(ANALYSIS / "stats_results.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    for extra in (args.audio, args.coupling, args.motion):
        if Path(extra).exists():
            e = pd.read_csv(extra)
            cols = [c for c in e.columns
                    if c not in df.columns or c in ("participant_id", "piece")]
            df = df.merge(e[cols], on=["participant_id", "piece"], how="left")

    out_frames = []

    # Family 1: condition contrast (self-selected vs sight-reading)
    fam1 = [condition_contrast(df, m) for m in
            ["rmssd_ms", "mean_hr_bpm", "RSA_P2T_Mean", "resp_rate",
             "hf_coherence", "hr_resp_xcorr_peak"] if m in df.columns]
    out_frames.append(fdr([r for r in fam1 if r["n_pairs"] >= 6], "condition_contrast"))

    # Family 2: Friedman across pieces
    fam2 = [friedman_across_pieces(df, m) for m in
            ["rmssd_ms", "mean_hr_bpm", "RSA_P2T_Mean", "resp_rate"] if m in df.columns]
    out_frames.append(fdr([r for r in fam2 if r["n_pairs"] >= 6], "friedman_pieces"))

    # Family 3: music -> physiology, piece-3-only (OLS; that's where music varies)
    if "chroma_entropy_mean" in df.columns:
        fam3 = []
        for outcome in ["rmssd_ms", "mean_hr_bpm", "RSA_P2T_Mean"]:
            if outcome not in df.columns:
                continue
            # univariate screen
            for pred in ["chroma_entropy_mean", "is_minor", "tempo_bpm",
                         "onset_rate_hz", "rms_mean"]:
                if pred in df.columns:
                    fam3.append(music_physio_model(df, outcome, [pred], piece3_only=True))
            # confirmatory: tonal feature ADJUSTED for tempo + loudness (the novel
            # bit is the tonal contribution beyond intensity)
            if {"tempo_bpm", "rms_mean"}.issubset(df.columns):
                fam3.append(music_physio_model(
                    df, outcome, ["chroma_entropy_mean"], piece3_only=True,
                    covariates=("tempo_bpm", "rms_mean")))
        out_frames.append(fdr([r for r in fam3 if pd.notna(r["p_value"])], "music_p3"))

    # Family 4: music -> physiology across ALL pieces (MixedLM, random intercept)
    if "chroma_entropy_mean" in df.columns:
        fam4 = []
        for outcome in ["mean_hr_bpm", "rmssd_ms"]:
            for pred in ["chroma_entropy_mean", "tempo_bpm", "onset_rate_hz"]:
                if outcome in df.columns and pred in df.columns:
                    fam4.append(music_physio_model(df, outcome, [pred], piece3_only=False))
        out_frames.append(fdr([r for r in fam4 if pd.notna(r["p_value"])], "music_allpieces"))

    # Family 5: does the condition effect survive the hand-motion (motor) covariate?
    if "motion_energy" in df.columns:
        fam5 = []
        for outcome in ["mean_hr_bpm", "rmssd_ms", "RSA_P2T_Mean"]:
            if outcome in df.columns:
                fam5.append(condition_lmm(df, outcome))                        # unadjusted
                fam5.append(condition_lmm(df, outcome, adjust=("motion_energy",)))  # adjusted
        # is motion itself higher in self-selected? (the confound's premise)
        fam5.append(condition_contrast(df, "motion_energy"))
        out_frames.append(fdr([r for r in fam5 if pd.notna(r.get("p_value"))],
                              "motor_confound"))

    results = pd.concat(out_frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    log.info("Wrote %s", out)
    show = [c for c in ["family", "metric", "outcome", "predictor", "adjusted_for",
                        "test", "n_pairs", "n_obs", "effect_rank_biserial", "beta",
                        "p_value", "p_perm", "p_fdr"] if c in results.columns]
    print(results[show].to_string(index=False))


if __name__ == "__main__":
    main()
