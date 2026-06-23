"""
plots.py
---------
Visualise the 2Inspire NeuroKit features.

Two families, both requested by the user:

  1. Box plots by piece — faceted panels of the scalar HRV / RSP features, one
     box per piece (session 1/2/3). ``box_ecg_by_piece.png`` and
     ``box_rsp_by_piece.png`` (RSP hue = thoracic vs abdominal).

  2. Continuous trajectories — for each continuous feature, one figure with
     three subplots (piece 1/2/3). Each subplot overlays the faint per-
     participant trajectory on the 0-100 % of-piece grid plus a bold mean line
     with a 95 % CI band (seaborn computes the band across participants).

Style mirrors eda_open_exploration.py: paper context, whitegrid, colorblind
palette, dpi=150, bbox_inches="tight".
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("plots")

sns.set_theme(context="paper", style="whitegrid", palette="colorblind")

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent  # repo root (modules live in neurokit_pipeline/)
DEFAULT_IN = PROJECT_ROOT / "results" / "neurokit_features"
DEFAULT_OUT = PROJECT_ROOT / "plots" / "neurokit"

# Curated scalar features (intersected with what's actually present).
ECG_BOX_FEATURES = [
    "ECG_Rate_Mean", "HRV_MeanNN", "HRV_RMSSD", "HRV_SDNN", "HRV_pNN50",
    "HRV_HF", "HRV_LF", "HRV_LFHF", "HRV_SD1", "HRV_SD2", "HRV_SampEn",
    "HRV_DFA_alpha1", "ecg_quality_mean",
]
RSP_BOX_FEATURES = [
    "RSP_Rate_Mean", "RSP_Rate_SD", "RSP_Amp_Mean", "RSP_RVT_Mean",
    "RSP_Inhale_Ratio", "RSP_Slope_Inhale_Ratio",
    "RRV_RMSSD", "RRV_SDBB", "RRV_LFHF",
]

PIECES = [1, 2, 3]


# ---------------------------------------------------------------------------
# Box plots by piece
# ---------------------------------------------------------------------------
def box_by_piece(df: pd.DataFrame, features: list[str], title: str,
                 out_path: Path, hue: str | None = None) -> None:
    feats = [f for f in features if f in df.columns and df[f].notna().any()]
    if not feats:
        log.warning("No box features present for %s", title)
        return
    ncols = 4
    nrows = (len(feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6))
    axes = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(feats):
        ax = axes[i]
        sns.boxplot(data=df, x="session_idx", y=feat, hue=hue, ax=ax,
                    width=0.6, showfliers=False)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("piece")
        ax.set_ylabel("")
        if hue is not None:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    for j in range(len(feats), len(axes)):
        axes[j].axis("off")

    if hue is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title=hue, loc="upper right",
                       bbox_to_anchor=(0.995, 0.99), fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Continuous trajectories with mean +/- 95% CI
# ---------------------------------------------------------------------------
def trajectory(df_cont: pd.DataFrame, feature: str, label: str,
               out_path: Path) -> None:
    sub = df_cont[df_cont["feature"] == feature]
    if sub.empty:
        log.warning("No continuous data for %s", feature)
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, piece in zip(axes, PIECES):
        ps = sub[sub["session_idx"] == piece]
        if ps.empty:
            ax.set_title(f"piece {piece} (no data)", fontsize=10)
            ax.set_xlabel("% of piece")
            continue
        # faint per-participant trajectories
        sns.lineplot(data=ps, x="t_pct", y="value", units="participant_id",
                     estimator=None, color="0.6", alpha=0.3, lw=0.8,
                     ax=ax, legend=False)
        # mean + 95% CI across participants
        sns.lineplot(data=ps, x="t_pct", y="value", estimator="mean",
                     errorbar=("ci", 95), color="C0", lw=2.2, ax=ax,
                     label="mean ± 95% CI")
        n = ps["participant_id"].nunique()
        ax.set_title(f"piece {piece}  (n={n})", fontsize=10)
        ax.set_xlabel("% of piece")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(label)
    fig.suptitle(f"{label} — trajectory across the piece", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot 2Inspire NeuroKit features.")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- box plots ----
    ecg_piece = in_dir / "ecg_hrv_piece.csv"
    if ecg_piece.exists():
        box_by_piece(pd.read_csv(ecg_piece), ECG_BOX_FEATURES,
                     "ECG / HRV features by piece", out_dir / "box_ecg_by_piece.png")
    rsp_piece = in_dir / "rsp_features_piece.csv"
    if rsp_piece.exists():
        box_by_piece(pd.read_csv(rsp_piece), RSP_BOX_FEATURES,
                     "Respiration features by piece", out_dir / "box_rsp_by_piece.png",
                     hue="channel")

    # ---- trajectories ----
    ecg_cont = in_dir / "ecg_continuous_1hz.csv"
    if ecg_cont.exists():
        dfc = pd.read_csv(ecg_cont)
        for feat, lab in [("hr_bpm", "Heart rate (bpm)"),
                          ("hrv_rmssd", "Rolling RMSSD (ms)")]:
            trajectory(dfc, feat, lab, out_dir / f"traj_{feat}.png")

    rsp_cont = in_dir / "rsp_continuous_1hz.csv"
    if rsp_cont.exists():
        dfc = pd.read_csv(rsp_cont)
        labels = {"rsp_rate": "Respiration rate (brpm)",
                  "rsp_amplitude": "Respiration amplitude"}
        for ch in ("thoracic", "abdominal"):
            chsub = dfc[dfc["channel"] == ch]
            for feat, lab in labels.items():
                trajectory(chsub, feat, f"{lab} — {ch}",
                           out_dir / f"traj_{feat}_{ch}.png")


if __name__ == "__main__":
    main()
