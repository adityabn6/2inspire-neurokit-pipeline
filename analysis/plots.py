"""
plots.py
--------
Figures for the clean 2Inspire music<->physiology analysis. Reads the per-piece
tables in results/analysis/ and writes PNGs to plots/analysis/.

Figures:
  1. fig_condition_contrast.png  — piece 3 (self-selected) vs sight-reading
     (within-participant mean of pieces 1&2), paired lines + box, per metric.
  2. fig_across_pieces.png       — boxplots of physiology by piece (1/2/3).
  3. fig_coupling_duration.png   — the breathing-cardiac coupling condition effect
     on FULL pieces vs DURATION-MATCHED 60 s windows (is it a length artifact?).
  4. fig_music_by_condition.png  — chroma entropy / mode / tempo / note-density by condition.
  5. fig_key_structure.png       — key counts by piece (pieces 1-2 standardized vs
     piece 3 self-selected) — the structural fact behind the design.

All identifiers are anonymized P-labels.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("plots")
sns.set_theme(context="paper", style="whitegrid", palette="colorblind")

_ROOT = Path(__file__).resolve().parent.parent
ANA = _ROOT / "results" / "analysis"
OUT = _ROOT / "plots" / "analysis"
HRV_GATED = {"rmssd_ms", "mean_hr_bpm", "RSA_P2T_Mean", "pnn50"}


def _load(name):
    p = ANA / name
    return pd.read_csv(p) if p.exists() else None


def paired(df, metric):
    """Return (read_mean, self_selected) arrays paired by participant."""
    d = df.copy()
    if metric in HRV_GATED and "hrv_qc_ok" in d.columns:
        d = d[d["hrv_qc_ok"]]
    sr = d[d.condition == "sight_reading"].groupby("participant_id")[metric].mean()
    ss = d[d.condition == "self_selected"].set_index("participant_id")[metric]
    idx = sr.index.intersection(ss.index)
    a, b = sr.loc[idx].to_numpy(float), ss.loc[idx].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def wilcoxon_p(a, b):
    try:
        return stats.wilcoxon(a, b)[1]
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
def fig_condition_contrast(df):
    metrics = [("rmssd_ms", "RMSSD (ms)"), ("mean_hr_bpm", "mean HR (bpm)"),
               ("RSA_P2T_Mean", "RSA P2T (ms)"), ("resp_rate", "resp rate (Hz)"),
               ("hf_coherence", "HR-resp HF coherence"),
               ("hr_resp_xcorr_peak", "HR-resp xcorr peak")]
    metrics = [m for m in metrics if m[0] in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 4.2))
    for ax, (m, lab) in zip(np.atleast_1d(axes), metrics):
        read, self_ = paired(df, m)
        if len(read) == 0:
            ax.set_title(f"{lab}\n(no data)"); ax.axis("off"); continue
        for r, s in zip(read, self_):
            ax.plot([0, 1], [r, s], color="0.7", lw=0.8, alpha=0.6, zorder=1)
        ax.scatter(np.zeros_like(read), read, color="C0", s=18, zorder=2)
        ax.scatter(np.ones_like(self_), self_, color="C1", s=18, zorder=2)
        ax.boxplot([read, self_], positions=[0, 1], widths=0.25,
                   showfliers=False, patch_artist=False, zorder=3)
        p = wilcoxon_p(self_, read)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["reading\n(P1,2)", "self\n(P3)"])
        ax.set_title(f"{lab}\nWilcoxon p={p:.3f}  n={len(read)}", fontsize=9)
    fig.suptitle("Condition contrast: self-selected vs sight-reading (paired)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_condition_contrast.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig_condition_contrast.png")


def fig_across_pieces(df):
    metrics = [("mean_hr_bpm", "mean HR (bpm)"), ("RSA_P2T_Mean", "RSA P2T (ms)"),
               ("resp_rate", "resp rate (Hz)"), ("rmssd_ms", "RMSSD (ms)")]
    metrics = [m for m in metrics if m[0] in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 4))
    for ax, (m, lab) in zip(np.atleast_1d(axes), metrics):
        d = df.copy()
        if m in HRV_GATED and "hrv_qc_ok" in d.columns:
            d = d[d["hrv_qc_ok"]]
        sns.boxplot(data=d, x="piece", y=m, ax=ax, showfliers=False, width=0.6)
        sns.stripplot(data=d, x="piece", y=m, ax=ax, color="0.3", size=3, alpha=0.5)
        try:
            w = d.pivot_table(index="participant_id", columns="piece", values=m).dropna()
            p = stats.friedmanchisquare(w[1], w[2], w[3])[1] if len(w) >= 6 else np.nan
        except Exception:
            p = np.nan
        ax.set_title(f"{lab}\nFriedman p={p:.3f}", fontsize=9)
        ax.set_xlabel("piece")
    fig.suptitle("Physiology across pieces (1,2 = sight-reading; 3 = self-selected)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_across_pieces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig_across_pieces.png")


def fig_coupling_duration(full, w60):
    metrics = [("hf_coherence", "HF coherence"), ("hr_resp_xcorr_peak", "xcorr peak")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.4))
    for ax, (m, lab) in zip(np.atleast_1d(axes), metrics):
        rows = []
        for df, wlab in [(full, "full piece"), (w60, "matched 60 s")]:
            if df is None or m not in df.columns:
                continue
            read, self_ = paired(df, m)
            p = wilcoxon_p(self_, read)
            for v in read:
                rows.append({"window": wlab, "cond": "reading", "val": v})
            for v in self_:
                rows.append({"window": wlab, "cond": "self", "val": v})
            ax.text(0.02, 0.98 - 0.07 * (wlab == "matched 60 s"),
                    f"{wlab}: p={p:.3f} (n={len(read)})",
                    transform=ax.transAxes, fontsize=8, va="top")
        rdf = pd.DataFrame(rows)
        sns.boxplot(data=rdf, x="window", y="val", hue="cond", ax=ax,
                    showfliers=False, width=0.6)
        ax.set_title(lab, fontsize=10); ax.set_xlabel(""); ax.set_ylabel(lab)
    fig.suptitle("Breathing-cardiac coupling by condition: full vs duration-matched", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_coupling_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig_coupling_duration.png")


def fig_music_by_condition(audio):
    feats = [("chroma_entropy_mean", "chroma entropy (bits)"),
             ("tempo_bpm", "tempo (bpm)"), ("onset_rate_hz", "note density (onset/s)"),
             ("harmonic_ratio", "harmonic ratio")]
    fig, axes = plt.subplots(1, len(feats) + 1, figsize=(3.0 * (len(feats) + 1), 4))
    for ax, (m, lab) in zip(axes[:-1], feats):
        sns.boxplot(data=audio, x="condition", y=m, ax=ax, showfliers=False, width=0.6,
                    order=["sight_reading", "self_selected"])
        sns.stripplot(data=audio, x="condition", y=m, ax=ax, color="0.3", size=3,
                      alpha=0.5, order=["sight_reading", "self_selected"])
        ax.set_title(lab, fontsize=9); ax.set_xlabel(""); ax.set_xticklabels(["read", "self"])
    # minor-mode proportion
    ax = axes[-1]
    prop = audio.groupby("condition")["is_minor"].mean().reindex(["sight_reading", "self_selected"])
    ax.bar(["read", "self"], prop.to_numpy(), color=["C0", "C1"])
    ax.set_ylim(0, 1); ax.set_title("minor-mode fraction", fontsize=9)
    fig.suptitle("Music features by condition", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_music_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig_music_by_condition.png")


def fig_key_structure(audio):
    a = audio.copy()
    a["key_mode"] = a["key"] + " " + a["mode"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, pc in zip(axes, (1, 2, 3)):
        s = a[a.piece == pc]["key_mode"].value_counts().head(8)
        ax.barh(s.index[::-1], s.to_numpy()[::-1], color="C0" if pc < 3 else "C1")
        cond = "sight-reading" if pc < 3 else "self-selected"
        ax.set_title(f"piece {pc} ({cond})", fontsize=10)
    fig.suptitle("Detected key by piece — pieces 1-2 are standardized, piece 3 is free", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_key_structure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig_key_structure.png")


def fig_motor_confound(df):
    """Self-selected has MORE hand motion yet LOWER HR -> not a motor artifact."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    order = ["sight_reading", "self_selected"]
    for ax, (m, lab) in zip(axes, [("motion_energy", "hand-motion energy (acc SD)"),
                                    ("mean_hr_bpm", "mean HR (bpm)")]):
        d = df.copy()
        if m in HRV_GATED and "hrv_qc_ok" in d.columns:
            d = d[d["hrv_qc_ok"]]
        sns.boxplot(data=d, x="condition", y=m, order=order, ax=ax,
                    showfliers=False, width=0.6)
        sns.stripplot(data=d, x="condition", y=m, order=order, ax=ax,
                      color="0.3", size=3, alpha=0.5)
        read, self_ = paired(d, m)
        ax.set_title(f"{lab}\nWilcoxon p={wilcoxon_p(self_, read):.3f}", fontsize=9)
        ax.set_xlabel(""); ax.set_xticklabels(["reading", "self"])
    fig.suptitle("Motor confound: more movement but lower HR in self-selected "
                 "(matched 60 s)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_motor_confound.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig_motor_confound.png")


def fig_music_physio(feat_w60, audio_mapped):
    """Music (mapped+trimmed) vs physiology — piece-3 scatters with OLS fit."""
    phys = feat_w60.copy()
    if "hrv_qc_ok" in phys.columns:
        phys = phys[phys["hrv_qc_ok"]]
    d = audio_mapped.merge(
        phys[["participant_id", "piece", "mean_hr_bpm", "rmssd_ms"]],
        on=["participant_id", "piece"], how="inner")
    d3 = d[d["piece"] == 3]
    pairs = [("chroma_entropy_mean", "mean_hr_bpm", "chroma entropy", "mean HR (bpm)"),
             ("chroma_entropy_mean", "rmssd_ms", "chroma entropy", "RMSSD (ms)"),
             ("onset_rate_hz", "mean_hr_bpm", "note density (onset/s)", "mean HR (bpm)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        s = d3[[x, y]].dropna()
        if len(s) >= 6:
            sns.regplot(data=s, x=x, y=y, ax=ax, scatter_kws=dict(s=22, alpha=0.7),
                        line_kws=dict(color="C1"))
            r, p = stats.pearsonr(s[x], s[y])
            ax.set_title(f"piece 3 (n={len(s)})  r={r:.2f}, p={p:.3f}", fontsize=9)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    fig.suptitle("Music → physiology in the self-selected piece (mapped + trimmed audio)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_music_physio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig_music_physio.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feat = _load("features_by_piece.csv")
    feat_w60 = _load("features_by_piece_w60.csv")
    coup = _load("coupling_by_piece.csv")
    coup_w60 = _load("coupling_by_piece_w60.csv")
    audio = _load("audio_features_by_piece.csv")
    motion_w60 = _load("motion_by_piece_w60.csv")

    def merge(f, c):
        if f is None:
            return f
        if c is not None:
            cc = c[[col for col in c.columns
                    if col not in f.columns or col in ("participant_id", "piece")]]
            f = f.merge(cc, on=["participant_id", "piece"], how="left")
        if audio is not None:
            ac = audio[[col for col in audio.columns
                        if col not in f.columns or col in ("participant_id", "piece")]]
            f = f.merge(ac, on=["participant_id", "piece"], how="left")
        return f

    full = merge(feat, coup)
    w60 = merge(feat_w60, coup_w60)

    if full is not None:
        fig_condition_contrast(full)
        fig_across_pieces(full)
    if full is not None and w60 is not None:
        fig_coupling_duration(full, w60)
    if audio is not None:
        fig_music_by_condition(audio)
        fig_key_structure(audio)
    if w60 is not None and motion_w60 is not None:
        mc = motion_w60[["participant_id", "piece", "motion_energy"]]
        fig_motor_confound(w60.merge(mc, on=["participant_id", "piece"], how="left"))
    audio_mapped = _load("audio_features_mapped.csv")
    if feat_w60 is not None and audio_mapped is not None:
        fig_music_physio(feat_w60, audio_mapped)
    log.info("done -> %s", OUT)


if __name__ == "__main__":
    main()
