# 2Inspire — feasible hypotheses for music ↔ physiology

A grounded brainstorm of what we can actually ask of this dataset, given what the
data supports and what the literature (the 6 papers in `Assets/`) has established.
Companion to the clean `analysis/` pipeline.

---

## What the data actually is (decisive for feasibility)

- **28 participants × 3 pieces.** Within-subject, repeated measures.
- **Pieces 1–2 are STANDARDIZED sight-reading** — the same scores for everyone
  (audio confirms: 23/28 and 19/28 detected as **G minor**). **Piece 3 is
  self-selected** (diverse keys, 39% minor, higher chroma entropy). This is the
  single most important structural fact:
  - The musical *content* is held constant across people in the reading condition.
  - **Between-person music variance lives almost entirely in piece 3.** A
    music-feature → physiology association is only estimable where music varies —
    so pooling correlations across all pieces (as the original notebooks did)
    mixes a no-variance condition with a high-variance one.
- **Piece durations vary 50 s – 16 min** (sight-reading ~1–2 min; self-selected
  often far longer). Duration is a confound for any HRV/RSA contrast.
- **No sub-second audio↔physiology sync** (integer-second timestamps, no shared
  pulse). → only **piece/condition-level** music↔physiology questions are valid;
  event-locked coupling is not.
- **HRV must be artifact-aware.** Even device RR is corrupted on some sessions
  (e.g. P2 RMSSD ≈ 400 ms); the clean pipeline corrects (Kubios) and QC-gates
  (68/77 sessions usable). **On short pieces, use mean HR / RMSSD / RSA; LF and
  LF/HF are unreliable** (Soliński 2024 avoid HRV entirely on 2–3 min pieces;
  Bernardi 2006 must correct LF/HF for respiration).
- **HR has a large motor component** when *playing* (Iñesta 2008: 72–85 % of max
  HR; piano ≈ 2.5 METs). Movella hand accelerometers exist → a movement covariate
  is available (parked, below).

## What the literature says we should expect

- **Music modulates the ANS, but effects are small and tempo/loudness-dominated**
  (Koelsch & Jäncke 2015; Bernardi 2006). Exciting/faster music ↑HR, ↓SDNN.
- **Performing modulates autonomics more than listening**, and **flow tracks
  HRV**: flow ↔ ↑HF-HRV pre-performance and ↓LF during; total power drops during
  performance (Jha 2022 — *same design*: 2 standardized + 1 self-selected piece).
  Notably, Jha found flow **highest in the standardized** piece.
- **Respiration drives HRV via RSA**; slower breathing → larger RSA
  (Vickhoff 2013).
- **The tractable modeling target on short pieces is the RR interval / time-domain
  HRV, predicted from loudness, tempo, note density** (Soliński 2024).

---

## Recommended program: 1 primary + 2 secondary (pre-register)

### H1 — PRIMARY. Self-selected vs sight-reading shifts autonomic balance
*(condition contrast; cleanest, best-powered, content-controlled)*

- **Claim.** Relative to standardized sight-reading (pieces 1–2), self-selected
  performance (piece 3) changes vagally-mediated cardiac control. **Directional
  prediction:** ↑RMSSD and ↑RSA, ↓mean HR (reduced visual/cognitive load,
  familiarity, flow). The opposite is equally interesting and publishable
  (self-selected pieces are more arousing/complex → sympathetic ↑; cf. Jha's
  flow-in-standardized result).
- **Readouts.** RMSSD, RSA (P2T, Porges–Bohrer), mean HR, respiration rate, HR–
  respiration coupling. (Not LF/HF.)
- **Model.** Paired Wilcoxon, piece 3 vs the within-participant mean of pieces
  1 & 2, with a within-pair sign-flip permutation null; Friedman across the three
  pieces; rank-biserial effect sizes; BH-FDR over the readout family. n ≈ 22–25
  QC-ok pairs. (Implemented in `stats_models.condition_contrast`.)
- **Confounds & controls.**
  - *Duration* (piece 3 longer): recompute on a **duration-matched first-60 s
    window** of every piece as the primary, full-piece as sensitivity. (Next
    code step — `build_features` currently uses full piece.)
  - *Repertoire*: content genuinely differs — that *is* the condition; report it.
  - *Motor effort*: longer/virtuosic piece 3 → more movement → ↑HR. Add the
    Movella motion covariate (parked) as a robustness check.
- **Why it's good.** Within-subject; the reading condition controls musical
  content; directly answers the PI's "piece 3 vs reading" instinct; maps onto a
  published design (Jha 2022) so it's interpretable and contrastable.

### H2 — SECONDARY. Harmonic/tonal features predict arousal (piece-3-only)
*(the novel "chroma → ANS" angle, done at a defensible granularity)*

- **Claim.** Across self-selected pieces, greater harmonic complexity (chroma
  entropy), minor mode, and lower tonal stability are associated with higher
  arousal (↑HR, ↓RMSSD), **after controlling for tempo, loudness, and note
  density** (so the *tonal* contribution — the genuinely novel part — is isolated;
  Bernardi/Soliński show tempo/loudness usually dominate).
- **Why piece-3-only.** Pieces 1–2 have ~no between-person music variance, so the
  association is only estimable in piece 3 (n ≈ 25). A secondary piece-level
  MixedLM across all pieces with `condition` + `duration` covariates uses the
  within-piece-3 variance plus the condition contrast.
- **Model.** `outcome ~ tonal_feature + tempo + rms + (1|participant)` (MixedLM);
  BH-FDR; report β and CI. Frame as **exploratory / hypothesis-generating** —
  n ≈ 25 gives low power for small effects. (Implemented in
  `stats_models.lmm_music_physio`.)
- **Novelty.** Tonal/harmonic structure → ANS in *performers* is essentially
  unstudied; most prior work is tempo/loudness in *listeners*.

### H3 — SUPPORTING. Breathing–cardiac coupling during instrumental playing
*(mechanism + a parasympathetic readout robust to the sync problem)*

- **Claim.** Characterize RSA and HR–respiration coupling during (non-singing)
  piano performance and test condition modulation (e.g. slower breathing / larger
  RSA in self-selected vs sight-reading). Respiration rate as a mediator
  (Vickhoff 2013: slower breathing → larger RSA).
- **Readouts.** RSA (nk.hrv_rsa), HF coherence + cross-correlation lag
  (`coupling.py`), respiration rate/amplitude.
- **Why it's safe.** Both signals come from the *same* device → coupling does not
  depend on audio sync. Breathing during piano (hands occupied, breathing
  "free", unlike singers/wind players) is under-characterized.

---

## Preliminary results from the clean pipeline (exploratory — not confirmatory)

Running `analysis/stats_models.py` on the 28×3 tables already gives a steer
(n ≈ 22–24 after HRV QC; BH-FDR within each family):

- **Strongest signal — breathing–cardiac coupling is weaker in self-selected than
  sight-reading:** HF coherence rank-biserial **−0.74 (p_fdr ≈ 0.005)** and HR–
  respiration cross-correlation peak **−0.71 (p_fdr ≈ 0.005)**. ⚠️ **Confounded by
  duration** (piece 3 is much longer and less stationary) — must be re-tested on
  duration-matched 60 s windows before it's believed. This is an H1/H3 result.
- **Across-piece differences (Friedman, p_fdr):** mean HR 0.004, respiration rate
  0.0005, RSA P2T 0.05; RMSSD n.s. Physiology clearly varies by piece.
- **Condition contrast on cardiac metrics:** RSA P2T higher in self-selected is
  only a trend (rank-biserial +0.47, p_fdr 0.11); RMSSD and mean HR n.s.
- **Cautionary (the audit's value, live):** an apparent "chroma entropy → HR"
  effect in piece 3 was **FDR-significant under a (mis-specified) mixed model with
  one observation per participant, and vanished (p≈0.15) under the correct OLS**.
  Adjusted for tempo+loudness it is at best a weak trend (p≈0.07). **H2 (chroma →
  ANS) is weak** on current evidence — keep it exploratory.

Net: the **condition contrast + coupling** axis (H1/H3) carries the signal; the
**chroma→physiology** axis (H2) does not survive correct modeling yet. Resolve the
duration confound first.

## Explicitly NOT feasible with this dataset

- **Event-locked music→beat coupling** (a chord/onset → a specific RR change):
  needs sub-second sync we don't have.
- **Continuous RR ~ continuous music-feature modeling à la Soliński 2024:** their
  method needs time-aligned continuous loudness/tempo/note-density vs RR; we lack
  the alignment. (Only piece-level aggregates are valid here.)
- **Sympathovagal LF/HF claims on the ~1–2 min sight-reading pieces.**

## Parked (high-novelty, needs more work)

- **Hand-motion as motor-confound control / expressivity signal.** Use Movella
  accelerometer energy to separate effort-driven from arousal-driven HR
  (Iñesta 2008), or as an expressive-gesture covariate for H2. Strongest novelty
  ("no one has looked at this"), but needs Movella processing + careful
  within-device temporal handling first.

---

## Suggested next decision

Lead with **H1** (condition contrast) as the headline result, reported with **H3**
(RSA/coupling) as the mechanistic readout; carry **H2** (chroma→ANS, piece-3) as
the novel exploratory extension. Immediate code step before running H1 as the
confirmatory test: add the **duration-matched 60 s window** option to
`build_features.py` so the piece-3-vs-reading contrast isn't a length artifact.
Then commit to the motor-confound (Movella) analysis only if H1 survives the
movement covariate.
