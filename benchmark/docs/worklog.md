# Mural Inpainting — Experiment Worklog

Chronological record of the HMAT improvement + benchmarking effort
(2026-06-09 → 2026-06-24). Protocol throughout: 256x256, fixed seeded masks
(mixed crack+peel, hole [0.2, 0.3]), full-image metrics with MAT's own
implementations (`cal_psnr_ssim_l1.py` + `cal_fid_pids_uids.py`), deterministic
inference. All rows in `results/test_metrics.csv` (full image) and
`results/hole_metrics.csv` (hole + boundary band).

## 1. LaMa baseline (the trigger)

Trained LaMa-fourier on the same DHMural data/masks in a dedicated `lama` conda
env (py3.8, torch 1.8.1+cu111; the official py3.6/cu102 env doesn't support the
RTX 3090). 55 epochs ≈ 582 kimg. Final fixed-mask benchmark:
PSNR 25.30 / SSIM 0.898 / L1 0.0193 / FID 17.77. See
`docs/lama_training_protocol.md`.

## 2. HMAT code review (why it was losing)

Multi-agent review (47 findings, all adversarially verified) of the MAT-codebase
HMAT. Critical issues (`docs/hmat_code_review.md`):
- Transformer token-validity mask was derived from learned MADF features and was
  **inverted** on the trained checkpoint (masked attention suppressed *known*
  context).
- The L1 reconstruction loss was computed "just for showing" and **never
  backpropagated** (paper claimed weight 1).
- Evaluation was stochastic (random z, always-on dropout) and the in-run
  fid2649/psnr2649 metrics actually ran on only **110 of 2,649 images** (proven
  from run-era bytecode).
- The run was aborted at 572/3000 kimg; baselines were not apples-to-apples.

Honest re-benchmark of the original checkpoint on the fixed protocol:
24.24 / 0.892 / 0.0216 / FID 15.25 (the in-run FID ~139 was a 110-image
artifact).

## 3. SOTA analysis (what to change)

5 web-research agents + 3 adversarial judges (`docs/sota_analysis_hmat.md`).
Verdict: **keep the architecture, overhaul the objective and recipe.** The gap
to LaMa was recipe/loss, not structure.

## 4. HMAT v2 fixes (all verified before any GPU-days)

- Token-validity mask now from the real hole mask; all-masked-window fallback.
- Hole-normalized L1 actually enters the objective (final + stage-1); added
  discriminator **feature-matching** (w 10) and **HRF ADE20k perceptual** loss
  (w 10); VGG19 removed.
- Deterministic evaluation (const noise, seeded z + masks, full 2,649 set).
- MADF reparametrized as 16+1 **basis convolutions** (exact identity: full-G
  diff 5e-7, 0 at uint8; ~3.7x faster, batch-gpu 4 on one 3090).
- TTUR (G 1e-3 / D 1e-4), R1 gamma 2, style mixing off.

## 5. HMAT v2 training

Resumed from the original k572 checkpoint: run 00012 (+1000 kimg) then
continuation run 00014 (+1000 kimg) = effective ~k2000 of the new recipe.
PSNR climbed 24.25 → 25.18 (in-run), converging ~25.39 in-run / 25.30 offline;
FID 15.3 → 12.5. Both axes improved together throughout.

## 6. Final SOTA tables — see `docs/final_results.md`

HMAT v2 (k2000) matches/beats LaMa on every DHMural metric and wins FID by 5.2;
beats EdgeConnect (self-trained 4th baseline) decisively; on Nine-Colored Deer
beats the original paper's HMAT/MADF/MAT.

## 7. Ablations (DHMural, equal budget = 600 kimg from scratch, v2 recipe)

Clean variants built from the *fixed* `mat.py` (the original `nonMADF/nonTF/
nonMGStyle.py` files were confounded — they re-introduced MAT noise injection).

**Component ablation (Table 3).** Full-image and hole/boundary-band:

| Removed | full PSNR | ΔPSNR | hole PSNR | Δhole | boundary PSNR | Δbnd |
|---|---|---|---|---|---|---|
| (full) | 24.887 | — | 18.781 | — | 26.939 | — |
| MADF | 24.787 | +0.100 | 18.682 | +0.100 | 26.740 | +0.199 |
| Mask-Guided Style | 24.871 | +0.016 | 18.765 | +0.016 | 26.920 | +0.019 |
| Teacher-Forcing | 24.896 | −0.009 | 18.790 | −0.009 | 26.957 | −0.018 |

Ranking: **MADF ≫ Mask-Guided Style > Teacher-Forcing (≈0)**. MADF is the key
architectural component, strongest at the valid/damaged boundary (+0.20 dB) —
its designed role. **Teacher-Forcing contributes nothing and was removed** from
the go-forward architecture. (The paper's larger deer ablation effects were
partly the confound above.)

**Loss ablation (Table 4).** TF-free architecture, old loss (adv + VGG only)
vs new loss (hole-L1 + FM + HRF), 600 kimg each:

| | full PSNR | SSIM | L1 | FID | hole PSNR | hole SSIM |
|---|---|---|---|---|---|---|
| new loss | 24.896 | 0.8962 | 0.0199 | 18.82 | 18.790 | 0.5930 |
| old loss | 23.776 | 0.8842 | 0.0230 | 18.45 | 17.671 | 0.5476 |
| **Δ (loss)** | **+1.12** | **+0.012** | **−14%** | −0.37 | **+1.12** | **+0.045** |

**The loss overhaul is worth +1.12 dB — ~11x the best architecture component.**
This is the dominant lever behind HMAT v2's leap from 23.11 (paper) to 25.30.

## 8. Decisions taken

- Removed Teacher-Forcing (no measurable contribution).
- Headline model = HMAT v2 k2000 (kept the trained checkpoint; TF is harmless
  given ≈0 effect).
- Deferred (not run): MAT-original + new-loss control (the experiment that would
  further isolate architecture-vs-loss credit); clean deer component ablation.

## Repro pointers

- Train HMAT v2: `MAT/README.md` (v2 recipe). Loss toggles: `--pr/--l1w/--fmw/--hrfw`.
- Ablation: `scripts/run_ablation.sh [kimg]`; loss ablation: `scripts/post_ablation_loss.sh`.
- Benchmark: `scripts/hmat_predict_fixed.py` + `scripts/eval_outputs.py`
  (+ `scripts/hole_metrics.py` for hole/boundary).
- Ablation generators: `networks/mat_nomadf.py`, `mat_notf.py`, `mat_nomgstyle.py`.
