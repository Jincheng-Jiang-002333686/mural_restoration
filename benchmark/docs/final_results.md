# Final Results — Mural Inpainting Comparison (2026-06-16)

All numbers: full-image metrics, fixed seeded masks (mixed crack+peel, hole
[0.2, 0.3]), one metric implementation (MAT's `cal_psnr_ssim_l1.py` +
`cal_fid_pids_uids.py`), deterministic inference. Rows in
`results/test_metrics.csv`.

## DHMural (10,584 train / 2,649 test, 256x256)

| Model | PSNR ↑ | SSIM ↑ | L1 ↓ | FID ↓ | P-IDS ↑ | U-IDS ↑ |
|---|---|---|---|---|---|---|
| **HMAT v2 (k2000)** | **25.30** | **0.899** | **0.0191** | **12.54** | **0.054** | **0.078** |
| LaMa-fourier | 25.30 | 0.898 | 0.0193 | 17.77 | 0.000 | 0.001 |
| HMAT (pre-fix, k572) | 24.24 | 0.892 | 0.0216 | 15.25 | 0.010 | 0.022 |
| EdgeConnect | 23.75 | 0.881 | 0.0237 | 32.54 | 0.000 | 0.000 |

HMAT v2 matches/beats LaMa on every metric; EdgeConnect is weakest on all,
with ~2.5x worse FID — empirical support for the paper's critique that
edge-guided inpainting struggles on degraded murals.

## Nine-Colored Deer (3,500 train / 500 test, 256x256)

| Model | PSNR ↑ | SSIM ↑ | L1 ↓ | FID ↓ |
|---|---|---|---|---|
| **HMAT v2 (k200)** | **28.47** | **0.919** | **0.0124** | 25.90 |
| HMAT (orig paper) | 27.37 | 0.903 | 0.014 | — |
| MADF (orig paper) | 27.05 | 0.898 | 0.015 | — |
| MAT (orig paper) | 26.53 | 0.898 | 0.015 | — |

HMAT v2 beats the original paper's own HMAT by +1.1 dB PSNR. (P-IDS/U-IDS are
unreliable at n=500 — SVM overfits — so FID is the realism metric here.)
LaMa/EdgeConnect deer rows: not yet run.

## Ablations (DHMural, equal budget 600 kimg from scratch, v2 recipe)

Clean variants built from the fixed `mat.py` (the original `non*.py` ablation
files were confounded). Rows in `results/test_metrics.csv` (full-image) and
`results/hole_metrics.csv` (hole + 8px boundary band).

### Table 3 — component ablation

| Model | PSNR | SSIM | L1 | FID | Hole PSNR | Boundary PSNR |
|---|---|---|---|---|---|---|
| Full | 24.89 | 0.896 | 0.0199 | 18.91 | 18.78 | 26.94 |
| w/o MADF | 24.79 | 0.895 | 0.0202 | 18.82 | 18.68 | 26.74 |
| w/o Mask-Guided Style | 24.87 | 0.896 | 0.0200 | 18.70 | 18.77 | 26.92 |
| w/o Teacher-Forcing | 24.90 | 0.896 | 0.0199 | 18.82 | 18.79 | 26.96 |

Component contribution: **MADF ≫ Mask-Guided Style > Teacher-Forcing (≈0)**.
MADF is the key architectural piece, strongest at the boundary (+0.20 dB).
Teacher-Forcing removed (no measurable effect).

### Table 4 — loss ablation (TF-free architecture)

| Loss | PSNR | SSIM | L1 | FID | Hole PSNR | Hole SSIM |
|---|---|---|---|---|---|---|
| New (hole-L1 + FM + HRF) | 24.90 | 0.896 | 0.0199 | 18.82 | 18.79 | 0.593 |
| Old (adv + VGG only) | 23.78 | 0.884 | 0.0230 | 18.45 | 17.67 | 0.548 |
| **Δ (loss overhaul)** | **+1.12** | **+0.012** | **−14%** | −0.37 | **+1.12** | **+0.045** |

The loss overhaul is worth **+1.12 dB PSNR — ~11x the best architecture
component** — the dominant driver of HMAT v2's gains.

## How HMAT v2 was obtained

Starting point (the original paper's run): HMAT trailed MADF, reported
23.11/0.868/0.025 on DHMural — but that was an undertrained checkpoint with a
broken eval (110-image in-run metric) and a buggy objective. The fixes
(`docs/hmat_code_review.md`, `docs/sota_analysis_hmat.md`):

1. Token-validity mask for the masked attention derived from the real hole
   mask (was inverted); content-based fallback for fully-masked windows.
2. Hole-normalized L1 actually entered the objective (final + stage-1);
   added discriminator feature matching (w 10) and HRF ADE20k perceptual loss
   (w 10); VGG19 removed.
3. Deterministic evaluation (const noise, seeded z + masks, full test set).
4. MADF reparametrized as 16+1 basis convolutions (exact identity, ~3.7x
   faster, batch-gpu 4 on one RTX 3090).
5. TTUR (G 1e-3 / D 1e-4), R1 gamma 2, style mixing off.

Trained from the k572 checkpoint: run 00012 (k572->k1572 effective) then
continuation run 00014 (->k2572 effective, ~k2000 of the new recipe). PSNR
converged at ~25.39 in-run / 25.30 offline; FID plateaued ~12.6.

## Reproduce

- HMAT v2 train: `MAT/README.md` (v2 recipe command).
- Benchmark any snapshot: `scripts/hmat_predict_fixed.py <snap> <out> [gt] [masks]`
  then `scripts/eval_outputs.py <out> <gt> <name> [masks]`.
- LaMa: `docs/lama_training_protocol.md`. EdgeConnect: `scripts/run_edgeconnect.sh`.
