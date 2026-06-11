# Does HMAT's structure need adjustment? — SOTA analysis (2026-06-10)

Method: 5 research agents with live web search (general inpainting SOTA 2022-26,
mural/heritage-specific work, small-data training recipes, loss landscape,
component-by-component audit of HMAT), merged into ranked recommendations, then
stress-tested by 3 adversarial judges (engineering feasibility on this codebase,
citation/evidence validity, benchmark impact per GPU-day). Items the judges
refuted or demoted are marked.

## Verdict

**Keep the architecture; overhaul the objective and inference protocol.**
For the constraints (10.5k domain images, one RTX 3090, bit-exact valid-pixel
preservation, single forward pass) HMAT's GAN-hybrid design class is where the
faithful-restoration literature still lives (CMAMRNet BMVC'25, SGTB, MGTNet).
The diffusion/flow frontier (DiffuMural, PixelHacker, FLUX-Fill, LDM-inpaint)
wins FID on big benchmarks but needs orders of magnitude more data/compute,
mostly operates in VAE latents that break the valid-pixel guarantee, and rarely
reports PSNR — the paper should position against it explicitly, not ignore it.

The ~1 dB PSNR deficit vs LaMa is mostly **recipe, not structure**:
1. LaMa's pixel-fidelity engines are a weight-100 discriminator feature-matching
   loss + a high-receptive-field perceptual loss (ADE20k dilated-ResNet50);
   HMAT has neither and uses the low-RF VGG19 loss LaMa's ablation showed
   inferior (FID 5.69 HRF vs 6.29 VGG). Notably, LaMa applies **zero** direct
   pixel loss to the hole (l1 weight_missing=0) — its 1 dB PSNR win comes
   entirely from FM+HRF+adv, the cleanest evidence that loss design, not raw L1
   weight, closes the gap.
2. LaMa trains D 10x slower than G (1e-3/1e-4); HMAT runs both at 2e-3 (double
   even MAT's official lr, at half the batch) with R1 gamma 10 where the ADA
   heuristic for 256px/batch-16 is ~0.8 — a discriminator-dominant recipe that
   maximizes texture (FID) and pays PSNR.
3. HMAT's second stage blends 50% of the 16x16 bottleneck features with
   w-derived noise and injects decoder noise — diversity machinery inherited
   from MAT that a deterministic restoration system does not want. (MAT's own
   ablation: this buys diversity/P-IDS, not fidelity.)

## Action plan (judge-corrected)

### Now (locked into the next retrain — one combined run, then one follow-up)
1. **Determinism sweep on the existing checkpoint** (hours, zero GPU-days,
   running): truncation_psi {1, 0.5, 0} x mul_map {0.5-const, 1.0} on the fixed
   2,649 benchmark → perception-distortion curve (cite Blau & Michaeli CVPR'18)
   + possibly free PSNR.
2. **Discriminator feature-matching loss (w≈10) + HRF perceptual (w≈3, ratio
   adv:HRF:FM = 1:3:10 from LaMa's verified config — transfer ratios, never
   absolute weights)**. The ADE20k ResNet50dilated weights are already on disk
   (lama_mat_comparison/torch_home) and LaMa's ResNetPL/feature-matching code is
   vendored locally. Keep hole-L1 at 10 (halve *it*, not FM, if pixel terms
   dominate at iteration 0).
3. **Stage-1 hole-L1** (img_stg1 is full-res 256 and already composited —
   simpler than the synthesis specced).
4. **TTUR G 1e-3 / D 1e-4, R1 gamma 2 (single value, no sweep — judges killed
   the 4-run gamma sweep as GPU-weeks for an FID metric HMAT already wins)**.
5. **All-masked-window fallback** in the masked attention (an 8x8 window fully
   inside a peel hole currently degenerates to uniform averaging) — cheap,
   targeted at the 55%-peel regime where HMAT plausibly loses to LaMa.
6. Keep checkpoint selection on the deterministic in-run metrics; note in the
   paper that snapshot selection uses the test protocol (same practice as the
   MAT/LaMa baselines here; no extra val split exists).

### Follow-up run (separate, attributable ablation row)
7. **Sobel gradient hole loss** (mechanism proven on the same DHMural imagery by
   ADF's ablation — but cite as inspiration only: their 19.07→23.64 jump is a
   blind-restoration protocol, expect tenths of a dB) **+ focal-frequency loss**
   (judges demoted: the "+0.42dB C2LGM" citation could not be verified — treat
   FFL as a disposable ablation row, claim nothing in advance).

### Engineering (parallel, no GPU)
8. **MADF reparametrization as 17 basis convolutions** — exact algebraic
   identity (the filter generator is a linear 1x1 conv), removes the ~3.6 GB/
   sample filter tensors that force batch_gpu=1; expect batch_gpu 3-6 after.
   Forward-equality unit test (rtol 1e-4), then treat the first big-batch run
   as a new baseline (optimizer dynamics change — "metric-neutral" applies to
   the forward pass only).

### Paper v2 (gated)
9. **FFC residual blocks in the FirstStage decoder** (FcF recipe; torch.fft
   works on torch 1.7.1 — the synthesis's "needs torch>=1.8" was wrong; vendor
   ffc.py from the local LaMa copy, zero-init the residual). GATE on the
   crack-vs-peel metric split: LaMa's ablation shows FFC pays off on *wide*
   masks; commit only if HMAT's deficit concentrates on large peel holes.
10. **Continuous mask through all 5 Swin stages + soften the feature-level gate**
    (CMT-style; keep the image-level hard composite untouched — it is the
    paper's differentiator). Boundary-band SSIM is the success metric.
11. **Credible table**: self-train original MAT (same harness — isolates the
    FirstStage contribution, the paper's central claim); ONE mural baseline
    (PRN is already cloned + has an env) — not both PRN and CMAMRNet; MuralDH
    second benchmark only after the MADF reparam (it is 512px — memory wall).
12. **MAE-FAR-style priors** (the one technique with published PSNR+FID gains
    simultaneously) — last, after a 1-hour pickle-round-trip smoke test (timm
    ViT inside the persistence-pickled Generator is the risk).

### Evaluation upgrades (hours, locked)
- Hole-only PSNR/L1 (full-image metrics dilute the signal ~4x at 20-30% holes),
  boundary-band SSIM (8px ring), crack-vs-peel split (recoverable by replaying
  the seeded mask generator — no new masks needed), LPIPS + P-IDS/U-IDS
  (tooling already in repo), 3x repeat of the old protocol to publish noise
  floors. State explicitly that ADF's published DHMural numbers (PSNR 23.64)
  are blind restoration, NOT mask-given inpainting — never put them in the same
  table without the caveat.

## Do-not-do (all upheld by judges)
- No diffusion/flow backbone (breaks valid-pixel guarantee; data/compute scale).
- No projected/ImageNet-feature discriminators (deflates FID metrics — and FID
  is HMAT's headline win; Kynkäänniemi et al. 2022).
- No raising hole-L1 beyond 10 to chase PSNR (direct perception-distortion
  trade against the FID 15.3 advantage).
- Never soften the image-level hard composite; soften only the redundant
  feature-level gate.
- No hinge loss swap; tune gamma instead. No DINOv2/CLIP as the main perceptual
  loss (good metrics, unstable training targets per arXiv 2509.20878).
- No blind copying of LaMa's absolute loss weights (normalizations differ).
- No ADA color augmentations (miscalibrated for paired inpainting D; leaks into
  narrow mural palettes). No widening of training masks to 10-50% for THIS
  benchmark (test masks are fixed at 20-30%; widening is robustness insurance
  that costs in-distribution capacity).
- No wholesale rewrites (PSM iterative, U-Net D, WavePaint) before the loss
  recipe lands.

## Context numbers (fixed-mask benchmark, 2,649 images)
| Model | PSNR | SSIM | L1 | FID |
|---|---|---|---|---|
| LaMa-fourier (best) | 25.30 | 0.898 | 0.0193 | 17.77 |
| HMAT 000560 (pre-fix) | 24.24 | 0.892 | 0.0216 | 15.25 |
| Target after recipe fixes | ~25.0-25.3 | ≥0.898 | ≤0.020 | ≤15.5 |

## Execution log (2026-06-10 evening)

- Determinism sweep on 000560: FLAT — psi {1, 0.5, 0} all 24.20/0.892/0.0217/FID
  ~15.25; mul_map=1.0 slightly worse (24.15). Conclusion: the PSNR gap lives in
  the weights, not the inference protocol. Rows in results/test_metrics.csv.
- MADF reparametrized as 16+1 basis convolutions (exact: full-G diff 5e-7, 0 at
  uint8). Steady-state training now 199 s/kimg at batch_gpu=4 (was 747 at
  batch_gpu=1) — 3.7x faster; minibatch-std functional again (group 4).
- Run 00012 launched: resume 000560, batch 16 / batch_gpu 4, kimg 1000,
  losses = adv(2 heads) + hole-L1 10 + stage-1 hole-L1 10 + FM 10 (12 D taps)
  + HRF 10 (VGG off), TTUR glr 1e-3 / dlr 1e-4, R1 gamma 2, style_mix 0,
  all-masked-window fallback + fixed validity mask active.
  Iteration-0 magnitudes (weighted): pixel 3.5 / FM 3.1 / adv 1.8 / HRF ~1.2.
