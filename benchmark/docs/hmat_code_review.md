# HMAT Code & Paper Review — 2026-06-10

Multi-agent review (6 dimensions, 47 findings, every one adversarially re-verified
against the code; 0 refuted). Scope: `MAT/` (HMAT implementation), losses, data,
metrics, ablation variants, baseline harnesses, paper claims.

## Headline

**The paper's conclusion "HMAT loses to MADF on DHMural" is not supported by the
experiments as run.** The reported HMAT row comes from an undertrained checkpoint
evaluated on a different mask family than the MADF baseline, with a stochastic
metric harness over only 110 of 2,649 images. Best existing snapshot already
~matches MADF. Two genuine model bugs and one missing loss term are the main
code-level levers.

## Critical bugs (model)

1. **Transformer validity mask is inverted in the trained model** —
   [mat.py:913] `mask_binary = (mask.mean(dim=1) > 0)` thresholds *learned MADF
   mask features* (ReLU'd conv outputs), not the binary hole mask. Empirically
   verified on checkpoint 000560: hole tokens are 100% "valid", known tokens
   ~4-5% "valid" — the masked attention (Eq. 5 of the paper) actively suppresses
   known context instead of invalid tokens. At init it is an all-ones no-op.
   Fix: `mask_binary = F.interpolate(masks_in, size=x_size, mode='nearest')`.
   Also: the -inf in the paper is -100 in code, and masking is applied in only
   2 of 5 transformer stages (mat.py:917-925).

2. **The L1 reconstruction loss is never trained** — loss.py:85-92 computes it
   under `# just for showing` and only logs it; `loss_Gmain_all = loss_Gmain +
   loss_Gmain_stg1 + pcp_loss` (line 91). The paper claims `L = L_adv + 10*R1 +
   1*L_perc + 1*L_L1`. MADF wins PSNR because its loss is pixel-dominated
   (hole_weight 6.0 / valid 1.0 in MADF.ipynb). Fix: add hole-weighted L1 to
   loss_Gmain_all (output is hard-composited at mat.py:1005, so full-image L1
   already acts only on holes).

3. **Evaluation is stochastic** — metrics run with `noise_mode='random'`,
   unseeded z per image, an always-on `F.dropout(..., training=True)` in stage 2
   (mat.py:991-992), and a fresh random mask per image per snapshot
   (dataset_256.py:272). Snapshot PSNR swings ±0.8 dB from this alone.
   Fix: `noise_mode='const'` + fixed z (G_kwargs in training_loop.py:473-475),
   gate the dropout on `self.training`, and evaluate on a fixed mask set.

## Critical experiment-setup problems

4. **Run aborted at 572/3000 kimg (19% of budget), metrics still rising.**
   Snapshot 000500: PSNR 23.94 / SSIM 0.8982 / L1 0.0220 — vs paper's HMAT row
   23.11/0.868/0.025 (which exactly matches old snapshot 000220 / run 00006) and
   vs MADF's claimed 24.03/0.887/0.022. Reporting the best snapshot alone nearly
   closes the gap; resuming training likely passes it.

5. **In-run metrics used 110 images, not 2,649** — proven from the run-era
   `.pyc` bytecode (`max_real=110, num_gen=110`); the source was edited to 2649
   after the run ended. FID-110 (~139) is small-sample-dominated and not
   comparable to anything; all PSNR comparisons are noise-dominated.

6. **Baselines are not apples-to-apples.** MADF was trained/evaluated in its own
   notebook on different data (train_mural_png 3,500 patches, not train_ref
   10,584), brush-only masks, pad+crop val, pixel-dominated loss. HMAT's current
   masks are crack+peel 'mixed' (generator written 2026-06-02, after the paper
   runs). The paper's mask description ("Random Brush Stroke") matches only the
   backup generator. The MADF triple 24.03/0.887/0.022 is not reproducible from
   any logged run.

7. **The Severe 40-50% setting has no code path** — dataset_256.py:168 hardcodes
   `self._hole_range = [0.2, 0.3]` and ignores its constructor argument.

8. **The paper's "off-the-shelf Refinement Network" does not exist** in the repo
   or in any metric path; reported numbers are refinement-free.

9. **Training recipe off-spec**: places256 recipe (8 GPUs, batch 64, lr 2e-3,
   gamma 10, ema 10) used verbatim at 1 GPU / batch 2 — no rescaling, no grad
   accumulation; documented loss explosion in the first ~16 kimg; mbstd group
   silently 2 instead of 8; dead path-length phase rescales G lr to 0.0016.
   Paper says batch 4 on 2 GPUs — neither matches the run.

10. **FID collapse unexplained but confounded**: HMAT FID ~139 vs original-MAT
    backup run ~20 — but both numbers come from different sample counts /
    mask eras; recompute on the fixed 2,649-mask benchmark before concluding.

## Paper-consistency corrections (for the text, regardless of code changes)

- Transformer: 6 heads (not 8; 8 is impossible at dim 180), 14 Swin blocks in
  5 stages with depths [2,3,4,3,2] (not "5 blocks"); masked attention uses -100
  bias in 2/5 stages.
- Encoder is not uniformly MADF: 256-res stem and the 180-d expansion are plain
  convs; channel schedule is effectively [64, 64→128, 180].
- Loss equation: two GAN heads (stage-1 + final) + lazy R1 (gamma/2 every 16
  steps); no L1 (see bug 2); effective G lr 0.0016.
- Ablation confound: nonTF / nonMADF / nonMGStyle each re-introduce MAT's
  to_square/mul_map noise-injection that full HMAT does not have, so they ablate
  more than one thing. Equal_Capacity / Heavy_Semantic_Bias style-dim ablations
  are clean (verified).
- Verified-correct claims: teacher-forcing hard gate (mat.py:771-786), output
  compositing Eq. 1 (mat.py:946, 1005), style-fusion dims 360/180/64.

## Ranked action plan (expected gain / cost)

1. **Report best checkpoint + resume training to ≥1500 kimg** (free / GPU-days).
   000500 already gives 23.94/0.898/0.022. ~428 s/kimg ⇒ ~5 days per +1000 kimg.
2. **Add hole-weighted L1 to the objective** (2 lines; +0.3-0.8 dB expected).
3. **Fix the validity-mask inversion** (1 line + retrain; structure & FID).
4. **Deterministic eval** (2 lines; removes ±0.8 dB jitter, reproducible table).
5. **Effective batch 4-8 via grad accumulation + lr retune** (stabilizes GAN).
6. **Style_mixing_prob 0.9 → 0** for inpainting fidelity (1 line, low risk).
7. **One shared benchmark for all methods**: fixed seeded masks per test image
   (already built: `lama_mat_comparison/data/mural_lama/val` + `masks/`),
   single metric script (`lama_mat_comparison/scripts/eval_outputs.py` = MAT's
   exact metric code), full 2,649 images, deterministic inference. Re-run MADF,
   MAT, HMAT (and LaMa, training now) on it; pick one mask story (mixed
   crack+peel or brush) and update paper Sec 4.1 accordingly.
8. Implement or drop the refinement-network claim; fix paper numbers (batch/GPU,
   heads/blocks, loss equation, mask algorithm).

Full machine-readable findings: see workflow output
(`/tmp/.../tasks/wao78yey1.output`, 47 confirmed findings with file:line
evidence and verifier notes).
