"""Final test-set evaluation with MAT's own metric implementations.

Usage (run in the `mural` conda env, which has MAT's deps incl. pyspng):
    python eval_outputs.py <pred_dir> <gt_dir> <model_name>

Computes PSNR / SSIM / L1 (MAT/evaluatoin/cal_psnr_ssim_l1.py) and
FID / P-IDS / U-IDS (MAT/evaluatoin/cal_fid_pids_uids.py, NVIDIA
inception-2015-12-05 detector) on full images, appends a row to
lama_mat_comparison/results/test_metrics.csv and logs the scores to the
wandb project `mural_inpainting` as a summary-only run.
"""
import csv
import os
import sys

PROJ = '/home/jincheng/Mural/mural_project'
sys.path.insert(0, os.path.join(PROJ, 'MAT', 'evaluatoin'))
sys.path.insert(0, PROJ)

import cal_psnr_ssim_l1
import cal_fid_pids_uids

RESULTS_CSV = os.path.join(PROJ, 'lama_mat_comparison', 'results', 'test_metrics.csv')


def main():
    pred_dir, gt_dir, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
    import glob as _glob
    n_images = len(_glob.glob(os.path.join(pred_dir, '*.png')))

    psnr, ssim, l1 = cal_psnr_ssim_l1.calculate_metrics(pred_dir, gt_dir)
    fid, pids, uids = cal_fid_pids_uids.calculate_metrics(pred_dir, gt_dir)

    print(f'\n=== {model_name} on {pred_dir} ===')
    print(f'psnr: {psnr:.4f}, ssim: {ssim:.4f}, l1: {l1:.4f}, '
          f'fid: {fid:.4f}, pids: {pids:.4f}, uids: {uids:.4f}')

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['model', 'mask_type', 'hole_range', 'n_images',
                        'psnr', 'ssim', 'l1', 'fid', 'pids', 'uids', 'pred_dir'])
        w.writerow([model_name, 'mixed', '[0.2,0.3]', n_images,
                    f'{psnr:.4f}', f'{ssim:.4f}', f'{l1:.4f}',
                    f'{fid:.4f}', f'{pids:.4f}', f'{uids:.4f}', pred_dir])
    print(f'appended to {RESULTS_CSV}')

    mask_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        PROJ, 'lama_mat_comparison', 'masks', 'test_mixed_hole02_03')

    try:
        import glob

        import numpy as np
        from PIL import Image

        import wandb
        run = wandb.init(project='mural_inpainting',
                         name=f'{model_name}-test-eval',
                         job_type='final_eval',
                         config=dict(model=model_name, mask_type='mixed',
                                     hole_range=[0.2, 0.3], n_images=n_images,
                                     pred_dir=pred_dir, gt_dir=gt_dir))
        wandb.log({'Test/psnr': psnr, 'Test/ssim': ssim, 'Test/l1': l1,
                   'Test/fid': fid, 'Test/pids': pids, 'Test/uids': uids})

        # sample panels: Original | Damaged | Restored (like the training grids)
        panels = []
        for p in sorted(glob.glob(os.path.join(pred_dir, '*.png')))[:12]:
            name = os.path.basename(p)
            stem = os.path.splitext(name)[0]
            gt = np.asarray(Image.open(os.path.join(gt_dir, name)).convert('RGB'), np.uint8)
            pred = np.asarray(Image.open(p).convert('RGB'), np.uint8)
            mask_path = os.path.join(mask_dir, f'{stem}_mask001.png')
            if os.path.exists(mask_path):
                hole = np.asarray(Image.open(mask_path).convert('L'), np.float32)[..., None] / 255.0
                damaged = (gt * (1.0 - hole)).astype(np.uint8)
            else:
                damaged = np.zeros_like(gt)
            panels.append(wandb.Image(np.concatenate([gt, damaged, pred], axis=1),
                                      caption=f'{stem}: Original | Damaged | Restored'))
        if panels:
            wandb.log({'test_samples': panels})
        run.finish()
        print('logged to wandb project mural_inpainting (metrics + sample panels)')
    except Exception as ex:  # metrics are already saved; wandb is best-effort here
        print(f'wandb logging failed: {ex}')


if __name__ == '__main__':
    main()
