"""Hole-only and boundary-band PSNR/SSIM/L1 for an inpainting output dir.

Full-image metrics are dominated by the ~70-80% valid region (bit-exact under
hard compositing), which dilutes component effects -- especially the
teacher-forcing gate, whose benefit is concentrated at hole boundaries. This
script measures the regions that actually differ:

  hole      : pixels inside the missing region (mask hole)
  boundary  : an 8px band straddling the hole boundary (seam detector)

PSNR/L1 are computed over the masked pixels directly; SSIM averages MAT's
Gaussian-windowed SSIM map over the masked pixels (standard masked-SSIM).

Usage: python hole_metrics.py <pred_dir> <gt_dir> <mask_dir> <model_name>
Mask convention: white(255) = hole. Appends to results/hole_metrics.csv.
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

PROJ = '/home/jincheng/Mural/mural_project'
CSV = os.path.join(PROJ, 'lama_mat_comparison/results/hole_metrics.csv')
BAND = 8


def ssim_map(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.transpose())
    mu1 = cv2.filter2D(img1, -1, w)
    mu2 = cv2.filter2D(img2, -1, w)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = cv2.filter2D(img1 ** 2, -1, w) - mu1_sq
    s2 = cv2.filter2D(img2 ** 2, -1, w) - mu2_sq
    s12 = cv2.filter2D(img1 * img2, -1, w) - mu1_mu2
    return ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))


def region_metrics(pred, gt, region):
    """region: bool HxW. Returns (psnr, ssim, l1) over region pixels."""
    if region.sum() == 0:
        return None
    diff2 = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)  # HxWx3
    mse = diff2[region].mean()
    psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    l1 = np.abs(pred.astype(np.float64) / 255 - gt.astype(np.float64) / 255)[region].mean()
    smap = np.stack([ssim_map(pred[..., c], gt[..., c]) for c in range(3)], axis=-1)
    ssim = smap[region].mean()
    return psnr, ssim, l1


def main():
    pred_dir, gt_dir, mask_dir, name = sys.argv[1:5]
    preds = sorted(glob.glob(os.path.join(pred_dir, '*.png')))
    hole_acc, band_acc = [], []
    for p in preds:
        fn = os.path.basename(p)
        stem = os.path.splitext(fn)[0]
        gt = cv2.imread(os.path.join(gt_dir, fn))
        pred = cv2.imread(p)
        m = cv2.imread(os.path.join(mask_dir, f'{stem}_mask001.png'), cv2.IMREAD_GRAYSCALE)
        hole = m > 127
        k = np.ones((2 * BAND + 1, 2 * BAND + 1), np.uint8)
        hu = hole.astype(np.uint8)
        band = (cv2.dilate(hu, k) > 0) & ~(cv2.erode(hu, k) > 0)
        hr = region_metrics(pred, gt, hole)
        br = region_metrics(pred, gt, band)
        if hr:
            hole_acc.append(hr)
        if br:
            band_acc.append(br)
    H = np.array(hole_acc)
    B = np.array(band_acc)
    hp, hs, hl = H.mean(0)
    bp, bs, bl = B.mean(0)
    print(f'{name}  n={len(H)}')
    print(f'  HOLE     psnr {hp:.4f}  ssim {hs:.4f}  l1 {hl:.4f}')
    print(f'  BOUNDARY psnr {bp:.4f}  ssim {bs:.4f}  l1 {bl:.4f}')
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    new = not os.path.exists(CSV)
    with open(CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['model', 'n', 'hole_psnr', 'hole_ssim', 'hole_l1',
                        'band_psnr', 'band_ssim', 'band_l1', 'pred_dir'])
        w.writerow([name, len(H), f'{hp:.4f}', f'{hs:.4f}', f'{hl:.4f}',
                    f'{bp:.4f}', f'{bs:.4f}', f'{bl:.4f}', pred_dir])
    print(f'appended to {CSV}')


if __name__ == '__main__':
    main()
