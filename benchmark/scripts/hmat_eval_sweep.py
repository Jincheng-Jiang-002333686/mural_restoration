"""Perception-distortion sweep on an existing HMAT snapshot (no training).

Builds a FRESH Generator from the current networks/mat.py source (eval-mode
dropout gating + mul_map override available), copies the snapshot weights in,
and renders the 2,649 fixed-mask test images under several inference settings:

  evalmode_psi1      seeded per-image z, truncation_psi=1, mul_map eval-const 0.5
  evalmode_psi05     truncation_psi=0.5
  evalmode_psi0      truncation_psi=0 (w = w_avg, fully deterministic style)
  evalmode_psi0_mm1  psi=0 + mul_map=1.0 (encoder features only, no w-noise blend)

Usage (mural env): python hmat_eval_sweep.py <snapshot.pkl> <variant>
"""
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image

PROJ = '/home/jincheng/Mural/mural_project'
sys.path.insert(0, os.path.join(PROJ, 'MAT'))
sys.path.insert(0, PROJ)

GT_DIR = os.path.join(PROJ, 'lama_mat_comparison/data/mural_lama/gt_256')
MASK_DIR = os.path.join(PROJ, 'lama_mat_comparison/masks/test_mixed_hole02_03')
OUT_ROOT = os.path.join(PROJ, 'lama_mat_comparison/outputs')
BATCH = 4

VARIANTS = {
    'evalmode_psi1': dict(psi=1.0, mul_map=None),
    'evalmode_psi05': dict(psi=0.5, mul_map=None),
    'evalmode_psi0': dict(psi=0.0, mul_map=None),
    'evalmode_psi0_mm1': dict(psi=0.0, mul_map=1.0),
}


def main():
    snapshot_pkl, variant = sys.argv[1], sys.argv[2]
    cfg = VARIANTS[variant]
    out_dir = os.path.join(OUT_ROOT, f'hmat_sweep_{variant}')
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda')

    with open(snapshot_pkl, 'rb') as f:
        data = pickle.load(f)
    G_old = data['G_ema']

    # fresh Generator from CURRENT source so eval-mode dropout gating applies
    from torch_utils import misc
    from networks.mat import Generator
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=256, img_channels=3)
    G = G.eval().requires_grad_(False)
    misc.copy_params_and_buffers(G_old, G, require_all=False)
    G = G.to(device)
    if cfg['mul_map'] is not None:
        G.synthesis.mul_map_const = float(cfg['mul_map'])
    del G_old, data

    names = sorted(f for f in os.listdir(GT_DIR) if f.endswith('.png'))
    assert len(names) == 2649, len(names)

    for start in range(0, len(names), BATCH):
        chunk = names[start:start + BATCH]
        imgs, masks, zs = [], [], []
        for j, name in enumerate(chunk):
            stem = os.path.splitext(name)[0]
            img = np.asarray(Image.open(os.path.join(GT_DIR, name)).convert('RGB'), np.float32)
            imgs.append(img.transpose(2, 0, 1) / 127.5 - 1.0)
            m = np.asarray(Image.open(os.path.join(MASK_DIR, f'{stem}_mask001.png')).convert('L'), np.float32)
            masks.append((1.0 - m / 255.0)[None])
            zs.append(np.random.RandomState(start + j).randn(G.z_dim))
        img_t = torch.from_numpy(np.stack(imgs)).float().to(device)
        mask_t = torch.from_numpy(np.stack(masks)).float().to(device)
        z_t = torch.from_numpy(np.stack(zs)).float().to(device)
        c_t = torch.zeros([len(chunk), 0], device=device)

        with torch.no_grad():
            out = G(img_t, mask_t, z_t, c_t, truncation_psi=cfg['psi'], noise_mode='const')
        out = ((out + 1.0) * 127.5).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
        for j, name in enumerate(chunk):
            Image.fromarray(out[j].transpose(1, 2, 0)).save(os.path.join(out_dir, name))
        if (start // BATCH) % 200 == 0:
            print(f'{variant} {start}/{len(names)}', flush=True)
    print(f'{variant} PREDICT_DONE')


if __name__ == '__main__':
    main()
