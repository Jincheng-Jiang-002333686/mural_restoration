"""Run an HMAT (MAT-codebase) snapshot on the fixed test masks, deterministically.

Usage (mural env, from anywhere):
    python hmat_predict_fixed.py <snapshot.pkl> <out_dir>

- Images: lama_mat_comparison/data/mural_lama/gt_256 (256x256 test_ref copies)
- Masks:  lama_mat_comparison/masks/test_mixed_hole02_03 (<stem>_mask001.png, 255=hole)
  converted to MAT convention (1=preserved, 0=hole).
- Deterministic: per-image seeded z and torch RNG (the pickled generator still
  contains the always-on dropout, so seeding the RNG is what makes it reproducible).
- The generator composites internally, so outputs preserve valid pixels exactly.
"""
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image

PROJ = '/home/jincheng/Mural/mural_project'
sys.path.insert(0, os.path.join(PROJ, 'MAT'))  # torch_utils/dnnlib for persistence unpickling
sys.path.insert(0, PROJ)  # some pickled refs use the 'MAT.' package prefix

GT_DIR = os.path.join(PROJ, 'lama_mat_comparison/data/mural_lama/gt_256')
MASK_DIR = os.path.join(PROJ, 'lama_mat_comparison/masks/test_mixed_hole02_03')
BATCH = 4
# optional overrides: argv[3]=gt_dir argv[4]=mask_dir
if len(sys.argv) > 3:
    GT_DIR = sys.argv[3]
if len(sys.argv) > 4:
    MASK_DIR = sys.argv[4]


def main():
    snapshot_pkl, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda')

    with open(snapshot_pkl, 'rb') as f:
        data = pickle.load(f)
    G = data['G_ema'].eval().requires_grad_(False).to(device)
    print(f'loaded {snapshot_pkl}: z_dim={G.z_dim} res={G.img_resolution}')

    names = sorted(f for f in os.listdir(GT_DIR) if f.endswith('.png'))
    assert len(names) > 0, GT_DIR

    for start in range(0, len(names), BATCH):
        chunk = names[start:start + BATCH]
        imgs, masks, zs = [], [], []
        for j, name in enumerate(chunk):
            stem = os.path.splitext(name)[0]
            img = np.asarray(Image.open(os.path.join(GT_DIR, name)).convert('RGB'), np.float32)
            imgs.append(img.transpose(2, 0, 1) / 127.5 - 1.0)
            m = np.asarray(Image.open(os.path.join(MASK_DIR, f'{stem}_mask001.png')).convert('L'), np.float32)
            masks.append((1.0 - m / 255.0)[None])  # 1=preserved
            zs.append(np.random.RandomState(start + j).randn(G.z_dim))
        img_t = torch.from_numpy(np.stack(imgs)).float().to(device)
        mask_t = torch.from_numpy(np.stack(masks)).float().to(device)
        z_t = torch.from_numpy(np.stack(zs)).float().to(device)
        c_t = torch.zeros([len(chunk), 0], device=device)

        torch.manual_seed(start)
        torch.cuda.manual_seed_all(start)
        with torch.no_grad():
            out = G(img_t, mask_t, z_t, c_t, noise_mode='const')
        out = ((out + 1.0) * 127.5).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
        for j, name in enumerate(chunk):
            Image.fromarray(out[j].transpose(1, 2, 0)).save(os.path.join(out_dir, name))
        if (start // BATCH) % 100 == 0:
            print(f'{start}/{len(names)}', flush=True)
    print('PREDICT_DONE')


if __name__ == '__main__':
    main()
