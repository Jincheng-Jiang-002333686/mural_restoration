"""Generate a training mask bank for EdgeConnect from the frozen MAT generator.

20,000 mixed crack+peel masks, hole range [0.2, 0.3], 256x256, white=hole
(EdgeConnect/LaMa convention). Seeded DISJOINTLY from the fixed test masks
(test masks use crc32(filename) ^ 42; here plain sequential seeds offset by
10_000_000 to avoid any overlap in practice).
"""
import os
import random
import sys

import cv2
import numpy as np

PROJ = '/home/jincheng/Mural/mural_project'
sys.path.insert(0, os.path.join(PROJ, 'lama_mat_comparison/external/lama/saicinpainting/training/data'))
from mat_mask_generator_256 import RandomMask

OUT = os.path.join(PROJ, 'lama_mat_comparison/data/ec_train_masks')
N = 20000

os.makedirs(OUT, exist_ok=True)
ratios = []
for i in range(N):
    seed = 10_000_000 + i
    np.random.seed(seed)
    random.seed(seed)
    m = RandomMask(256, hole_range=[0.2, 0.3], mask_type='mixed')[0]  # 1=preserved
    ratios.append(1.0 - float(m.mean()))
    cv2.imwrite(os.path.join(OUT, f'mask_{i:05d}.png'), ((1.0 - m) * 255).astype(np.uint8))
    if (i + 1) % 2000 == 0:
        print(f'{i + 1}/{N}', flush=True)

r = np.array(ratios)
print(f'hole ratio mean={r.mean():.4f} min={r.min():.4f} max={r.max():.4f}')
print('MASKBANK_DONE')
