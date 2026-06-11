"""Prepare the LaMa dataset layout from train_ref / test_ref.

- Resizes all images to 256x256 with cv2.INTER_LINEAR (same as MAT's dataset_256.py).
- Generates ONE fixed mask per test image with the frozen MAT mask generator
  (mask_type='mixed', hole_range=[0.2, 0.3]), seeded per filename for reproducibility.
- Saves masks in LaMa convention: white (255) = damaged/missing, black (0) = preserved.
  (MAT convention is inverted: 1 = preserved.)

Layout produced under lama_mat_comparison/data/mural_lama/:
  train/         10584 images from train_ref
  val/           2649 images from test_ref + <name>_mask001.png fixed masks
  visual_test/   first 15 val images + masks (qualitative monitoring)
Masks are also copied to lama_mat_comparison/masks/test_mixed_hole02_03/.
"""
import os
import random
import shutil
import sys
import zlib

import cv2
import numpy as np

PROJ = '/home/jincheng/Mural/mural_project'
COMP = os.path.join(PROJ, 'lama_mat_comparison')
sys.path.insert(0, os.path.join(COMP, 'external/lama/saicinpainting/training/data'))
from mat_mask_generator_256 import RandomMask  # frozen copy of MAT/datasets/mask_generator_256.py

SRC_TRAIN = os.path.join(PROJ, 'train_ref')
SRC_TEST = os.path.join(PROJ, 'test_ref')
ROOT = os.path.join(COMP, 'data/mural_lama')
MASK_BACKUP = os.path.join(COMP, 'masks/test_mixed_hole02_03')
GLOBAL_SEED = 42
HOLE_RANGE = [0.2, 0.3]
MASK_TYPE = 'mixed'
N_VISUAL = 15

def load_256(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'cannot read {path}')
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256))
    return img

def main():
    for sub in ['train', 'val', 'visual_test']:
        os.makedirs(os.path.join(ROOT, sub), exist_ok=True)
    os.makedirs(MASK_BACKUP, exist_ok=True)

    train_files = sorted(os.listdir(SRC_TRAIN))
    for i, f in enumerate(train_files):
        cv2.imwrite(os.path.join(ROOT, 'train', f), load_256(os.path.join(SRC_TRAIN, f)))
        if (i + 1) % 2000 == 0:
            print(f'train {i + 1}/{len(train_files)}', flush=True)

    test_files = sorted(os.listdir(SRC_TEST))
    hole_ratios = []
    for i, f in enumerate(test_files):
        stem = os.path.splitext(f)[0]
        cv2.imwrite(os.path.join(ROOT, 'val', f), load_256(os.path.join(SRC_TEST, f)))

        seed = (zlib.crc32(f.encode('utf-8')) ^ GLOBAL_SEED) & 0xffffffff
        np.random.seed(seed)
        random.seed(seed)
        mat_mask = RandomMask(256, hole_range=HOLE_RANGE, mask_type=MASK_TYPE)[0]  # 1=preserved
        hole_ratios.append(1.0 - float(mat_mask.mean()))
        lama_mask = ((1.0 - mat_mask) * 255).astype(np.uint8)  # 255=hole
        mask_name = f'{stem}_mask001.png'
        cv2.imwrite(os.path.join(ROOT, 'val', mask_name), lama_mask)
        shutil.copy(os.path.join(ROOT, 'val', mask_name), os.path.join(MASK_BACKUP, mask_name))
        if (i + 1) % 500 == 0:
            print(f'val {i + 1}/{len(test_files)}', flush=True)

    for f in test_files[:N_VISUAL]:
        stem = os.path.splitext(f)[0]
        shutil.copy(os.path.join(ROOT, 'val', f), os.path.join(ROOT, 'visual_test', f))
        shutil.copy(os.path.join(ROOT, 'val', f'{stem}_mask001.png'),
                    os.path.join(ROOT, 'visual_test', f'{stem}_mask001.png'))

    hr = np.array(hole_ratios)
    print(f'masks: n={len(hr)} hole_ratio mean={hr.mean():.4f} min={hr.min():.4f} max={hr.max():.4f}')
    in_range = np.mean((hr > HOLE_RANGE[0]) & (hr < HOLE_RANGE[1]))
    print(f'fraction strictly inside {HOLE_RANGE}: {in_range:.4f}')
    print('DATASET_DONE')

if __name__ == '__main__':
    main()
