import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    """
    Generate a single free-form stroke mask.
    """
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)  # random number of vertices
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        # Random angle in radians
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = (np.random.randint(10, maxBrushWidth + 1) // 2) * 2  # Ensure even number
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = int(np.maximum(np.minimum(nextY, h - 1), 0))
        nextX = int(np.maximum(np.minimum(nextX, w - 1), 0))
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_stroke_mask(im_size,
                         min_ratio=0.2,
                         max_ratio=0.3,
                         max_parts=7,
                         maxVertex=25,
                         maxLength=100,
                         maxBrushWidth=24,
                         maxAngle=360):
    """Generate free-form stroke mask"""
    while True:
        mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
        parts = random.randint(1, max_parts)
        for _ in range(parts):
            mask += np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        mask = np.minimum(mask, 1.0)

        coverage = np.mean(mask)  # fraction of pixels that are 1
        if coverage >= min_ratio and coverage <= max_ratio:
            return mask


class InpaintingDataset(Dataset):
    def __init__(
            self,
            image_dir,
            target_size=256,
            min_mask_ratio=0.2,
            max_mask_ratio=0.3,
            masks_per_image=3  # Number of different masks to generate per image
    ):
        """
        Dataset for VAR inpainting with multiple masks per image

        Args:
            image_dir: Directory with images
            target_size: Size to resize images to
            min_mask_ratio: Minimum ratio of pixels to mask
            max_mask_ratio: Maximum ratio of pixels to mask
            masks_per_image: Number of different random masks to generate for each image
        """
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, '*.jpg')) +
            glob.glob(os.path.join(image_dir, '*.png'))
        )
        self.target_size = target_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.masks_per_image = masks_per_image

        # Image normalization transform
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        print(f"Dataset loaded with {len(self.image_paths)} images × {masks_per_image} masks = {len(self)} total samples")

    def center_crop_and_resize(self, image):
        """Center crop the image to a square and resize to target_size"""
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        right = left + side
        bottom = top + side
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        return image

    def generate_random_mask(self):
        """
        Generate a random mask where:
        - 1 = hole/masked region
        - 0 = valid/known region
        """
        # Generate stroke mask where strokes=1 (holes), background=0
        mask_array = generate_stroke_mask(
            (self.target_size, self.target_size),
            min_ratio=self.min_mask_ratio,
            max_ratio=self.max_mask_ratio
        )

        # Return the single-channel mask
        return mask_array[:, :, 0]

    def __len__(self):
        return len(self.image_paths) * self.masks_per_image

    def __getitem__(self, index):
        # Calculate the original image index and mask variation
        image_idx = index // self.masks_per_image
        # Seed for reproducibility, but different for each mask variation
        mask_seed = index % self.masks_per_image

        # Set random seed based on index and mask variation for reproducibility
        random.seed(image_idx * 100 + mask_seed)
        np.random.seed(image_idx * 100 + mask_seed)

        # Load and transform the ground truth image
        image_path = self.image_paths[image_idx]
        gt_image = Image.open(image_path).convert("RGB")
        gt_image = self.center_crop_and_resize(gt_image)

        # Convert to tensor and normalize to [-1, 1]
        gt_tensor = transforms.ToTensor()(gt_image)
        gt_tensor = self.normalize(gt_tensor)

        # Generate mask (1=holes to be filled, 0=known regions)
        mask_array = self.generate_random_mask()
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()  # (1, H, W)

        # Create masked input image (apply mask to ground truth)
        # Masked regions will be set to value 0 after normalization (which is -1 in [-1,1] range)
        masked_tensor = gt_tensor * (1 - mask_tensor) - 1.0 * mask_tensor

        # Reset random seed to avoid affecting other parts of training
        random.seed(None)
        np.random.seed(None)

        return {
            "masked_image": masked_tensor,  # (3, H, W) in [-1, 1]
            "mask": mask_tensor,            # (1, H, W) with 1=holes, 0=valid
            "original_image": gt_tensor     # (3, H, W) in [-1, 1]
        }