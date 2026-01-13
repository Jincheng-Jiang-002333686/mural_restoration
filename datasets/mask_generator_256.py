import numpy as np
from PIL import Image, ImageDraw
import math
import random


def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(s, hole_range=[0.4, 0.5]):
    """
    Generate random mask with only brush strokes.
    Target hole ratio between 0.2 and 0.3.
    """
    while True:
        mask = np.ones((s, s), np.uint8)

        # Start with fewer brush strokes and add more if needed
        # Estimate: start with 4-8 strokes for 20-30% coverage
        num_strokes = np.random.randint(4, 8)

        # Generate brush stroke mask
        brush_mask = RandomBrush(num_strokes, s)

        # Apply brush strokes as holes (invert the brush mask)
        mask = np.logical_and(mask, 1 - brush_mask).astype(np.uint8)

        # Calculate hole ratio
        hole_ratio = 1 - np.mean(mask)

        # Check if within desired range
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def BatchRandomMask(batch_size, s, hole_range=[0.4, 0.5]):
    return np.stack([RandomMask(s, hole_range=hole_range) for _ in range(batch_size)], axis=0)


if __name__ == '__main__':
    # res = 512
    res = 256
    cnt = 2000
    tot = 0
    ratios = []

    for i in range(cnt):
        mask = RandomMask(s=res, hole_range=[0.2, 0.3])
        ratio = mask.mean()
        tot += ratio
        ratios.append(ratio)

    print(f"Average preserved ratio: {tot / cnt:.3f}")
    print(f"Average hole ratio: {1 - tot / cnt:.3f}")
    print(f"Min hole ratio: {1 - max(ratios):.3f}")
    print(f"Max hole ratio: {1 - min(ratios):.3f}")
