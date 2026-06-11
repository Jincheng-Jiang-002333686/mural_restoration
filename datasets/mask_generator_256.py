import numpy as np
from PIL import Image, ImageDraw
import math
import random


def _draw_jagged_line(draw, points, fill=1, width=1, skip_prob=0.06):
    if len(points) < 2:
        return
    for p0, p1 in zip(points[:-1], points[1:]):
        if np.random.random() > skip_prob:
            draw.line([p0, p1], fill=fill, width=width)


def _random_walk_points(s, start, angle, steps, step_len, angle_jitter):
    x, y = start
    points = [(int(x), int(y))]
    for _ in range(steps):
        angle += np.random.normal(0, angle_jitter)
        x += step_len * np.random.uniform(0.6, 1.35) * math.cos(angle)
        y += step_len * np.random.uniform(0.6, 1.35) * math.sin(angle)
        x = np.clip(x, 0, s - 1)
        y = np.clip(y, 0, s - 1)
        points.append((int(x), int(y)))
        if x <= 0 or x >= s - 1 or y <= 0 or y >= s - 1:
            break
    return points


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


def RandomCrack(s, min_cracks=4, max_cracks=12, min_width=1, max_width=3):
    mask = Image.new('L', (s, s), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(np.random.randint(min_cracks, max_cracks + 1)):
        side = np.random.randint(4)
        if side == 0:
            start = (np.random.randint(0, s), 0)
            angle = np.random.uniform(0.15 * math.pi, 0.85 * math.pi)
        elif side == 1:
            start = (s - 1, np.random.randint(0, s))
            angle = np.random.uniform(0.65 * math.pi, 1.35 * math.pi)
        elif side == 2:
            start = (np.random.randint(0, s), s - 1)
            angle = np.random.uniform(1.15 * math.pi, 1.85 * math.pi)
        else:
            start = (0, np.random.randint(0, s))
            angle = np.random.uniform(-0.35 * math.pi, 0.35 * math.pi)

        steps = np.random.randint(max(14, s // 14), max(24, s // 6))
        step_len = np.random.uniform(s / 95, s / 45)
        width = np.random.randint(min_width, max_width + 1)
        points = _random_walk_points(s, start, angle, steps, step_len, angle_jitter=0.38)
        _draw_jagged_line(draw, points, width=width, skip_prob=0.03)

        for _ in range(np.random.randint(1, 4)):
            if len(points) < 5 or np.random.random() > 0.7:
                continue
            branch_start = points[np.random.randint(1, len(points) - 1)]
            branch_angle = angle + np.random.choice([-1, 1]) * np.random.uniform(0.25 * math.pi, 0.65 * math.pi)
            branch_steps = np.random.randint(5, max(7, steps // 2))
            branch = _random_walk_points(s, branch_start, branch_angle, branch_steps, step_len, angle_jitter=0.5)
            _draw_jagged_line(draw, branch, width=max(1, width - 1), skip_prob=0.12)

    return np.asarray(mask, np.uint8)


def RandomPeeling(s):
    # Use the original free-form brush morphology as paint-loss/peeling damage.
    return RandomBrush(np.random.randint(4, 8), s)


def _random_component(s, mask_type):
    if mask_type == 'brush':
        return RandomBrush(np.random.randint(2, 6), s)
    if mask_type == 'crack':
        return RandomCrack(s)
    if mask_type == 'peel':
        return RandomPeeling(s)
    if mask_type == 'mixed':
        mode = random.choices(['crack', 'peel'], weights=[0.45, 0.55])[0]
        return _random_component(s, mode)
    raise ValueError(f'Unsupported mask_type: {mask_type}')


def _sample_mask_type(mask_type):
    if isinstance(mask_type, (list, tuple)):
        return random.choice(mask_type)
    return mask_type


def RandomMask(s, hole_range=[0.4, 0.5], mask_type='mixed', max_attempts=200):
    """
    Generate a random mural degradation mask.

    The returned mask keeps the original convention: 1 is preserved area and
    0 is damaged/missing area. The hole ratio is still constrained by
    hole_range, so the 20-30% and 40-50% experiment settings remain comparable.
    """
    best_mask = None
    best_error = float('inf')

    for _ in range(max_attempts):
        hole = np.zeros((s, s), np.uint8)
        curr_type = _sample_mask_type(mask_type)
        max_components = 20 if curr_type == 'crack' else 12

        if curr_type == 'mixed':
            hole = np.logical_or(hole, RandomCrack(s)).astype(np.uint8)
            hole = np.logical_or(hole, RandomPeeling(s)).astype(np.uint8)

        for _ in range(max_components):
            hole = np.logical_or(hole, _random_component(s, curr_type)).astype(np.uint8)
            hole_ratio = np.mean(hole)

            if hole_range is None:
                mask = (1 - hole).astype(np.uint8)
                return mask[np.newaxis, ...].astype(np.float32)

            target = 0.5 * (hole_range[0] + hole_range[1])
            error = abs(hole_ratio - target)
            if error < best_error:
                best_error = error
                best_mask = (1 - hole).astype(np.uint8)
            if hole_range[0] < hole_ratio < hole_range[1]:
                mask = (1 - hole).astype(np.uint8)
                return mask[np.newaxis, ...].astype(np.float32)
            if hole_ratio > hole_range[1]:
                break

    return best_mask[np.newaxis, ...].astype(np.float32)


def BatchRandomMask(batch_size, s, hole_range=[0.4, 0.5], mask_type='mixed'):
    return np.stack([RandomMask(s, hole_range=hole_range, mask_type=mask_type) for _ in range(batch_size)], axis=0)


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
