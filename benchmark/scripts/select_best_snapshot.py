"""Print the best HMAT snapshot(s) by in-run metrics for a training run dir.

Emits lines "<tag> <snapshot_path>". Picks best-PSNR and best-FID snapshots;
if they coincide, emits a single 'final' row. Also always includes the last
snapshot if distinct (paper convention often reports the final checkpoint).
"""
import json
import os
import sys

run = sys.argv[1]


def load(metric):
    rows = []
    path = os.path.join(run, f'metric-{metric}.jsonl')
    for line in open(path):
        r = json.loads(line)
        snap = r['snapshot_pkl']
        if snap.endswith('000000.pkl'):
            continue  # skip the tick-0 resume baseline
        rows.append((snap, r['results']))
    return rows


psnr_rows = load('psnr2649_full')
fid_rows = load('fid2649_full')

best_psnr = max(psnr_rows, key=lambda x: x[1]['psnr'])[0]
best_fid = min(fid_rows, key=lambda x: x[1]['fid2649_full'])[0]
last = psnr_rows[-1][0]

picks = {}
picks[best_psnr] = 'bestpsnr'
picks.setdefault(best_fid, 'bestfid')
picks.setdefault(last, 'final')

for snap, tag in picks.items():
    print(f'{tag} {os.path.join(run, os.path.basename(snap))}')
