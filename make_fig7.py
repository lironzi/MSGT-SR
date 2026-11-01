#!/usr/bin/env python
"""
make_fig7.py: Fig. 7 of the paper (loss-weight ablation) from the retrained A1 / A2 / A3 models.

This script runs the three retrained models on the SAME test crops (session 111022_3, the same
500-crop sequence and flips as eval_ablation.py) and writes one PNG per crop
and model, at the native HR size (168 x 128, 8-bit grey, like the old panels).

Usage (from the folder that contains runs/, Data/, Model/, misc/; one GPU or CPU, ~1 minute):
    python make_fig7.py
    python make_fig7.py --crops 47,56,61 --runs runs/A1_full_w111_* runs/A2_full_w039_18_* runs/A3_full_w003_*

Outputs (default folder fig7_out/):
    HR_<i>.png, A1_<i>.png, A2_<i>.png, A3_<i>.png   panels (same grey scale within each crop)
    fig7_preview.png                                  all panels in one image, for a quick look
    fig7_crops.csv                                    PSNR / MAE of every panel (raw counts)
Copy fig7_out/ to the Overleaf project as figs/fig7/.
"""
import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from Dataset import ThermalVisDataset            # noqa: E402
from misc.trainer import _build_net              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=None,
                help="run folders in the order A1 A2 A3 (default: newest runs/A1_full_w111_*, A2_full_w039_18_*, A3_full_w003_*)")
ap.add_argument("--labels", default="A1,A2,A3")
ap.add_argument("--crops", default="47,56,61", help="test-crop indices (the old Fig. 7 used 47, 56, 61)")
ap.add_argument("--data", default="Data")
ap.add_argument("--session", default="111022_3")
ap.add_argument("--n_crops", type=int, default=500, help="must equal the evaluation setting so crops/flips match")
ap.add_argument("--range", choices=["hr", "each"], default="hr",
                help="grey scale per crop: 'hr' = min/max of the HR target for all panels of that crop "
                     "(comparable panels, default); 'each' = every panel stretched on its own")
ap.add_argument("--out", default="fig7_out")
args = ap.parse_args()

labels = args.labels.split(",")
if args.runs is None:
    args.runs = []
    for pat in ["A1_full_w111_*", "A2_full_w039_18_*", "A3_full_w003_*"]:
        c = sorted(glob.glob(str(ROOT / "runs" / pat)), key=os.path.getmtime)
        if not c:
            raise SystemExit(f"no run folder runs/{pat}; pass --runs explicitly")
        args.runs.append(c[-1])
if len(args.runs) != len(labels):
    raise SystemExit("--runs and --labels must have the same length")
crops = [int(c) for c in args.crops.split(",")]
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(folder):
    folder = Path(folder)
    cfg = yaml.safe_load(open(folder / "config.yaml"))
    net = _build_net(cfg)
    ck = torch.load(folder / "best.pth", map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
    return net.to(DEV).eval(), cfg, ck.get("epoch")


nets = []
for lab, f in zip(labels, args.runs):
    net, cfg, ep = load(f)
    nets.append((lab, net))
    print(f"{lab}: {f}  (weights {cfg.get('weights')}, best epoch {None if ep is None else ep + 1})")

# crops: built sequentially, exactly as in the evaluation scripts (same positions AND flips)
ds = ThermalVisDataset(dataPath=args.data, scale_factor=8, HR_downsample=2, original_size=(256, 336),
                       LR_upsample=0, Sessions=[args.session], numPermutationPerImage=args.n_crops)
m, s = ds.stats[args.session][-1]
mosaic = ds.ThermalVisImgs[args.session][-1]
peak = float(mosaic.max()) - float(mosaic.min())
samples = {}
for i in range(max(crops) + 1):
    item = ds[i]
    if i in crops:
        samples[i] = item


def to_png(x, lo, hi, path):
    from PIL import Image
    a = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path)


rows, panels = [], []
with torch.no_grad():
    for i in crops:
        LR, HR, vis = samples[i]
        hr = (HR.float() * s + m)[0].numpy()
        outs = {}
        for lab, net in nets:
            o = net(LR.unsqueeze(0).float().to(DEV), vis.unsqueeze(0).float().to(DEV))[-1]
            outs[lab] = (o.cpu() * s + m)[0, 0].numpy()
        lo, hi = float(hr.min()), float(hr.max())
        for name, img in [("HR", hr)] + list(outs.items()):
            l_, h_ = (lo, hi) if args.range == "hr" else (float(img.min()), float(img.max()))
            to_png(img, l_, h_, OUT / f"{name}_{i}.png")
            if name != "HR":
                mse = float(np.mean((img - hr) ** 2))
                rows.append(dict(crop=i, model=name, psnr=10 * math.log10(peak ** 2 / max(mse, 1e-12)),
                                 mae=float(np.mean(np.abs(img - hr)))))
        panels.append([hr] + [outs[l] for l in labels])
        print(f"crop {i}: " + ", ".join(f"{r['model']} {r['psnr']:.2f} dB" for r in rows if r["crop"] == i))

with open(OUT / "fig7_crops.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["crop", "model", "psnr", "mae"])
    w.writeheader()
    w.writerows(rows)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(crops), 1 + len(labels), figsize=(2.2 * (1 + len(labels)), 1.8 * len(crops)))
    ax = np.atleast_2d(ax)
    for r, (i, row) in enumerate(zip(crops, panels)):
        lo, hi = float(row[0].min()), float(row[0].max())
        for c, img in enumerate(row):
            ax[r, c].imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title((["HR"] + labels)[c], fontsize=9)
        ax[r, 0].set_ylabel(f"crop {i}", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_preview.png", dpi=150)
except Exception as e:
    print(f"(preview skipped: {e})")
print(f"\nPanels in {OUT}/  ->  copy this folder to Overleaf as figs/fig7/")
