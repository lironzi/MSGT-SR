#!/usr/bin/env python
"""
eval_ablation.py: evaluate MSGT-SR and its ablation runs.

Run from the MSGT-SR folder (ModalityFusion). It finds the run folders written by misc/trainer.py
(runs/<run_name>_<date-time>/ with config.yaml + best.pth), rebuilds each model from its own
config.yaml (model name, stn_enable), and scores every run on

  * the VALIDATION session (301121_1): model selection is made here
  * the TEST session (111022_3, 500 fixed crops)

Outputs (in --out):
  ablation_table.csv/.tex   val + test metrics, params, latency for every run (+ the paper checkpoint)
  paired_vs_ref.csv         per-crop paired test of every run against the reference run (test set)
  stn_sweep.csv/.tex        visible-guide shift sweep for the STN runs (A1, C1, paper model)
  fig_val_curves.png        validation L1 per epoch for every run (from val_log.csv)
  fig_stn_sweep.png         PSNR vs shift; STN-estimated shift
  per_crop_metrics.csv, summary.md, run_info.json

Usage:
  python eval_ablation.py                                   # all runs/A*_, B*_, C*_ folders + paper model
  python eval_ablation.py --paper_ckpt runs/paper/epoch_32.pth
  python eval_ablation.py --runs runs/A1_full_w111_14_10_2026-09_00 runs/A3_full_w003_14_10_2026-09_00
  python eval_ablation.py --quick                           # 50 crops, 3 shifts: smoke test
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn
import yaml
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Dataset import ThermalVisDataset                                         # noqa: E402
from DISTS import DISTS                                                       # noqa: E402
from misc.trainer import _build_net                                           # noqa: E402
from misc.eval_extras import (psnr_from_mse, bicubic_x8, CropLocator, shift_visible, STNProbe,  # noqa: E402
                              modality_disagreement, measure_latency, bootstrap_ci, write_csv)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=None, help="run folders (default: runs/[ABC]*_*)")
ap.add_argument("--paper_ckpt", default="runs/paper/epoch_32.pth",
                help="the submitted model (MSGNet_w_F_Full); '' to skip")
ap.add_argument("--ref", default="A1", help="run-name prefix used as reference in the paired tests")
ap.add_argument("--data", default="Data")
ap.add_argument("--test_session", default="111022_3")
ap.add_argument("--val_session", default="301121_1")
ap.add_argument("--n_crops", type=int, default=500)
ap.add_argument("--shifts", default="0,0;1,1;2,2;3,3;5,5;8,8;-3,0;0,-3")
ap.add_argument("--sweep", default="A1,C1,paper", help="run-name prefixes evaluated in the shift sweep")
ap.add_argument("--buffers_km", default="0,1,2")
ap.add_argument("--row_m", type=float, default=3.92)
ap.add_argument("--out", default="results/ablation")
ap.add_argument("--no_latency", action="store_true")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.n_crops, args.shifts = min(args.n_crops, 50), "0,0;3,3;-3,0"

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHIFTS = [tuple(int(v) for v in s.split(",")) for s in args.shifts.split(";") if s.strip()]
SWEEP = [s.strip() for s in args.sweep.split(",") if s.strip()]
torch.backends.cudnn.benchmark = False


def log(*a):
    print(*a, flush=True)
    with open(OUT / "log.txt", "a") as f:
        print(*a, file=f)


log(f"== eval_ablation  {time.strftime('%Y-%m-%d %H:%M')}  device={DEV}")

# ---------------------------------------------------------------- runs
runs = OrderedDict()
folders = args.runs if args.runs else sorted(glob.glob("runs/[ABC][0-9]*_*"))
for fd in folders:
    fd = Path(fd)
    if not (fd / "config.yaml").exists() or not (fd / "best.pth").exists():
        log(f"skip {fd}: no config.yaml/best.pth")
        continue
    cfg = yaml.safe_load(open(fd / "config.yaml"))
    name = cfg.get("run_name") or fd.name
    runs[name] = dict(cfg=cfg, ckpt=fd / "best.pth", folder=fd)
if args.paper_ckpt and Path(args.paper_ckpt).exists():
    runs["paper"] = dict(cfg=dict(model="MSGNet_w_F_Full", stn_enable=1, weights=[1, 1, 1]),
                         ckpt=Path(args.paper_ckpt), folder=Path(args.paper_ckpt).parent)
elif args.paper_ckpt:
    log(f"paper checkpoint {args.paper_ckpt} not found: skipped")
if not runs:
    raise SystemExit("no runs found")
log("runs: " + ", ".join(f"{k} ({v['cfg']['model']}, stn={v['cfg'].get('stn_enable', 1)}, w={v['cfg'].get('weights')})"
                         for k, v in runs.items()))


def load(run):
    net = _build_net(run["cfg"])
    ck = torch.load(run["ckpt"], map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.to(DEV).eval(), ck.get("epoch", None)


# ---------------------------------------------------------------- data (cached once per session)
def make_split(session):
    ds = ThermalVisDataset(dataPath=args.data, scale_factor=8, HR_downsample=2, original_size=(256, 336),
                           LR_upsample=0, Sessions=[session], numPermutationPerImage=args.n_crops)
    m, s = ds.stats[session][-1]
    t = ds.ThermalVisImgs[session][-1]
    samples = [ds[i] for i in range(args.n_crops)]
    return dict(ds=ds, m=m, s=s, peak=float(t.max()) - float(t.min()), samples=samples,
                locs=[CropLocator(ds, session)(i) for i in range(args.n_crops)])


splits = OrderedDict(test=make_split(args.test_session))
if args.val_session:
    splits["val"] = make_split(args.val_session)
for k, v in splits.items():
    log(f"{k}: session {getattr(args, k + '_session')}, {args.n_crops} crops, peak {v['peak']:g}")

DIST = DISTS().to(DEV).eval()


def mm(x):
    return (x - x.amin((2, 3), keepdim=True)) / (x.amax((2, 3), keepdim=True) - x.amin((2, 3), keepdim=True) + 1e-8)


@torch.no_grad()
def evaluate(net, split, name, tx=0, ty=0, probe=None, extra=None):
    rows = []
    for i, (LR, HR, vis) in enumerate(split["samples"]):
        LR, HR, vis = LR[None].float().to(DEV), HR[None].float().to(DEV), vis[None].float().to(DEV)
        out = net(LR, shift_img(vis, tx, ty))[-1] * split["s"] + split["m"] if net is not None \
            else bicubic_x8(LR) * split["s"] + split["m"]
        hr = HR * split["s"] + split["m"]
        mse = float(Fnn.mse_loss(out, hr))
        r = OrderedDict(model=name, split=extra, shift_x=tx, shift_y=ty, crop_idx=i, **split["locs"][i],
                        mae=float((out - hr).abs().mean()), mse=mse, psnr=psnr_from_mse(mse, split["peak"]),
                        ssim=float(structural_similarity(hr[0, 0].cpu().numpy(), out[0, 0].cpu().numpy(),
                                                         data_range=split["peak"])),
                        dists=float(DIST(mm(hr), mm(out))))
        r.update(modality_disagreement(out, hr, vis))
        if probe is not None:
            r["stn_tx_px"], r["stn_ty_px"] = probe.read()
        rows.append(r)
    return rows


def shift_img(v, tx, ty):
    return shift_visible(v, tx, ty)


def agg(rows, peak):
    a = OrderedDict(n=len(rows), PSNR=psnr_from_mse(np.mean([r["mse"] for r in rows]), peak))
    for k in ["psnr", "mae", "ssim", "dists", "grad_ratio_thermal_only", "grad_ratio_visible_only"]:
        m_, lo, hi = bootstrap_ci([r[k] for r in rows])
        a[k], a[k + "_lo"], a[k + "_hi"] = m_, lo, hi
    return a


# ---------------------------------------------------------------- evaluation loop
all_rows, table, sweep_rows = [], [], []
t0 = time.time()
for name, run in list(runs.items()) + [("Bicubic", None)]:
    if run is None:
        net, epoch, n_par, probe = None, None, None, None
    else:
        net, epoch = load(run)
        n_par = sum(p.numel() for p in net.parameters()) / 1e6
        probe = STNProbe(net)
    row = OrderedDict(run=name, model=run["cfg"]["model"] if run else "bicubic",
                      stn_enable=run["cfg"].get("stn_enable", 1) if run else "",
                      weights=str(run["cfg"].get("weights")) if run else "",
                      best_epoch=(epoch + 1) if epoch is not None else "", params_M=n_par)
    for sp_name, sp in splits.items():
        rows = evaluate(net, sp, name, probe=probe, extra=sp_name)
        all_rows += rows
        a = agg(rows, sp["peak"])
        row.update({f"{sp_name}_PSNR": a["PSNR"], f"{sp_name}_MAE": a["mae"],
                    f"{sp_name}_MAE_CI": f"[{a['mae_lo']:.2f}, {a['mae_hi']:.2f}]",
                    f"{sp_name}_SSIM": a["ssim"], f"{sp_name}_DISTS": a["dists"]})
    if run is not None and any(name.startswith(s) for s in SWEEP):
        for (tx, ty) in SHIFTS:
            rows = evaluate(net, splits["test"], name, tx, ty, probe=probe, extra="test")
            if (tx, ty) != (0, 0):
                all_rows += rows
            a = agg(rows, splits["test"]["peak"])
            sr = OrderedDict(run=name, shift_x=tx, shift_y=ty, PSNR=a["PSNR"], MAE=a["mae"], DISTS=a["dists"],
                             stn_tx=np.nanmean([r.get("stn_tx_px", np.nan) for r in rows]),
                             stn_ty=np.nanmean([r.get("stn_ty_px", np.nan) for r in rows]))
            sweep_rows.append(sr)
        if name.startswith(("A1", "paper")) and run["cfg"].get("stn_enable", 1):
            core = net.module if hasattr(net, "module") else net
            core.stn_enable = 0                              # same weights, STN bypassed at inference
            for (tx, ty) in SHIFTS:
                rows = evaluate(net, splits["test"], name + " (STN bypassed)", tx, ty, extra="test")
                a = agg(rows, splits["test"]["peak"])
                sweep_rows.append(OrderedDict(run=name + " (STN bypassed)", shift_x=tx, shift_y=ty, PSNR=a["PSNR"],
                                              MAE=a["mae"], DISTS=a["dists"], stn_tx=np.nan, stn_ty=np.nan))
            core.stn_enable = 1
    if run is not None and not args.no_latency:
        lat = measure_latency(lambda a_, b_: net(a_, b_), DEV)
        row["latency_ms"], row["latency_sd"] = lat["mean_ms"], lat["std_ms"]
    table.append(row)
    log(f"{name:28s} test PSNR {row['test_PSNR']:.3f}  val PSNR {row.get('val_PSNR', float('nan')):.3f}  "
        f"({(time.time() - t0) / 60:.1f} min)")
    del net
    if DEV.type == "cuda":
        torch.cuda.empty_cache()

write_csv(all_rows, OUT / "per_crop_metrics.csv")
write_csv(table, OUT / "ablation_table.csv")
write_csv(sweep_rows, OUT / "stn_sweep.csv")

# paired tests vs reference on the test split (+ border buffers)
ref = next((k for k in runs if k.startswith(args.ref)), None)
paired = []
if ref:
    for name in [k for k in list(runs) + ["Bicubic"] if k != ref]:
        for buf in [float(b) for b in args.buffers_km.split(",")]:
            A = {r["crop_idx"]: r for r in all_rows if r["model"] == ref and r["split"] == "test"
                 and r["shift_x"] == 0 and r["shift_y"] == 0 and r["row_min"] * args.row_m / 1000 >= buf}
            B = {r["crop_idx"]: r for r in all_rows if r["model"] == name and r["split"] == "test"
                 and r["shift_x"] == 0 and r["shift_y"] == 0 and r["row_min"] * args.row_m / 1000 >= buf}
            ids = sorted(set(A) & set(B))
            for k, better in [("psnr", 1), ("mae", -1), ("dists", -1)]:
                d = np.array([A[i][k] - B[i][k] for i in ids])
                m_, lo, hi = bootstrap_ci(d)
                try:
                    from scipy.stats import wilcoxon
                    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else float("nan")
                except Exception:
                    p = float("nan")
                paired.append(OrderedDict(comparison=f"{ref} - {name}", buffer_km=buf, n=len(ids), metric=k,
                                          mean_diff=m_, ci_lo=lo, ci_hi=hi, p=p,
                                          ref_better_pct=100 * np.mean(np.sign(d) == better)))
    write_csv(paired, OUT / "paired_vs_ref.csv")

# LaTeX table
with open(OUT / "ablation_table.tex", "w") as f:
    f.write("%% PSNR = 10log10(peak^2/mean MSE); val = %s, test = %s, %d crops each\n"
            % (args.val_session, args.test_session, args.n_crops))
    f.write("\\begin{tabular}{llccccccc}\n\\hline\n Run & Weights & Val PSNR & Val MAE & Test PSNR & Test MAE & "
            "Test SSIM & Test DISTS & Params\\\\\n\\hline\n")
    for r in table:
        f.write(f" {r['run']} & {r['weights']} & {r.get('val_PSNR', float('nan')):.2f} & {r.get('val_MAE', float('nan')):.2f} & "
                f"{r['test_PSNR']:.2f} & {r['test_MAE']:.2f} & {r['test_SSIM']:.3f} & {r['test_DISTS']:.3f} & "
                f"{'' if r['params_M'] is None else '%.2f M' % r['params_M']}\\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
with open(OUT / "stn_sweep.tex", "w") as f:
    names = list(OrderedDict.fromkeys(r["run"] for r in sweep_rows))
    f.write("\\begin{tabular}{l" + "c" * len(names) + "}\n\\hline\n $(t_x,t_y)$ & " + " & ".join(names) + "\\\\\n\\hline\n")
    for (tx, ty) in SHIFTS:
        vals = []
        for n in names:
            v = [r for r in sweep_rows if r["run"] == n and r["shift_x"] == tx and r["shift_y"] == ty]
            vals.append(f"{v[0]['PSNR']:.2f}" if v else "--")
        f.write(f" ({tx:+d},{ty:+d}) & " + " & ".join(vals) + "\\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

# figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, run in runs.items():
        p = run["folder"] / "val_log.csv"
        if name != "paper" and p.exists():
            d = np.genfromtxt(p, delimiter=",", names=True)
            if d.size:
                ax.plot(np.atleast_1d(d["epoch"]), np.atleast_1d(d["val_l1"]), label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation L1 (normalised)"); ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig_val_curves.png", dpi=200); plt.close(fig)
    if sweep_rows:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for n in OrderedDict.fromkeys(r["run"] for r in sweep_rows):
            rs = sorted([r for r in sweep_rows if r["run"] == n and r["shift_x"] == r["shift_y"] and r["shift_x"] >= 0],
                        key=lambda r: r["shift_x"])
            ax[0].plot([r["shift_x"] for r in rs], [r["PSNR"] for r in rs], marker="o", label=n)
            if np.isfinite(rs[0]["stn_tx"]):
                b = rs[0]["stn_tx"]
                ax[1].plot([r["shift_x"] for r in rs], [r["stn_tx"] - b for r in rs], marker="o", label=n)
        mx = max(s[0] for s in SHIFTS)
        ax[1].plot([0, mx], [0, mx], "k--", lw=1, label="ideal")
        ax[0].set_xlabel("visible-guide shift $t_x=t_y$ [HR px]"); ax[0].set_ylabel("test PSNR [dB]")
        ax[1].set_xlabel("applied shift [HR px]"); ax[1].set_ylabel("STN-recovered $t_x$ [HR px]")
        for a_ in ax:
            a_.grid(alpha=.3); a_.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "fig_stn_sweep.png", dpi=200); plt.close(fig)
except Exception as e:
    log(f"(figures skipped: {e})")

# summary
with open(OUT / "summary.md", "w") as f:
    f.write(f"# Ablation evaluation ({time.strftime('%Y-%m-%d')}), {args.n_crops} crops per split\n\n")
    cols = ["run", "model", "stn_enable", "weights", "best_epoch", "params_M", "val_PSNR", "val_MAE",
            "test_PSNR", "test_MAE", "test_MAE_CI", "test_SSIM", "test_DISTS", "latency_ms"]
    f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for r in table:
        f.write("| " + " | ".join(f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                                  for c in cols) + " |\n")
    if paired:
        f.write(f"\n## Paired vs {ref} (test; mean diff, 95% CI, Wilcoxon p, % crops {ref} better)\n\n")
        for r in paired:
            f.write(f"- {r['comparison']} | {r['buffer_km']} km | {r['metric']}: {r['mean_diff']:+.4f} "
                    f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['p']:.1e}, {r['ref_better_pct']:.1f}%\n")
    if sweep_rows:
        f.write("\n## Shift sweep (test PSNR)\n\n")
        for r in sweep_rows:
            f.write(f"- {r['run']} ({r['shift_x']},{r['shift_y']}): {r['PSNR']:.3f} dB, MAE {r['MAE']:.2f}, "
                    f"STN est ({r['stn_tx']:+.2f}, {r['stn_ty']:+.2f})\n")
json.dump(dict(date=time.strftime("%Y-%m-%d %H:%M"), device=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
               runs={k: str(v["ckpt"]) for k, v in runs.items()}, args=vars(args)),
          open(OUT / "run_info.json", "w"), indent=2)
log(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT}")
