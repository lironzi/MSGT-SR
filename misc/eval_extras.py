"""
eval_extras.py: evaluation helpers.

Model-agnostic helpers used by test.py (MSGT-SR) and, unchanged, by the QQ /
GuidedSR / SwinFuSR / ChasNet test scripts, so that every model is scored on the
*same* 500 test crops with the *same* code.

Contents
  - per-crop metrics -> CSV, so paired tests / confidence intervals are possible
  - crop location (rows in the test mosaic) -> distance to the train/test border
  - controlled translation of the visible input + the STN's estimated shift
  - thermal-only vs visible-only edge analysis (does visible texture leak in?)
  - latency with an explicit protocol (warm-up, CUDA sync, fp32, batch 1)
"""
import csv
import math
import time

import numpy as np
import torch
import torch.nn.functional as Fnn
import torchvision.transforms.functional as TF


# --------------------------------------------------------------------------
# Basic metrics
# --------------------------------------------------------------------------
def session_data_range(ds, session):
    """max - min of the raw thermal mosaic (the PSNR / SSIM peak). 1538 for 111022_3."""
    t = ds.ThermalVisImgs[session][-1]
    return float(t.max()) - float(t.min())


def psnr_from_mse(mse, data_range):
    return 10.0 * math.log10(data_range ** 2 / max(mse, 1e-12))


def bicubic_x8(LR):
    """Bicubic x8 reference, same call as the BIC images written by test.py."""
    return Fnn.interpolate(LR, scale_factor=8, mode="bicubic", align_corners=False)


# --------------------------------------------------------------------------
# Crop location inside the test mosaic
# --------------------------------------------------------------------------
class CropLocator:
    """Maps a test crop back to rows/cols of the *un-rotated* session mosaic.

    Dataset.py rotates the whole mosaic (expand=True) and then crops, so we rotate
    a row/column index map with exactly the same call and crop it identically.
    Rows are counted from the northern edge (the GeoTIFF tie-point is the top-left
    corner), i.e. row_min * pixel_size = distance to the border with 111022_2.
    """

    def __init__(self, ds, session):
        self.ds, self.session = ds, session
        H, W = ds.ThermalVisImgs[session][-1].shape
        rows = torch.arange(1, H + 1, dtype=torch.float32).view(1, H, 1).expand(1, H, W)
        cols = torch.arange(1, W + 1, dtype=torch.float32).view(1, 1, W).expand(1, H, W)
        self.index_map = torch.cat([rows, cols], 0)  # 0 = outside the mosaic after rotation

    def __call__(self, idx):
        x, y, theta = self.ds.cropPositions[self.session][idx]
        h, w = self.ds.original_size
        r = TF.rotate(self.index_map, theta, interpolation=TF.InterpolationMode.NEAREST, expand=True)
        crop = r[:, y:y + h, x:x + w]
        valid = crop[0] > 0
        rr, cc = crop[0][valid] - 1, crop[1][valid] - 1
        return dict(x_rot=int(x), y_rot=int(y), theta=float(theta),
                    row_min=float(rr.min()), row_max=float(rr.max()),
                    row_center=float(rr.mean()), col_center=float(cc.mean()),
                    frac_valid=float(valid.float().mean()))


# --------------------------------------------------------------------------
# Visible-input perturbation + STN read-out
# --------------------------------------------------------------------------
def shift_visible(vis, tx, ty):
    """Translate the HR visible input by integer (tx, ty) HR pixels, replicate padding.
    out[y, x] = vis[y - ty, x - tx]  (positive tx = content moves right)."""
    tx, ty = int(tx), int(ty)
    if tx == 0 and ty == 0:
        return vis
    H, W = vis.shape[-2:]
    v = Fnn.pad(vis, (max(tx, 0), max(-tx, 0), max(ty, 0), max(-ty, 0)), mode="replicate")
    x0, y0 = max(-tx, 0), max(-ty, 0)
    return v[..., y0:y0 + H, x0:x0 + W]


class STNProbe:
    """Records the translation estimated by MSGT-SR's STN (2-parameter, translation only).

    affine_grid uses normalised coordinates (align_corners=False), so a theta
    translation t corresponds to t * W/2 HR pixels. grid_sample then *samples*
    at x + t, i.e. it undoes a visible shift of +t. So after shifting the visible
    input by +tx, a perfect STN reports est_tx ~= est_tx(at shift 0) + tx.
    Note: the regressor bound tanh * 20/W in normalised units is +/-10 HR px.
    """

    def __init__(self, net):
        core = net.module if hasattr(net, "module") else net
        self.last = None
        self.handle = None
        if hasattr(core, "theta_regressor"):
            self.handle = core.theta_regressor.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        fm = inputs[0]
        H, W = fm.shape[-2:]
        t = output[:, :, 2].detach().cpu()
        self.last = (t[:, 0] * W / 2.0, t[:, 1] * H / 2.0)

    def read(self, z=0):
        if self.last is None:
            return float("nan"), float("nan")
        v = (float(self.last[0][z]), float(self.last[1][z]))
        self.last = None
        return v


# --------------------------------------------------------------------------
# Thermal-only vs visible-only structure
# --------------------------------------------------------------------------
_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)


def _grad_mag(x):
    k = _SOBEL_X.to(x.device, x.dtype)
    gx = Fnn.conv2d(x, k, padding=1)
    gy = Fnn.conv2d(x, k.transpose(2, 3), padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)[..., 1:-1, 1:-1]  # drop padded border


def modality_disagreement(output, HR, vis, q=0.90, tol_px=1):
    """Per-crop edge analysis, for batch element 0 (test uses batch_size=1).

    thermal-only edges : strong in HR thermal, no strong visible edge within tol_px
    visible-only edges : strong in visible,   no strong HR-thermal edge within tol_px
    Strong = top (1-q) of the gradient magnitude of that crop.

    grad_ratio_visible_only > 1  -> visible texture transferred into the thermal output
    grad_ratio_thermal_only < 1  -> thermal-only detail (anomalies) smoothed away
    """
    out, hr = output[:1], HR[:1]
    g_hr, g_out = _grad_mag(hr), _grad_mag(out)
    g_vis = _grad_mag(vis[:1].mean(1, keepdim=True))
    t_edge = g_hr >= torch.quantile(g_hr.flatten(), q)
    v_edge = g_vis >= torch.quantile(g_vis.flatten(), q)
    k = 2 * tol_px + 1
    t_dil = Fnn.max_pool2d(t_edge.float(), k, 1, tol_px) > 0
    v_dil = Fnn.max_pool2d(v_edge.float(), k, 1, tol_px) > 0
    t_only, v_only = t_edge & ~v_dil, v_edge & ~t_dil
    err = (out - hr).abs()[..., 1:-1, 1:-1]

    def masked(a, m):
        return float(a[m].mean()) if m.any() else float("nan")

    return dict(
        frac_thermal_only=float(t_only.float().mean()),
        frac_visible_only=float(v_only.float().mean()),
        mae_thermal_only=masked(err, t_only),
        mae_visible_only=masked(err, v_only),
        grad_ratio_thermal_only=masked(g_out, t_only) / max(masked(g_hr, t_only), 1e-12),
        grad_ratio_visible_only=masked(g_out, v_only) / max(masked(g_hr, v_only), 1e-12),
    )


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------
@torch.no_grad()
def measure_latency(forward, device, lr_shape=(1, 1, 16, 21), vis_shape=(1, 3, 128, 168),
                    n_warmup=20, n_runs=100, dtype=torch.float32):
    """forward(LR, vis) -> anything.  Times model forward only (no data loading)."""
    LR = torch.randn(lr_shape, device=device, dtype=dtype)
    vis = torch.randn(vis_shape, device=device, dtype=dtype)
    cuda = device.type == "cuda"
    for _ in range(n_warmup):
        forward(LR, vis)
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        forward(LR, vis)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return dict(mean_ms=float(t.mean()), std_ms=float(t.std()), median_ms=float(np.median(t)),
                p95_ms=float(np.percentile(t, 95)), n_warmup=n_warmup, n_runs=n_runs,
                batch=lr_shape[0], lr_hw=f"{lr_shape[2]}x{lr_shape[3]}",
                hr_hw=f"{vis_shape[2]}x{vis_shape[3]}", precision=str(dtype).replace("torch.", ""),
                device=torch.cuda.get_device_name(device) if cuda else "cpu",
                torch=torch.__version__, cudnn_benchmark=torch.backends.cudnn.benchmark)


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (n_boot, x.size))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, data_range, label=""):
    """Dataset-level summary. 'psnr_of_mean_mse' = convention of Table 1;
    'psnr' mean is the per-crop average (reported alongside its 95% CI)."""
    mse = np.mean([r["mse"] for r in rows])
    print(f"\n=== {label}  (n={len(rows)}, peak={data_range:g}) ===")
    print(f"PSNR of mean MSE : {psnr_from_mse(mse, data_range):.3f} dB   (Table 1 convention)")
    for k in ["psnr", "mae", "ssim", "dists", "lpips", "bic_psnr", "bic_mae",
              "mae_thermal_only", "mae_visible_only", "grad_ratio_thermal_only",
              "grad_ratio_visible_only", "stn_tx_px", "stn_ty_px"]:
        if k in rows[0]:
            m, lo, hi = bootstrap_ci([r[k] for r in rows])
            sd = np.nanstd([r[k] for r in rows])
            print(f"{k:24s}: {m:9.4f}   95% CI [{lo:.4f}, {hi:.4f}]   per-crop SD {sd:.4f}")
