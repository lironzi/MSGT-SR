from Dataset import ThermalVisDataset
import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from pathlib import Path
import datetime
from misc.MultiScaleLoss import MultiScaleLoss
from torchvision.transforms import Resize, InterpolationMode
from tqdm import tqdm
from misc.logger import Logger
import yaml

# ---------------------------------------------------------------------------------------------
# Training loop. Notes:
#   1. fixed random seed (cfg "seed", default 0) -> every ablation run starts from the same init
#   2. validation set reset before every validation -> identical crops AND flips each epoch
#   3. stn_enable from the config is now passed to MSGNet_w_F_Full (it was silently ignored)
#   4. new model names for the architecture ablation (Model/MSGNet_ablation.py):
#        "MSGNet_single_guidance"  progressive x2-x2-x2, visible fused only at x8
#        "MSGNet_oneshot"          one x8 upsampling, visible fused once at x8
#   5. run folder = runs/<run_name>_<date-time> (cfg "run_name"), so parallel jobs don't collide
#   6. model modules are imported only when selected, so this file also works in a repo that
#      contains only MSGNet_w_F_Full (no MSGNet / ChasNet / base3)
#   7. a per-epoch CSV (val_log.csv) next to train_log.txt
#   8. optional gradient accumulation (cfg "accum_steps", default 1): effective batch = batch_size,
#      processed as accum_steps micro-batches of batch_size/accum_steps on ONE GPU (less memory).
#      Same optimiser updates as the full batch (losses are means); the CBAM BatchNorm sees
#      micro-batches, exactly as each GPU did under DataParallel.
# Everything else (loss, optimiser, data, best-on-validation checkpointing) is unchanged.
# ---------------------------------------------------------------------------------------------


def _build_net(cfg):
    name = cfg["model"]
    if name == "MSGNet_w_F_Full":
        from Model import MSGNet_w_F_Full
        return MSGNet_w_F_Full.Net(stn_enable=cfg.get("stn_enable", 1),
                                   base_mode=cfg.get("base_mode", "nearest"),       # optional variant switch
                                   cbam_spatial=bool(cfg.get("cbam_spatial", 1)))   # optional variant switch
    if name == "MSGNet_single_guidance":
        from Model import MSGNet_ablation
        return MSGNet_ablation.Net(stn_enable=cfg.get("stn_enable", 1), guidance="single",
                                   base_mode=cfg.get("base_mode", "nearest"), cbam_spatial=bool(cfg.get("cbam_spatial", 1)))
    if name == "MSGNet_oneshot":
        from Model import MSGNet_ablation
        return MSGNet_ablation.Net(stn_enable=cfg.get("stn_enable", 1), upsampling="oneshot",
                                   base_mode=cfg.get("base_mode", "nearest"), cbam_spatial=bool(cfg.get("cbam_spatial", 1)))
    if name == "MSGNet_w_F":
        from Model import MSGNet_w_F
        return MSGNet_w_F.Net(num_channels=cfg["num_channels"], stn_enable=cfg["stn_enable"])
    if name == "MSGNet":
        from Model import MSGNet
        return MSGNet.Net(num_channels=cfg["num_channels"])
    if name == "NAF":
        from Model.base3 import Net
        return Net()
    if name == "ChasNet":
        from Model import ChasNet
        return ChasNet.Net(scale_factor=cfg["train_dataset"]["scale_factor"])
    if name == "MSGNet_w_F2":
        from Model import MSGNet_w_F2
        return MSGNet_w_F2.Net()
    raise ValueError(f"unknown model {name}")


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = int(cfg.get("seed", 0))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        net = _build_net(cfg).to(self.device)
        print(f"model {cfg['model']}: {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M parameters, "
              f"stn_enable={cfg.get('stn_enable', 1)}, weights={cfg['weights']}, seed={seed}")
        n_gpu = torch.cuda.device_count()
        print(f"GPUs visible: {n_gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}) "
              f"-> DataParallel {'ON' if n_gpu > 1 else 'OFF'}", flush=True)
        if n_gpu > 1:
            net = torch.nn.DataParallel(net)
        self.net = net
        self.best_val_loss = float("inf")
        self.loss = MultiScaleLoss(device=self.device, alpha=cfg["alpha"], weights=cfg["weights"])
        self.l1_loss = torch.nn.L1Loss()
        self.train_dataset = ThermalVisDataset(**cfg["train_dataset"])
        self.val_dataset = ThermalVisDataset(**cfg["val_dataset"])
        # gradient accumulation: the effective batch stays cfg["batch_size"]; each forward/backward
        # sees batch_size/accum_steps samples (e.g. 96 = 2 x 48), the optimiser steps once per 96.
        self.accum = int(cfg.get("accum_steps", 1))
        assert cfg["batch_size"] % self.accum == 0, "batch_size must be divisible by accum_steps"
        self.micro_bs = cfg["batch_size"] // self.accum
        print(f"effective batch {cfg['batch_size']} = {self.accum} x {self.micro_bs} (accum_steps={self.accum})", flush=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.micro_bs, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.micro_bs, shuffle=False)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])

        self.start_epoch = 0

        # ----------------------------
        # Create run folder
        # ----------------------------
        self.project_root = Path(__file__).resolve().parents[1]
        resume_path = cfg.get("resume_path", None)
        if resume_path:
            resume_path = Path(resume_path)
            if not resume_path.is_absolute():
                resume_path = (self.project_root / resume_path).resolve()
            self.run_dir = resume_path.parent            # resume: keep writing into the same run folder
        else:
            now = datetime.datetime.now()
            stamp = f"{now.day:02d}_{now.month:02d}_{now.year}-{now.hour:02d}_{now.minute:02d}"
            folder = f"{cfg['run_name']}_{stamp}" if cfg.get("run_name") else stamp
            self.run_dir = Path("runs") / folder
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.project_root / self.run_dir)
        with open(self.run_dir / ("config_resume.yaml" if resume_path else "config.yaml"), "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, default_flow_style=False)
        self.csv_path = self.run_dir / "val_log.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w") as f:
                f.write("epoch,train_loss,val_l1,best\n")

        # Resume if requested (e.g. resume_path: runs/<run>/last.pth)
        if resume_path:
            self._load_checkpoint(resume_path)

        # ----------------------------
        # Resizers for multi-scale supervision
        # ----------------------------
        GT_size = cfg["train_dataset"]["original_size"]
        hr_downsample = cfg["train_dataset"].get("HR_downsample", 1)
        if hr_downsample:
            GT_size = (GT_size[0] // hr_downsample, GT_size[1] // hr_downsample)

        self.re_2 = Resize(size=(GT_size[0] // 4, GT_size[1] // 4), interpolation=InterpolationMode.BICUBIC, antialias=True)
        self.re_4 = Resize(size=(GT_size[0] // 2, GT_size[1] // 2), interpolation=InterpolationMode.BICUBIC, antialias=True)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt["model_state_dict"]
        model_keys = list(self.net.state_dict().keys())
        ckpt_keys = list(state_dict.keys())
        if not ckpt_keys[0].startswith("module.") and model_keys[0].startswith("module."):
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif ckpt_keys[0].startswith("module.") and not model_keys[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        csv_path = Path(path).parent / "val_log.csv"           # recover the best validation loss so far
        if csv_path.exists():
            vals = [float(l.split(",")[2]) for l in open(csv_path).read().splitlines()[1:] if l.strip()]
            self.best_val_loss = min(vals) if vals else float("inf")
        self.logger.log_string(f"Resumed from epoch {self.start_epoch}\n")

    def train(self):
        for epoch in range(self.start_epoch, self.cfg["num_epochs"]):
            self.net.train()
            running_loss = 0.0
            self.train_dataset.setSeed(epoch + 1)

            n_steps, n_micro, step_loss = 0, 0, 0.0
            self.optimizer.zero_grad()
            for iteration, (LR, HR, vis) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                LR = LR.to(self.device).float()
                HR = HR.to(self.device).float()
                vis = vis.to(self.device).float()

                HR_0 = self.re_2(HR)
                HR_1 = self.re_4(HR)

                if self.cfg["model"] == "ChasNet":
                    try:
                        outputs = self.net(LR)
                    except Exception:
                        continue
                else:
                    outputs = self.net(LR, vis)
                if self.cfg["model"] == "NAF" or self.cfg["model"] == "ChasNet":
                    loss = self.l1_loss(HR, outputs)
                else:
                    loss = self.loss(outputs, [HR_0, HR_1, HR])

                (loss / self.accum).backward()          # sum of loss/accum over micro-batches = full-batch mean
                step_loss += loss.item() / self.accum
                n_micro += 1
                if n_micro == self.accum:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    running_loss += step_loss
                    n_micro, step_loss, n_steps = 0, 0.0, n_steps + 1
            if n_micro:
                # leftover micro-batches at the end of the epoch: the original trainer took one smaller
                # last step (6000 crops / 96 -> 62 full steps + one of 48); reproduce it exactly
                for prm in self.net.parameters():
                    if prm.grad is not None:
                        prm.grad.mul_(self.accum / n_micro)
                self.optimizer.step()
                self.optimizer.zero_grad()
                running_loss += step_loss * self.accum / n_micro
                n_steps += 1

            avg_loss = running_loss / max(n_steps, 1)
            val_result = self.validate()
            self.logger.log_epoch(epoch, avg_loss, val_result)
            is_best = val_result < self.best_val_loss
            if is_best:
                self.best_val_loss = val_result
                self.logger.log_string(f"New best model at epoch {epoch+1} with val loss {val_result:.6f}\n")
                self.logger.save_checkpoint(self.net, self.optimizer, epoch, avg_loss,
                                            path=self.run_dir / "best.pth")
            # always keep the last epoch too, so an interrupted job can be resumed (resume_path)
            self.logger.save_checkpoint(self.net, self.optimizer, epoch, avg_loss, path=self.run_dir / "last.pth")
            with open(self.csv_path, "a") as f:
                f.write(f"{epoch + 1},{avg_loss:.6f},{val_result:.6f},{int(is_best)}\n")
        # marker used by run_ablation.sh to skip finished runs (and to auto-resume unfinished ones)
        with open(self.run_dir / "DONE", "w") as f:
            f.write(f"finished {self.cfg['num_epochs']} epochs; best val L1 {self.best_val_loss:.6f}\n")

    def validate(self):
        self.net.eval()
        self.val_dataset.setSeed(0)          # identical validation crops + flips at every epoch
        running_val_loss = 0.0
        n = 0
        with torch.no_grad():
            for LR, HR, vis in self.val_loader:
                LR = LR.to(self.device).float()
                HR = HR.to(self.device).float()
                vis = vis.to(self.device).float()
                if self.cfg["model"] == "ChasNet":
                    try:
                        outputs = self.net(LR)
                    except Exception:
                        continue
                else:
                    outputs = self.net(LR, vis)
                if self.cfg["model"] == "NAF" or self.cfg["model"] == "ChasNet":
                    loss_main = self.l1_loss(outputs, HR)
                else:
                    loss_main = self.l1_loss(outputs[2], HR)
                running_val_loss += loss_main.item()
                n += 1
        return running_val_loss / max(n, 1)
