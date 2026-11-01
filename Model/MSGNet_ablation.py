"""
MSGNet_ablation.py: architecture controls for the ablation study.

Built from exactly the same blocks as Model/MSGNet_w_F_Full.py (the paper model, left untouched).
The Fuser, upsample and Theta_regressor classes are imported from it, so nothing is re-implemented.

  guidance="single"      (variant a): progressive x2 -> x2 -> x2 as in MSGT-SR, but the visible image
                          is fused ONLY at x8. At x2/x4 the same Fuser block is kept with the thermal
                          features as its own guide ("self-guided"), so depth and parameter count stay
                          the same and only the visible information at x2/x4 is removed.

  upsampling="oneshot"   (variant b): the same three thermal RDBs at LR resolution, ONE x8 sub-pixel
                          upsampling (conv C->64C + PixelShuffle(8) + conv), then STN + fusion at x8.
                          There are no intermediate scales, so the visible image is necessarily fused once.
                          forward() still returns 3 outputs (the first two are the nearest-upsampled
                          LR, carrying no gradient), so the existing loss/test code works unchanged;
                          train it with weights [0, 0, 3].

The defaults (guidance="multiscale", upsampling="progressive") reproduce MSGNet_w_F_Full exactly,
with identical parameter names, so an MSGNet_w_F_Full checkpoint loads into Net().
"""
import torch
from torch import nn

from Model.RDN import RDB
from Model.MSGNet_w_F_Full import Fuser, upsample, Theta_regressor


class Net(nn.Module):
    def __init__(self, band_numbers=3, num_channels=64, stn_enable=1,
                 guidance="multiscale", upsampling="progressive",
                 base_mode="nearest", cbam_spatial=True):
        # base_mode / cbam_spatial: same switches as MSGNet_w_F_Full (defaults = submitted design)
        super().__init__()
        self.base_mode = base_mode
        assert guidance in ("multiscale", "single")
        assert upsampling in ("progressive", "oneshot")
        if upsampling == "oneshot":
            guidance = "single"                      # no intermediate scales to guide
        self.guidance, self.upsampling = guidance, upsampling
        C, growth_rate, num_layers = num_channels, num_channels, 8
        self.num_block = 3

        self.RDB_thermal = nn.ModuleList([RDB(C, growth_rate, num_layers) for _ in range(3)])
        self.RDB_vis = RDB(C, growth_rate, num_layers)
        self.conv_vis_0 = nn.Conv2d(band_numbers, C, kernel_size=3, padding=1)
        self.conv_thermal_0 = nn.Conv2d(1, C, kernel_size=3, padding=1)
        self.theta_regressor = Theta_regressor(num_channels=C)
        self.stn_enable = stn_enable

        if upsampling == "progressive":
            self.upsample_list = nn.ModuleList([upsample(num_channels=C, scale_factor=2) for _ in range(3)])
            self.output_conv = nn.ModuleList([nn.Conv2d(C, 1, kernel_size=1) for _ in range(3)])
            self.fusion_list = nn.ModuleList([Fuser(num_channels=C, cbam_spatial=cbam_spatial) for _ in range(3)])
            if guidance == "multiscale":
                self.downsample_x2 = nn.Conv2d(C, C, padding=1, kernel_size=3, stride=2)
        else:
            self.upsample_x8 = upsample(num_channels=C, scale_factor=8)
            self.output_conv_x8 = nn.Conv2d(C, 1, kernel_size=1)
            self.fusion_x8 = Fuser(num_channels=C, cbam_spatial=cbam_spatial)

    def _up(self, LR, s):
        if self.base_mode == "nearest":
            return torch.nn.functional.interpolate(LR, scale_factor=s)
        return torch.nn.functional.interpolate(LR, scale_factor=s, mode=self.base_mode, align_corners=False)

    def _align(self, vis, thermal):
        if self.stn_enable:
            thetas = self.theta_regressor(torch.concat([vis, thermal], dim=1))
            grid = torch.nn.functional.affine_grid(thetas, size=vis.shape).to(vis.device)
            vis = torch.nn.functional.grid_sample(vis, grid)
        return vis

    def forward(self, LR, vis):
        vis = self.RDB_vis(self.conv_vis_0(vis))
        thermal = self.conv_thermal_0(LR)

        if self.upsampling == "oneshot":
            for rdb in self.RDB_thermal:
                thermal = rdb(thermal)
            thermal = self.upsample_x8(thermal)
            vis = self._align(vis, thermal)
            thermal = self.fusion_x8(thermal, vis)
            out = self._up(LR, 8) + self.output_conv_x8(thermal)
            with torch.no_grad():
                aux = [self._up(LR, s) for s in (2, 4)]
            return aux + [out]

        # progressive x2 -> x2 -> x2 (identical to MSGNet_w_F_Full when guidance == "multiscale")
        if self.guidance == "multiscale":
            d2 = self.downsample_x2(vis)
            vis_list = [self.downsample_x2(d2), d2]
        output_list = []
        for i in range(self.num_block - 1):
            thermal = self.RDB_thermal[i](thermal)
            up = self.upsample_list[i](thermal)
            guide = vis_list[i] if self.guidance == "multiscale" else up    # variant (a): self-guided
            thermal = self.fusion_list[i](up, guide)
            output_list.append(self._up(LR, (i + 1) * 2) + self.output_conv[i](thermal))
        thermal = self.RDB_thermal[-1](thermal)
        thermal = self.upsample_list[-1](thermal)
        vis = self._align(vis, thermal)
        thermal = self.fusion_list[-1](thermal, vis)
        output_list.append(self._up(LR, 8) + self.output_conv[-1](thermal))
        return output_list
