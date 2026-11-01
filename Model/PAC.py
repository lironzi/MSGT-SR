import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PAC2d(nn.Module):
    """
    Pixel-Adaptive Convolution (PAC) — PyTorch implementation.
    Implements the core idea from Su et al., CVPR 2019.

    y_i = sum_j [ W_c[..., j] * phi(g_i, g_j) * x_j ]
    where:
      - W_c are learnable kernel weights (out_channels x in_channels x KxK)
      - phi(g_i, g_j) is a guidance-based affinity (e.g. exp(-||g_i-g_j||^2/(2*sigma^2)))

    Args:
        in_channels: input channels
        out_channels: output channels
        guide_channels: channels in guidance map (g). Could be 1 (grayscale) or 3 (RGB) or learned embedding.
        kernel_size: spatial kernel size (int)
        stride, padding, dilation: standard conv args
        sigma_learnable: if True, learn a per-output-channel sigma; otherwise fixed scalar sigma
        sigma_init: initial sigma value (scalar) if not learnable
        guidance_hidden: hidden dim for guidance embedding (if None uses guide_channels)
        normalize: if True, normalize affinities so sum_j phi = 1 (local softmax-like)
        bias: add learnable bias
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        guide_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        sigma_learnable=False,
        sigma_init=0.1,
        guidance_hidden=None,
        normalize=True,
        bias=True,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size should be odd for symmetric neighborhoods"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.guide_channels = guide_channels
        self.kernel_size = kernel_size
        self.K = kernel_size * kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.dilation = dilation
        self.normalize = normalize

        # Learnable convolution kernel (like Conv2d weights)
        # Shape: (out_ch, in_ch, K)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.K) * math.sqrt(2.0 / (in_channels * self.K))
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Guidance embedding: a tiny MLP implemented as 1x1 conv(s)
        gh = guidance_hidden if guidance_hidden is not None else guide_channels
        if guide_channels != gh:
            self.guidance_embed = nn.Sequential(
                nn.Conv2d(guide_channels, gh, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(gh, gh, kernel_size=1, bias=False),
            )
            self.final_guide_ch = gh
        else:
            # single 1x1 conv to allow learnable projection (keeps API simple)
            self.guidance_embed = nn.Conv2d(guide_channels, gh, kernel_size=1, bias=False)
            self.final_guide_ch = gh

        # sigma: either learnable per-output-channel (out_channels,) or fixed scalar
        if sigma_learnable:
            # store log-sigma to ensure positivity
            self.log_sigma = nn.Parameter(torch.log(torch.ones(out_channels) * sigma_init))
            self.sigma_learnable = True
        else:
            self.register_buffer("sigma", torch.tensor(float(sigma_init)))
            self.sigma_learnable = False

        # small eps for stability when normalizing
        self.eps = 1e-6

    def _get_sigma(self):
        if self.sigma_learnable:
            # ensure positivity
            return torch.clamp(self.log_sigma.exp(), min=1e-6)  # shape (out_channels,)
        else:
            return self.sigma  # scalar

    def forward(self, x, g):
        """
        Args:
            x: input feature map (B, C_in, H, W)
            g: guidance map (B, G, H, W) — can be RGB or learned guide
        Returns:
            out: (B, C_out, H_out, W_out)
        """
        B, C_in, H, W = x.shape
        assert C_in == self.in_channels, "input channel mismatch"

        # --- 1. embed guidance ---
        g_emb = self.guidance_embed(g)  # (B, G_emb, H, W)
        B, G_emb, H_g, W_g = g_emb.shape
        assert H_g == H and W_g == W, "guidance must match spatial size of x (or be pre-resized)"

        # --- 2. unfold guidance and input patches ---
        # unfold returns shape (B, G_emb * K, L) where L = H_out * W_out
        # here we want H_out = floor((H + 2*pad - dilation*(k-1)-1)/stride +1) but for same padding H_out=H when stride=1
        g_unf = F.unfold(g_emb, kernel_size=self.kernel_size,
                         dilation=self.dilation, padding=self.padding, stride=self.stride)
        # shape: (B, G_emb * K, L)
        L = g_unf.shape[-1]  # number of spatial positions after stride

        # reshape to (B, G_emb, K, L)
        g_unf = g_unf.view(B, G_emb, self.K, L)  # (B, G, K, L)

        # central guidance g_i: extract center index among K (middle of kernel)
        center_idx = self.K // 2
        # get center map unfolded (we can also use indexing into g_emb)
        # make (B, G, 1, L)
        g_center = g_unf[:, :, center_idx:center_idx+1, :]  # (B, G, 1, L)

        # compute guidance differences: (B, G, K, L) - (B, G, 1, L) -> (B, G, K, L)
        diff = g_unf - g_center  # broadcasting

        # --- 3. affinity (phi) computation ---
        # squared L2 norm across guidance channels: (B, K, L)
        dist2 = (diff * diff).sum(dim=1)  # (B, K, L)

        sigma = self._get_sigma()
        if self.sigma_learnable:
            # sigma per-output-channel: shape (out_channels,)
            # but dist2 is not yet per-output-channel. We'll handle multiplication later.
            # We compute affinities common to all outputs using a scalar mean sigma to avoid per-output expensive ops
            # Option: use per-output sigma to scale kernels differently by broadcasting later.
            # For simplicity, compute a "shared" affinity here:
            # Use mean sigma for affinity; per-output scaling will be applied to weights if desired.
            sigma_for_affinity = sigma.mean()
        else:
            sigma_for_affinity = sigma

        # phi = exp(-dist2 / (2 * sigma^2))
        denom = 2.0 * float(sigma_for_affinity.item()) ** 2 + self.eps
        phi = torch.exp(-dist2 / denom)  # (B, K, L)

        if self.normalize:
            # normalize affinities so sum_j phi = 1 for each center pixel i
            phi_sum = phi.sum(dim=1, keepdim=True)  # (B, 1, L)
            phi = phi / (phi_sum + self.eps)  # (B, K, L)

        # --- 4. extract x patches ---
        x_unf = F.unfold(x, kernel_size=self.kernel_size,
                         dilation=self.dilation, padding=self.padding, stride=self.stride)
        # x_unf: (B, C_in * K, L)
        x_unf = x_unf.view(B, C_in, self.K, L)  # (B, C_in, K, L)

        # multiply patches by phi: broadcast phi to channels
        # weighted_patches: (B, C_in, K, L)
        weighted = x_unf * phi.unsqueeze(1)

        # collapse the K dimension with kernel weights
        # weight: (C_out, C_in, K)
        # we want to compute output at each spatial location via: out_b[:, l] = sum_{c_in, k} weight * weighted[:,:,k,l]
        # using einsum: "bckl,ock->bol" then reshape to spatial map
        out_unrolled = torch.einsum("bckl,ock->bol", weighted, self.weight)  # (B, C_out, L)

        # reshape to (B, C_out, H_out, W_out)
        H_out = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        out = out_unrolled.view(B, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

if __name__=="__main__":
    # example
    B, C_in, H, W = 2, 64, 128, 128
    x = torch.randn(B, C_in, H, W)
    g = torch.randn(B, C_in, H, W)  # e.g., RGB guidance

    pac = PAC2d(in_channels=64, out_channels=128, guide_channels=64, kernel_size=3,
                sigma_learnable=False, sigma_init=0.1, normalize=True)

    y = pac(x, g)  # (B, 128, H, W) when stride=1
