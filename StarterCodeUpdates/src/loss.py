import torch
import torch.nn as nn
import math

def make_gabor_kernel(
    kernel_size: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> torch.Tensor:
    """
    Returns a single Gabor filter of shape (1, 1, kernel_size, kernel_size), on CPU.
    - kernel_size: must be odd (7, 9, 11, etc.)
    - sigma:  std dev of Gaussian envelope
    - theta:  orientation (radians, float)
    - lambd:  wavelength (pixels)
    - gamma:  aspect ratio (default 0.5)
    - psi:    phase offset (default 0.0)
    """
    half = kernel_size // 2

    # Build a 2D meshgrid in CPU floats
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )

    # Rotate coordinates by theta (use math.cos/math.sin since theta is a float)
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)

    # 2D Gabor formula
    gb = torch.exp(
        -0.5 * ((x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (sigma ** 2))
    ) * torch.cos(2.0 * math.pi * x_theta / lambd + psi)

    # Normalize so sum of absolute values = 1 (optional but common)
    gb = gb / gb.abs().sum()

    return gb.view(1, 1, kernel_size, kernel_size)  # shape (1,1,kH,kW)


def make_gabor_bank(
    kernel_size: int,
    sigmas: list[float],
    lambdas: list[float],
    thetas: list[float],
) -> torch.Tensor:
    """
    Builds a bank of Gabor filters on CPU. Returns a tensor of shape
    (N_filters, 1, kernel_size, kernel_size), where N_filters = len(sigmas)*len(lambdas)*len(thetas).
    """
    filters = []
    for sigma in sigmas:
        for lambd in lambdas:
            for theta in thetas:
                gb = make_gabor_kernel(
                    kernel_size=kernel_size,
                    sigma=sigma,
                    theta=theta,
                    lambd=lambd,
                    gamma=0.5,
                    psi=0.0,
                )
                filters.append(gb)
    return torch.cat(filters, dim=0)  # shape: (N_filters, 1, kH, kW)



def gabor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gabor_kernels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Channel-agnostic Gabor-loss.

    pred, target: (B, C, H, W)  with C >= 1
    gabor_kernels: (N_filters, 1, kH, kW), already on the correct device.

    We convolve each of the N filters over each channel of pred/target,
    then average L1 over B, C, N, H, W (if reduction='mean') or sum (if 'sum').
    If reduction='none', returns a tensor of shape (B, C, N, H, W).
    """
    B, C, H, W = pred.shape
    N, Ck, kH, kW = gabor_kernels.shape
    assert Ck == 1, "gabor_kernels must have shape (N, 1, kH, kW)."

    pad_h = kH // 2
    pad_w = kW // 2

    # Reshape to (B*C, 1, H, W) using reshape (not view)
    pred_resh = pred.reshape(B * C, 1, H, W)
    targ_resh = target.reshape(B * C, 1, H, W)

    # Convolve with all N filters in one go:
    #   input:  (B*C, 1, H, W)
    #   weight: (N,   1, kH, kW)
    # → output: (B*C, N, H, W)
    conv_pred = nn.functional.conv2d(pred_resh, gabor_kernels, padding=(pad_h, pad_w))
    conv_true = nn.functional.conv2d(targ_resh, gabor_kernels, padding=(pad_h, pad_w))

    # Now reshape back to (B, C, N, H, W)
    conv_pred = conv_pred.reshape(B, C, N, H, W)
    conv_true = conv_true.reshape(B, C, N, H, W)

    diff = torch.abs(conv_pred - conv_true)  # (B, C, N, H, W)

    if reduction == "mean":
        return diff.mean()
    elif reduction == "sum":
        return diff.sum()
    else:
        return diff  # (B, C, N, H, W)



def cartesian_to_polar(
    img: torch.Tensor,
    center: tuple[float, float],
    R: int | None = None,
    Theta: int | None = None,
) -> torch.Tensor:
    """
    Channel‐agnostic Cartesian→Polar warp.

    img:    (B, C, H, W), C >= 1, on any device
    center: (x0, y0) in pixel coords (floats)
    R:      number of radial samples (int). If None, defaults to floor(max corner distance).
    Theta:  number of angular samples (int). If None, defaults to W.

    Returns: (B, C, R, Theta)

    Each of the C channels is sampled identically. Internally, we build a grid
    on the same device/dtype as img, then call nn.functional.grid_sample.
    """
    B, C, H, W = img.shape
    assert H >= 1 and W >= 1, "Spatial dims must be positive."

    device = img.device
    dtype = img.dtype

    x0, y0 = center

    # 1) Decide R_max & R if None
    if R is None:
        # Compute maximum distance from (x0, y0) to any corner
        corners = torch.tensor(
            [[-x0, -y0], [W - 1 - x0, -y0], [-x0, H - 1 - y0], [W - 1 - x0, H - 1 - y0]],
            device=device,
            dtype=dtype,
        )  # shape (4, 2)
        dists = torch.sqrt((corners ** 2).sum(dim=1))
        R_max = float(dists.max().item())
        R = math.floor(R_max)
    else:
        R_max = float(R - 1)

    if Theta is None:
        Theta = W

    # 2) Build r_lin and θ_lin
    r_lin = torch.linspace(0.0, R_max, steps=R, device=device, dtype=dtype)       # (R,)
    theta_lin = torch.linspace(0.0, 2.0 * math.pi, steps=Theta, device=device, dtype=dtype)  # (Θ,)

    # 3) Make meshgrid (R × Θ)
    r_grid, theta_grid = torch.meshgrid(r_lin, theta_lin, indexing="ij")  # both (R, Θ)

    # 4) Convert (r,θ) → Cartesian float coords
    x_cart = x0 + r_grid * torch.cos(theta_grid)  # (R, Θ)
    y_cart = y0 + r_grid * torch.sin(theta_grid)  # (R, Θ)

    # 5) Normalize to [-1, 1] for grid_sample
    #    x_norm = (2 * x_cart / (W-1)) - 1
    #    y_norm = (2 * y_cart / (H-1)) - 1
    x_norm = (2.0 * x_cart / float(W - 1)) - 1.0  # (R, Θ)
    y_norm = (2.0 * y_cart / float(H - 1)) - 1.0  # (R, Θ)

    # 6) Build base grid shape (R, Θ, 2), then expand to (B, R, Θ, 2)
    grid = torch.stack([x_norm, y_norm], dim=-1)   # (R, Θ, 2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, R, Θ, 2)

    # 7) Use nn.functional.grid_sample to warp each channel of img
    #    Input:  (B, C, H, W)
    #    Grid:   (B, R, Θ, 2)
    #    Output: (B, C, R, Θ)
    polar = nn.functional.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
   
    return polar  # (B, C, R, Θ)

def l1_gabor_hessian_polar_loss(y_pred,y_true,module):
    y_pred = y_pred.unsqueeze(1)
    y_true = y_true.unsqueeze(1)
    # 1) Pixel L1 loss
    loss_reg = nn.functional.l1_loss(y_pred, y_true)

    # 2) gabor loss
    loss_gabor = gabor_loss(y_pred, y_true, module.gabor_kernels)

    # 3) Hessian (per‐channel, via reshape)
    B, C, H, W = y_pred.shape
    # B, H, W = y_pred.shape
    # y_pred = y_pred.unsqueeze(1)  # → (B, 1, H, W)
    # y_true = y_true.unsqueeze(1)  # → (B, 1, H, W)
    dxx_pred = nn.functional.conv2d(y_pred.reshape(-1, 1, H, W), module.dxx_k, padding=1).reshape(B, C, H, W)
    dxx_true = nn.functional.conv2d(y_true.reshape(-1, 1, H, W), module.dxx_k, padding=1).reshape(B, C, H, W)
    dyy_pred = nn.functional.conv2d(y_pred.reshape(-1, 1, H, W), module.dyy_k, padding=1).reshape(B, C, H, W)
    dyy_true = nn.functional.conv2d(y_true.reshape(-1, 1, H, W), module.dyy_k, padding=1).reshape(B, C, H, W)
    dxy_pred = nn.functional.conv2d(y_pred.reshape(-1, 1, H, W), module.dxy_k, padding=1).reshape(B, C, H, W)
    dxy_true = nn.functional.conv2d(y_true.reshape(-1, 1, H, W), module.dxy_k, padding=1).reshape(B, C, H, W)

    loss_hess_per_pixel = (
        (dxx_pred - dxx_true).pow(2)
        + 2 * (dxy_pred - dxy_true).pow(2)
        + (dyy_pred - dyy_true).pow(2)
    )
    loss_hess = loss_hess_per_pixel.mean().sqrt()

    # 4) Polar loss
    if module.center_x is None or module.center_y is None:
        module.center_x = (W - 1) / 2.0
        module.center_y = (H - 1) / 2.0
    
    y_pred_polar = cartesian_to_polar(y_pred, center=(module.center_x, module.center_y))
    y_true_polar = cartesian_to_polar(y_true, center=(module.center_x, module.center_y))
    loss_polar = nn.functional.l1_loss(y_pred_polar, y_true_polar)

    # Combine
    λ_gabor = 0.5
    λ_hess  = 0.2
    λ_pol   = 0.3
    loss = loss_reg + λ_gabor * loss_gabor + λ_hess * loss_hess + λ_pol * loss_polar
    return loss


def dice_loss(pred, target, epsilon=1e-6):
    # Flatten the tensors for global Dice computation
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    intersection = 2.0 * torch.sum(pred_flat * target_flat)
    denominator = torch.sum(pred_flat ** 2) + torch.sum(target_flat ** 2) + epsilon
    loss = 1.0 - (intersection / denominator)
    return loss

def composite_tas_pr_loss(preds, targets, tas_index, pr_index, module, alpha=1, beta=1):
    tas_pred = preds[:, tas_index]
    tas_true = targets[:, tas_index]
    pr_pred = preds[:, pr_index]
    pr_true = targets[:, pr_index]

    tas_loss = l1_gabor_hessian_polar_loss(tas_pred,tas_true,module)#torch.nn.functional.mse_loss(tas_pred, tas_true)
    pr_loss = 0.5*dice_loss(pr_pred, pr_true)+0.5*torch.nn.functional.mse_loss(pr_pred,pr_true)
    #pr_loss = log_cosh_loss(pr_pred,pr_true)

    return (alpha * tas_loss + beta * pr_loss)