import cv2
import numpy as np
import torch
import torch.nn.functional as F


def ERP2Persp(img, FOV, THETA, PHI, height, width, mode='bilinear'):

    img_h, img_w = img.shape[:2]
    FOV = np.radians(FOV)
    THETA = np.radians(THETA)
    PHI = np.radians(PHI)
    x = np.linspace(-np.tan(FOV / 2), np.tan(FOV / 2), width)
    y = np.linspace(-np.tan(FOV / 2), np.tan(FOV / 2), height)
    xv, yv = np.meshgrid(x, -y)

    zv = np.ones_like(xv)
    vec = np.stack([xv, yv, zv], axis=-1)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    vec = vec / norm

    def rot_matrix_y(theta):
        return np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])

    def rot_matrix_x(phi):
        return np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi),  np.cos(phi)],
        ])

    rot = rot_matrix_y(THETA) @ rot_matrix_x(PHI)
    vec = vec @ rot.T  # apply rotation

    lon = np.arctan2(vec[..., 0], vec[..., 2])  # theta
    lat = np.arcsin(vec[..., 1])               # phi

    # Normalize lon/lat to pixel coordinates
    u = (lon + np.pi) / (2 * np.pi) * img_w
    v = (np.pi / 2 - lat) / np.pi * img_h

    # Interpolate
    if mode == 'nearest':
        u = u.round().astype(int)
        v = v.round().astype(int)
        u = np.clip(u, 0, img_w - 1)
        v = np.clip(v, 0, img_h - 1)
        persp = img[v, u]
    else:
        # bilinear
        persp = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    return persp

def rot_matrix_y(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [ 0,            1, 0           ],
                     [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)

def rot_matrix_x(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi),  np.cos(phi)]], dtype=np.float32)

def _global_pct_norm(x, low, high, eps):
    flat = x.reshape(-1)
    lo_q = torch.quantile(flat, q=low)
    hi_q = torch.quantile(flat, q=high)
    return (x.clamp(min=lo_q, max=hi_q) - lo_q) / (hi_q - lo_q + eps)


def _build_midband_gate(x, low_q, high_q, floor, eps):
    flat = x.reshape(-1)
    low_v = torch.quantile(flat, q=low_q)
    high_v = torch.quantile(flat, q=high_q)
    center = 0.5 * (low_v + high_v)
    radius = 0.5 * (high_v - low_v)

    if radius <= eps:
        return torch.ones_like(x)

    gate = 1.0 - torch.abs(x - center) / (radius + eps)
    gate = gate.clamp(min=0.0, max=1.0)
    return floor + (1.0 - floor) * gate

@torch.no_grad()
def build_selfview_confidence(
        self_attn_per_layer,
        toks_per_img_total=1374,
        num_special_tokens=5,
        patch_grid_hw=(37, 37),
        img_hw=(518, 518),
        use_logits: bool = False,
        head_reduce: str = "median",
        layer_weights=None,
        percentile_clip=(0.05, 0.95),
        eps: float = 1e-8,
):
    if isinstance(self_attn_per_layer, dict):
        masked_data = self_attn_per_layer["masked"]
        unmasked_data = self_attn_per_layer.get("unmasked", masked_data)
        layers = masked_data if isinstance(masked_data, list) else [masked_data]
        sharp_layers = unmasked_data if isinstance(unmasked_data, list) else [unmasked_data]
    elif isinstance(self_attn_per_layer, torch.Tensor):
        layers = [self_attn_per_layer]
        sharp_layers = layers
    else:
        layers = self_attn_per_layer
        sharp_layers = layers

    I, Hh, N, _ = layers[0].shape
    Ttot = toks_per_img_total
    S = num_special_tokens
    Hp, Wp = patch_grid_hw
    Tpatch = Hp * Wp
    assert N == Ttot
    L = len(layers)

    device = layers[0].device

    if layer_weights is None:
        layer_weights = torch.ones(L, dtype=torch.float32, device=device) / max(L, 1)
    else:
        layer_weights = torch.tensor(layer_weights, dtype=torch.float32, device=device)
        layer_weights = layer_weights / (layer_weights.sum() + eps)

    patch_slice = slice(S, Ttot)
    sharp_all = torch.zeros(I, Tpatch, device=device)
    recv_all = torch.zeros(I, Tpatch, device=device)
    sym_all = torch.zeros(I, Tpatch, device=device)

    def _to_prob(blk):
        if use_logits:
            blk = blk - blk.max(dim=-1, keepdim=True).values
            return blk.softmax(dim=-1)
        blk = torch.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
        return blk / (blk.sum(dim=-1, keepdim=True) + eps)

    for k in range(L):
        A_blk = _to_prob(layers[k][:, :, patch_slice, patch_slice].float())
        A_unmasked = sharp_layers[k][:, :, patch_slice, patch_slice].float()
        A_unmasked = torch.nan_to_num(A_unmasked, nan=0.0, posinf=0.0, neginf=0.0)
        A_unmasked = A_unmasked / (A_unmasked.sum(dim=-1, keepdim=True) + eps)
        sharp_h = (A_unmasked * A_blk).clamp(min=0.0).sqrt().sum(dim=-1)

        recv_h = A_blk.sum(dim=-2)
        col_sum = A_blk.sum(dim=-2, keepdim=False).unsqueeze(-1)
        A_prime = A_blk.transpose(-2, -1) / (col_sum + eps)
        sym_h = (A_blk * A_prime).clamp(min=0.0).sqrt().sum(dim=-1)

        if head_reduce == "median":
            sharp_k = sharp_h.median(dim=1).values
            recv_k = recv_h.median(dim=1).values
            sym_k = sym_h.median(dim=1).values
        else:
            sharp_k = sharp_h.mean(dim=1)
            recv_k = recv_h.mean(dim=1)
            sym_k = sym_h.mean(dim=1)

        w = layer_weights[k]
        sharp_all += w * sharp_k
        recv_all += w * recv_k
        sym_all += w * sym_k

    low, high = percentile_clip

    sharp_norm = _global_pct_norm(sharp_all, low=low, high=high, eps=eps)
    recv_norm = _global_pct_norm(recv_all, low=low, high=high, eps=eps)
    sym_norm = _global_pct_norm(sym_all, low=low, high=high, eps=eps)

    recv_gate_map = _build_midband_gate(recv_norm, 0, 0.6, 0.5, eps)
    recv_for_combine = recv_gate_map * recv_norm

    score_patch = sym_norm.clamp(min=eps) * (1.0 + recv_for_combine) + 0.5 * sharp_norm
    score_patch = _global_pct_norm(score_patch, low=low, high=high, eps=eps)
    score_floor = 0.05
    score_patch = score_floor + (1.0 - score_floor) * score_patch

    S_pix_list = []
    for i in range(I):
        S_grid = score_patch[i].reshape(1, 1, Hp, Wp)
        S_pix = F.interpolate(S_grid, size=img_hw, mode='bilinear', align_corners=False)
        S_pix_list.append(S_pix)
    S_pix = torch.cat(S_pix_list, dim=0)
    return S_pix


def depth_set_to_equirect_attention(
    depths,
    view_params,
    attention_maps=None,
    H=1024, W=2048,
    interp='nearest',
    min_weight=0.0,
):
    assert len(depths) == len(view_params)
    lon = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
    lat = np.linspace( np.pi/2, -np.pi/2, H, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x = np.sin(lon_grid) * np.cos(lat_grid)
    y = np.sin(lat_grid)
    z = np.cos(lon_grid) * np.cos(lat_grid)
    d_w = np.stack([x, y, z], axis=-1).astype(np.float32)
    d_w /= (np.linalg.norm(d_w, axis=-1, keepdims=True) + 1e-8)

    interp_flag = cv2.INTER_NEAREST if interp == 'nearest' else cv2.INTER_LINEAR
    coverage = np.zeros((H, W), dtype=np.int32)
    erp_num = np.zeros((H, W), dtype=np.float32)
    erp_den = np.zeros((H, W), dtype=np.float32)
    erp_wsum = np.zeros((H, W), dtype=np.float32)

    weights= build_selfview_confidence(attention_maps)[:, 0, :, :].cpu()

    for i, (depth, (yaw_deg, pitch_deg, fov_deg), w_i) in enumerate(zip(depths, view_params, weights)):
        depth = np.asarray(depth, dtype=np.float32)
        Hc, Wc = depth.shape
        yaw   = np.deg2rad(np.float32(yaw_deg))
        pitch = np.deg2rad(np.float32(pitch_deg))
        fov   = np.deg2rad(np.float32(fov_deg))
        t_src = np.tan(fov * 0.5).astype(np.float32)
        R = rot_matrix_y(yaw) @ rot_matrix_x(pitch)
        d_c = d_w @ R
        zc = d_c[..., 2]
        xc = d_c[..., 0] / (zc + 1e-8)
        yc = d_c[..., 1] / (zc + 1e-8)
        vis = (zc > 0) & (np.abs(xc) <= t_src) & (np.abs(yc) <= t_src)
        if not np.any(vis):
            continue

        u = ((xc / t_src + 1.0) * 0.5) * (Wc - 1)
        v = ((-yc / t_src + 1.0) * 0.5) * (Hc - 1)
        mapx = u.astype(np.float32)
        mapy = v.astype(np.float32)

        sampled_depth = cv2.remap(depth, mapx, mapy,
                                  interpolation=interp_flag,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0.0).astype(np.float32)
        valid_depth = (sampled_depth > 0) & vis

        if w_i is None:
            sampled_w = np.ones_like(sampled_depth, dtype=np.float32)
        else:
            w_i = np.asarray(w_i, dtype=np.float32)
            if w_i.ndim == 3 and w_i.shape[0] == 1:
                w_i = w_i[0]
            assert w_i.shape == (Hc, Wc)
            sampled_w = cv2.remap(w_i, mapx, mapy,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0.0).astype(np.float32)

        sampled_w = np.where(valid_depth, sampled_w, 0.0)
        if min_weight > 0:
            sampled_w = np.where(sampled_w >= min_weight, sampled_w, 0.0)
        coverage += (sampled_w > 0).astype(np.int32)
        erp_num += sampled_w * sampled_depth
        erp_den += sampled_w
        erp_wsum += sampled_w

    erp_depth = np.zeros((H, W), dtype=np.float32)
    nonzero = erp_den > 0
    erp_depth[nonzero] = erp_num[nonzero] / (erp_den[nonzero] + 1e-12)

    return erp_depth




