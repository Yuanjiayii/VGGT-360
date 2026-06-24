import cv2, numpy as np


def SA_confidence(rgb_equi, tau=0.5, margin=0.05):
    """
    structure aware confidence map generation
    """
    g = cv2.cvtColor(rgb_equi, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = cv2.GaussianBlur(mag, (0,0), 1.0)
    p95 = np.percentile(mag, 80)
    mag_filtered = np.where(mag >= p95, mag, 0)

    p1, p99 = np.percentile(mag_filtered, 1), np.percentile(mag_filtered, 99)
    T = np.clip((mag_filtered - p1) / (p99 - p1 + 1e-6), 0, 1)

    t0 = float(np.median(T))
    M = 1.0 / (1.0 + np.exp(-(T - t0) / max(tau, 1e-6)))
    mask_rgb = (rgb_equi.sum(axis=-1) > 0).astype(np.float32)

    Wc = edge_relaxation_window(H=rgb_equi.shape[0], W=rgb_equi.shape[1], margin=margin)
    M = np.clip(M ** Wc, 0.10, 1.0)
    return M, mask_rgb


def edge_relaxation_window(H, W, margin=0.04):

    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]
    d = np.maximum(np.abs(xx), np.abs(yy))
    t = (d - (1 - margin)) / max(margin, 1e-6)
    t = np.clip(t, 0.0, 1.0)

    W_center = 0.5 * (1 + np.cos(t * np.pi))
    return W_center.astype(np.float32)

