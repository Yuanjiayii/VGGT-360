import cv2, numpy as np

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
    vec = vec @ rot.T

    lon = np.arctan2(vec[..., 0], vec[..., 2])
    lat = np.arcsin(vec[..., 1])

    u = (lon + np.pi) / (2 * np.pi) * img_w
    v = (np.pi / 2 - lat) / np.pi * img_h

    if mode == 'nearest':
        u = u.round().astype(int)
        v = v.round().astype(int)
        u = np.clip(u, 0, img_w - 1)
        v = np.clip(v, 0, img_h - 1)
        persp = img[v, u]
    else:
        persp = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    return persp

def perspective_confidence_map(rgb_equi, tau=0.5):
    g = cv2.cvtColor(rgb_equi, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.0)


    p1, p99 = np.percentile(mag, 1), np.percentile(mag, 99)
    T = np.clip((mag - p1) / (p99 - p1 + 1e-6), 0.01, 0.99)

    t0 = float(np.median(T))
    conf_map = 1.0 / (1.0 + np.exp(-(T - t0) / max(tau, 1e-6)))

    valid_mask = (rgb_equi.sum(axis=2) > 0)

    conf_map[~valid_mask] = 0.0

    return conf_map, valid_mask



def confidence_score_from_map(conf_map, valid_mask,
                              trim_ratio=0.1,
                              min_valid_ratio=0.1):
    H, W = conf_map.shape
    valid = valid_mask & np.isfinite(conf_map)
    valid_ratio = valid.sum() / float(H * W + 1e-6)
    if valid_ratio < min_valid_ratio or valid.sum() == 0:
        return 1.0

    vals = conf_map[valid].ravel()
    vals.sort()
    n = len(vals)
    a = int(n * trim_ratio)
    b = int(n * (1.0 - trim_ratio))
    if b <= a:
        return float(vals.mean())
    return float(vals[a:b].mean())


def pick_two_least_confident(base_rgbs):
    scores = []
    conf_maps = []
    for rgb in base_rgbs:
        conf_map, valid_mask = perspective_confidence_map(rgb)
        s = confidence_score_from_map(conf_map, valid_mask)
        conf_maps.append(conf_map)
        scores.append(s)
    order = np.argsort(np.asarray(scores))
    top2 = order[:2].tolist()
    return top2, scores, conf_maps

def augment_for_top2(base, top2, max_total=12,
                     neighbor_fov=85.0,
                     dyaw=12.0, dpitch=12.0,
                     min_sep_all=10.0):
    def yaw_pitch_to_vec(y, p):
        y_, p_ = np.deg2rad(y), np.deg2rad(p)
        return np.array([np.cos(p_)*np.cos(y_), np.sin(p_), np.cos(p_)*np.sin(y_)], dtype=np.float64)
    def ang_deg(v1, v2):
        v1 = v1/ (np.linalg.norm(v1)+1e-9); v2 = v2/ (np.linalg.norm(v2)+1e-9)
        return np.degrees(np.arccos(np.clip(np.dot(v1,v2), -1.0, 1.0)))
    def norm_yaw(y): return y % 360.0
    def clip_pitch(p): return float(np.clip(p, -85.0, 85.0))
    def min_ang_to_views(yaw, pitch, views):
        v = yaw_pitch_to_vec(yaw, pitch)
        return min(ang_deg(v, yaw_pitch_to_vec(vy, vp)) for (vy, vp, _) in views)

    views = base.copy()
    extra = max_total - len(views)
    if extra <= 0:
        return views

    for idx in top2:
        if extra <= 0: break
        y0, p0, _ = base[idx]

        if abs(p0) < 80.0:
            cands = [(y0 - dyaw, p0 + dpitch),
                     (y0 + dyaw, p0 - dpitch)
                     ]
        else:
            pole_dp = 20.0
            tgt_p = clip_pitch(p0 - np.sign(p0)*pole_dp)
            cands = [(y0, tgt_p),
                     (norm_yaw(y0 + 90.0), tgt_p)
                     ]


        for (cy, cp) in cands:
            if extra <= 0: break
            cy = norm_yaw(cy); cp = clip_pitch(cp)
            if min_ang_to_views(cy, cp, views) >= min_sep_all:
                views.append((cy, cp, neighbor_fov))
                extra -= 1
    return views

def view_generation(pano_img, FOV):

    view_params = [(0.0, 0.0, FOV), (60.0, 0.0, FOV), (120.0, 0.0, FOV), (180.0, 0.0, FOV), (240.0, 0.0, FOV),
                   (300.0, 0.0, FOV), (0.0, 90.0, FOV), (0.0, -90.0, FOV)]

    persp_imgs = []
    for i, (yaw, pitch, fov) in enumerate(view_params):
        persp_img = ERP2Persp(pano_img, FOV=fov, THETA=yaw, PHI=pitch, height=512, width=512, mode='bilinear')
        persp_imgs.append(persp_img)

    top2, per_scores, conf_maps = pick_two_least_confident(persp_imgs)

    views_final = augment_for_top2(view_params, top2, max_total=12,
                                   neighbor_fov=90.0,
                                   dyaw=40.0, dpitch=40.0,
                                   min_sep_all=10.0)
    return views_final


