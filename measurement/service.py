import math
import numpy as np
from ultralytics import YOLO

CONF = 0.3
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8m-pose.pt")
    return _model


def pt(kps, i):
    return np.array([kps[i, 0], kps[i, 1]]), kps[i, 2]


def dist(kps, i, j):
    p1, c1 = pt(kps, i)
    p2, c2 = pt(kps, j)
    if c1 < CONF or c2 < CONF:
        return 0.0
    return float(np.linalg.norm(p1 - p2))


def get_best_keypoints(img_rgb, min_torso_conf=0.5):
    model = get_model()
    res = model(img_rgb, verbose=False)[0]
    if res.keypoints is None:
        return None, "no person detected"
    all_kps = res.keypoints.data.cpu().numpy()
    if len(all_kps) == 0:
        return None, "no keypoints returned"
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    kps = all_kps[idx]
    torso_ids = [5, 6, 11, 12]
    torso_conf = float(np.mean([kps[i, 2] for i in torso_ids]))
    if torso_conf < min_torso_conf:
        return kps, f"low torso confidence ({torso_conf:.2f})"
    return kps, f"ok (torso conf {torso_conf:.2f})"


def estimate_height_pixels(kps):
    nose_y = kps[0, 1]
    le_y, le_c = kps[1, 1], kps[1, 2]
    re_y, re_c = kps[2, 1], kps[2, 2]
    ls_y, ls_c = kps[5, 1], kps[5, 2]
    rs_y, rs_c = kps[6, 1], kps[6, 2]
    la_y, la_c = kps[15, 1], kps[15, 2]
    ra_y, ra_c = kps[16, 1], kps[16, 2]

    eye_ys = [y for y, c in [(le_y, le_c), (re_y, re_c)] if c > CONF]
    eye_y = np.mean(eye_ys) if eye_ys else nose_y

    sh_ys = [y for y, c in [(ls_y, ls_c), (rs_y, rs_c)] if c > CONF]
    if not sh_ys:
        return 0.0, "shoulders not detected"
    sh_y = np.mean(sh_ys)

    eye_to_sh = sh_y - eye_y
    if eye_to_sh <= 0:
        head_top_y = nose_y - 0.6 * abs(sh_y - nose_y)
    else:
        head_top_y = eye_y - 1.5 * eye_to_sh

    ank_ys = [y for y, c in [(la_y, la_c), (ra_y, ra_c)] if c > CONF]
    if not ank_ys:
        return 0.0, "ankles not detected"
    ankle_y = np.mean(ank_ys)

    span = ankle_y - head_top_y
    if span <= 0:
        return 0.0, "invalid vertical span"
    height_px = span / 0.96
    return float(height_px), "ok"


def check_front_pose(kps):
    issues = []
    ls, lsc = pt(kps, 5)
    rs, rsc = pt(kps, 6)
    lh, lhc = pt(kps, 11)
    rh, rhc = pt(kps, 12)
    if min(lsc, rsc, lhc, rhc) < CONF:
        issues.append("torso keypoints weak")
        return issues
    sh_w = abs(ls[0] - rs[0])
    hip_w = abs(lh[0] - rh[0])
    sh_v = abs(ls[1] - rs[1])
    if sh_w > 0 and (sh_v / sh_w) > 0.25:
        issues.append("shoulders tilted — stand straight")
    torso_h = abs((ls[1] + rs[1]) / 2 - (lh[1] + rh[1]) / 2)
    if torso_h > 0 and sh_w / torso_h < 0.45:
        issues.append("body may not be fully frontal")
    if hip_w <= 0:
        issues.append("hips not detected")
    return issues


def side_depth_px(kps_side, img_side):
    model = get_model()
    res = model(img_side, verbose=False)[0]
    bbox_w = None
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        b = boxes[int(np.argmax(areas))]
        bbox_w = float(b[2] - b[0])
    d_sh = dist(kps_side, 5, 6)
    d_hip = dist(kps_side, 11, 12)
    return bbox_w, d_sh, d_hip


def compute_torso_widths(kps_front):
    ls, lsc = pt(kps_front, 5)
    rs, rsc = pt(kps_front, 6)
    lh, lhc = pt(kps_front, 11)
    rh, rhc = pt(kps_front, 12)
    sh_w = abs(ls[0] - rs[0]) if min(lsc, rsc) > CONF else 0.0
    hip_w = abs(lh[0] - rh[0]) if min(lhc, rhc) > CONF else 0.0
    chest_w = 0.87 * sh_w if sh_w > 0 else 0.0
    if hip_w > 0:
        waist_w = 0.92 * hip_w
    else:
        waist_w = 0.78 * sh_w
    hip_true_w = hip_w / 0.94 if hip_w > 0 else 0.0
    return sh_w, chest_w, waist_w, hip_true_w


def ellipse_circ(width, depth, correction=1.0):
    if width <= 0 or depth <= 0:
        return 0.0
    a, b = width / 2.0, depth / 2.0
    c = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    return c * correction


def arm_length_px(kps):
    def side_len(s, e, w):
        if min(kps[s, 2], kps[e, 2], kps[w, 2]) < CONF:
            return 0.0, 0.0
        upper = np.linalg.norm(pt(kps, s)[0] - pt(kps, e)[0])
        fore = np.linalg.norm(pt(kps, e)[0] - pt(kps, w)[0])
        conf = (kps[s, 2] + kps[e, 2] + kps[w, 2]) / 3.0
        return upper + fore, conf
    l_len, l_conf = side_len(5, 7, 9)
    r_len, r_conf = side_len(6, 8, 10)
    best = l_len if l_conf >= r_conf else r_len
    return best * 1.08 if best > 0 else 0.0


def inseam_px(kps):
    def side(h, k, a):
        if min(kps[h, 2], kps[k, 2], kps[a, 2]) < CONF:
            return 0.0, 0.0
        thigh = np.linalg.norm(pt(kps, h)[0] - pt(kps, k)[0])
        shin = np.linalg.norm(pt(kps, k)[0] - pt(kps, a)[0])
        conf = (kps[h, 2] + kps[k, 2] + kps[a, 2]) / 3.0
        return thigh + shin, conf
    l, lc = side(11, 13, 15)
    r, rc = side(12, 14, 16)
    return l if lc >= rc else r


def pixel_to_cm(px, height_px, user_height_cm):
    if height_px <= 0:
        return 0.0
    return round(float(px) * user_height_cm / height_px, 1)


def sane(v, lo, hi):
    return v if lo <= v <= hi else 0.0


def extract_measurements(front_img, side_img, user_height_cm: float) -> dict:
    """
    Main function. Takes front image (numpy RGB), side image (numpy RGB),
    and user height in cm. Returns measurements dict + diagnostics dict.
    """
    kps_f, status_f = get_best_keypoints(front_img)
    kps_s, status_s = get_best_keypoints(side_img)

    diagnostics = {
        "front_detection": status_f,
        "side_detection": status_s,
    }

    if kps_f is None or kps_s is None:
        return {"error": "YOLO failed to detect a person in one of the images.", "diagnostics": diagnostics}

    issues = check_front_pose(kps_f)
    if issues:
        diagnostics["front_pose_warnings"] = "; ".join(issues)

    height_px, h_status = estimate_height_pixels(kps_f)
    diagnostics["height_calibration"] = f"{h_status}, {height_px:.1f}px = {user_height_cm}cm"

    if height_px <= 0:
        return {"error": "Could not calibrate height. Ensure full body is visible.", "diagnostics": diagnostics}

    sh_w_px, chest_w_px, waist_w_px, hip_w_px = compute_torso_widths(kps_f)
    bbox_depth_px, side_sh_kp, side_hip_kp = side_depth_px(kps_s, side_img)

    if bbox_depth_px and bbox_depth_px > 0:
        chest_d_px = 0.95 * bbox_depth_px
        waist_d_px = 0.82 * bbox_depth_px
        hip_d_px = 0.98 * bbox_depth_px
        diagnostics["depth_source"] = f"side bbox width ({bbox_depth_px:.0f}px)"
    else:
        chest_d_px = 0.75 * chest_w_px
        waist_d_px = 0.80 * waist_w_px
        hip_d_px = 0.78 * hip_w_px
        diagnostics["depth_source"] = "fallback ratio (no side bbox)"

    sh_cm    = pixel_to_cm(sh_w_px,    height_px, user_height_cm)
    chest_w  = pixel_to_cm(chest_w_px, height_px, user_height_cm)
    chest_d  = pixel_to_cm(chest_d_px, height_px, user_height_cm)
    waist_w  = pixel_to_cm(waist_w_px, height_px, user_height_cm)
    waist_d  = pixel_to_cm(waist_d_px, height_px, user_height_cm)
    hip_w    = pixel_to_cm(hip_w_px,   height_px, user_height_cm)
    hip_d    = pixel_to_cm(hip_d_px,   height_px, user_height_cm)
    arm_cm   = pixel_to_cm(arm_length_px(kps_f), height_px, user_height_cm)
    inseam_cm = pixel_to_cm(inseam_px(kps_f),    height_px, user_height_cm)

    chest_circ = sane(round(ellipse_circ(chest_w, chest_d, correction=1.03), 1), 60, 160)
    waist_circ = sane(round(ellipse_circ(waist_w, waist_d, correction=1.00), 1), 50, 150)
    hip_circ   = sane(round(ellipse_circ(hip_w,   hip_d,   correction=1.02), 1), 60, 170)

    measurements = {
        "shoulder_width_cm":      sh_cm,
        "chest_circumference_cm": chest_circ,
        "waist_circumference_cm": waist_circ,
        "hip_circumference_cm":   hip_circ,
        "arm_length_cm":          arm_cm,
        "inseam_cm":              inseam_cm,
    }

    diagnostics["widths_cm"] = f"sh={sh_cm}, chest={chest_w}, waist={waist_w}, hip={hip_w}"
    diagnostics["depths_cm"] = f"chest={chest_d}, waist={waist_d}, hip={hip_d}"

    return {"measurements": measurements, "diagnostics": diagnostics}