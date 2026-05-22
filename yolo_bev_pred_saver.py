# import os
# import cv2
# import numpy as np
# from pathlib import Path
# from ultralytics import YOLO

# # ---------------- CONFIG ----------------
# MODEL_PATH     = "best_3h_full.pt"                # path to YOLOv12-OBB weights
# VAL_IMAGES_DIR = "ds-yolo-3obj-3h2/images/val"    # validation images folder
# SAVE_TXT_DIR   = "pedestrian_detect_bev/txt/"  # folder to save prediction txt files
# SAVE_IMG_DIR   = "pedestrian_detect_bev/images/"          # folder to save overlay images
# CONF_THRES     = 0.01                           # confidence threshold
# IOU_THRES      = 0.5                              # NMS IoU threshold
# DEVICE         = "0"                              # e.g. "0", "cpu"
# WORKERS        = 32
# # -----------------------------------------

# # Class config (id -> name) — adjust order if your dataset differs
# CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
# ABBR = {"Car": "Car", "Pedestrian": "Ped", "Cyclist": "Cyc"}

# # Colors in BGR (OpenCV)
# COLORS = {
#     "Car":        (255, 255, 0),   # cyan
#     "Pedestrian": (255, 0, 255),   # purple
#     "Cyclist":    (0, 255, 255),   # yellow
# }

# # Drawing params
# BOX_THICK   = 1
# FONT        = cv2.FONT_HERSHEY_SIMPLEX
# FONT_SCALE  = 0.35
# FONT_THICK  = 1
# TEXT_Y_OFF  = 2  # pixels above top edge

# os.makedirs(SAVE_TXT_DIR, exist_ok=True)
# os.makedirs(SAVE_IMG_DIR, exist_ok=True)

# # Load model
# model = YOLO(MODEL_PATH)

# # Run inference (we draw/save ourselves, so save=False here)
# results = model.predict(
#     source=VAL_IMAGES_DIR,
#     device=DEVICE,
#     conf=CONF_THRES,
#     iou=IOU_THRES,
#     save=False,
#     save_txt=False,
#     verbose=True,
#     workers=WORKERS,
# )

# def draw_obb(img, pts, color, label):
#     """
#     pts: numpy array shape (8,) or (4,2) in pixel coords
#     color: BGR tuple
#     label: small text to put near top edge
#     """
#     pts = pts.reshape(-1, 2).astype(int)
#     # draw polygon
#     cv2.polylines(img, [pts], isClosed=True, color=color, thickness=BOX_THICK, lineType=cv2.LINE_AA)
#     # place tiny label near the top-most vertex
#     top_idx = np.argmin(pts[:, 1])
#     tx, ty = int(pts[top_idx, 0]), int(max(0, pts[top_idx, 1] - TEXT_Y_OFF))
#     cv2.putText(img, label, (tx, ty), FONT, FONT_SCALE, color, FONT_THICK, cv2.LINE_AA)

# def normalize_xyxyxyxy(xyxyxyxy, w, h):
#     """Return list of 8 normalized floats [x1/w, y1/h, ..., x4/w, y4/h]."""
#     out = []
#     for i, v in enumerate(xyxyxyxy):
#         out.append(float(v) / (w if i % 2 == 0 else h))
#     return out

# # Iterate results, save TXT and overlay image
# for r in results:
#     image_path = r.path
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"[warn] Failed to read image: {image_path}")
#         continue
#     h, w = img.shape[:2]

#     base = Path(image_path).stem
#     ext  = Path(image_path).suffix

#     # Write prediction file
#     txt_path = Path(SAVE_TXT_DIR) / f"{base}.txt"
#     ped_count = 0

#     with open(txt_path, "w") as f:
#         if r.obb is not None and len(r.obb):
#             for box in r.obb:
#                 cls_id = int(box.cls[0])
#                 conf   = float(box.conf[0])

#                 # class name + abbrev + color
#                 try:
#                     cls_name = CLASS_NAMES[cls_id]
#                 except IndexError:
#                     # unknown class id, skip
#                     continue
#                 label = ABBR.get(cls_name, str(cls_id))
#                 color = COLORS.get(cls_name, (200, 200, 200))

#                 # xyxyxyxy (pixel)
#                 xyxyxyxy = box.xyxyxyxy[0].cpu().numpy().flatten()
#                 # draw on image
#                 draw_obb(img, xyxyxyxy, color, label)
#                 if cls_name == "Pedestrian":
#                     ped_count += 1

#                 # save normalized corners to TXT
#                 norm = normalize_xyxyxyxy(xyxyxyxy, w, h)
#                 coords_str = " ".join(f"{v:.6f}" for v in norm)
#                 f.write(f"{cls_id} {conf:.6f} {coords_str}\n")
#         else:
#             # create empty file for consistency
#             pass

#     # Save overlay image with pedestrian count appended
#     out_name = f"{base}_ped{ped_count}{ext}"
#     out_path = Path(SAVE_IMG_DIR) / out_name
#     cv2.imwrite(str(out_path), img)

# print(f"TXT predictions saved to: {SAVE_TXT_DIR}")
# print(f"Overlay images saved to:  {SAVE_IMG_DIR}")



import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH     = "best_newset_full-s.pt"                 # path to YOLOv12-OBB weights
VAL_IMAGES_DIR = "ds-yolo-3obj-3h2-lowZwideJitterAug-amp-newSET/images/val"     # validation images folder
SAVE_TXT_DIR   = "pedestrian_detect_bev_new/txt/"      # folder to save prediction txt files
SAVE_IMG_DIR   = "pedestrian_detect_bev_new/images/"   # folder to save overlay images
CONF_THRES     = 0.01                              # confidence threshold
IOU_THRES      = 0.5                               # NMS IoU threshold
DEVICE         = "0"                               # e.g. "0", "cpu"
WORKERS        = 32
# -----------------------------------------

# Class config (id -> name) — adjust order if your dataset differs
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
ABBR = {"Car": "Car", "Pedestrian": "Ped", "Cyclist": "Cyc"}

# Colors in BGR (OpenCV)
COLORS = {
    "Car":        (255, 255, 0),   # cyan
    "Pedestrian": (255,   0, 255), # purple
    "Cyclist":    (  0, 255, 255), # yellow
}

# Drawing params
BOX_THICK   = 1
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.35
FONT_THICK  = 1
TEXT_Y_OFF  = 2  # pixels above top edge

os.makedirs(SAVE_TXT_DIR, exist_ok=True)
os.makedirs(SAVE_IMG_DIR, exist_ok=True)

# ---------------- Utils ----------------
def draw_obb(img, pts, color, label):
    """
    pts: numpy array shape (8,) or (4,2) in pixel coords (ORIGINAL image frame)
    color: BGR tuple
    label: small text to put near the top-most vertex
    """
    pts = pts.reshape(-1, 2).astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=BOX_THICK, lineType=cv2.LINE_AA)
    top_idx = np.argmin(pts[:, 1])
    tx, ty = int(pts[top_idx, 0]), int(max(0, pts[top_idx, 1] - TEXT_Y_OFF))
    cv2.putText(img, label, (tx, ty), FONT, FONT_SCALE, color, FONT_THICK, cv2.LINE_AA)

def normalize_xyxyxyxy(xyxyxyxy, w, h):
    """Return list of 8 normalized floats [x1/w, y1/h, ..., x4/w, y4/h] in the ORIGINAL image frame."""
    out = []
    for i, v in enumerate(xyxyxyxy):
        out.append(float(v) / (w if i % 2 == 0 else h))
    return out

def estimate_infer_shape_and_unwarp(all_polys, W0, H0):
    """
    Heuristic Plan-B unwarper:
    - Ultralytics may run rectangular inference (e.g., 800x704). Predicted vertices can be in that canvas.
    - We estimate the inference canvas (Winfer, Hinfer) from the max predicted coordinates.
    - If Winfer>W0 or Hinfer>H0, we scale back to the original (sx=W0/Winfer, sy=H0/Hinfer).
    Returns: list of polygons mapped to original frame, and (sx, sy, Winfer, Hinfer)
    """
    if len(all_polys) == 0:
        return all_polys, (1.0, 1.0, W0, H0)

    # Stack all points to estimate bounds
    stacked = np.vstack([p.reshape(-1, 2) for p in all_polys]).astype(np.float32)
    max_x = float(stacked[:, 0].max()) if stacked.size else 0.0
    max_y = float(stacked[:, 1].max()) if stacked.size else 0.0

    # If preds already sit within original frame (allow tiny eps), skip
    eps = 1.001
    if max_x <= W0 * eps and max_y <= H0 * eps:
        return all_polys, (1.0, 1.0, W0, H0)

    # Otherwise, assume these maxima reflect the inference canvas edge
    Winfer = max(max_x, 1.0)
    Hinfer = max(max_y, 1.0)
    sx = W0 / Winfer
    sy = H0 / Hinfer

    unwarped = []
    for p in all_polys:
        q = p.reshape(-1, 2).astype(np.float32)
        q[:, 0] *= sx
        q[:, 1] *= sy
        unwarped.append(q.reshape(-1))
    return unwarped, (sx, sy, Winfer, Hinfer)

# ---------------- Inference ----------------
model = YOLO(MODEL_PATH)

# We draw/save ourselves, so keep save=False here
results = model.predict(
    source=VAL_IMAGES_DIR,
    device=DEVICE,
    conf=CONF_THRES,
    iou=IOU_THRES,
    save=False,
    save_txt=False,
    verbose=True,
    workers=WORKERS,
)

# ---------------- Iterate results, save TXT + overlay ----------------
for r in results:
    image_path = r.path
    img = cv2.imread(image_path)
    if img is None:
        print(f"[warn] Failed to read image: {image_path}")
        continue

    H0, W0 = img.shape[:2]
    base = Path(image_path).stem
    ext  = Path(image_path).suffix

    txt_path = Path(SAVE_TXT_DIR) / f"{base}.txt"
    ped_count = 0

    # Collect all polygons first (as they come from the predictor)
    raw_polys = []
    meta_for_poly = []  # (cls_id, conf, cls_name)

    if r.obb is not None and len(r.obb):
        for box in r.obb:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            # Map class id -> name
            try:
                cls_name = CLASS_NAMES[cls_id]
            except IndexError:
                continue

            xyxyxyxy = box.xyxyxyxy[0].detach().cpu().numpy().astype(np.float32).flatten()
            raw_polys.append(xyxyxyxy)
            meta_for_poly.append((cls_id, conf, cls_name))

    # Unwarp to original frame if needed (Plan B)
    unwarped_polys, (sx, sy, Winfer, Hinfer) = estimate_infer_shape_and_unwarp(raw_polys, W0, H0)

    # Now draw and save normalized coords (in ORIGINAL frame)
    with open(txt_path, "w") as f:
        for poly, (cls_id, conf, cls_name) in zip(unwarped_polys, meta_for_poly):
            label = ABBR.get(cls_name, str(cls_id))
            color = COLORS.get(cls_name, (200, 200, 200))

            # Draw on original image
            draw_obb(img, poly, color, label)
            if cls_name == "Pedestrian":
                ped_count += 1

            # Save normalized corners (original frame)
            norm = normalize_xyxyxyxy(poly, W0, H0)
            coords_str = " ".join(f"{v:.6f}" for v in norm)
            f.write(f"{cls_id} {conf:.6f} {coords_str}\n")

    # Save overlay image with pedestrian count appended
    out_name = f"{base}_ped{ped_count}{ext}"
    out_path = Path(SAVE_IMG_DIR) / out_name
    cv2.imwrite(str(out_path), img)

print(f"TXT predictions saved to: {SAVE_TXT_DIR}")
print(f"Overlay images saved to:  {SAVE_IMG_DIR}")
