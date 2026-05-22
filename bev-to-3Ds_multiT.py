#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import concurrent.futures
import threading
import math

# ---------------- BEV field of view (meters) ----------------
x_min, x_max = 0.0, 70.0     # forward range (Velodyne X)
y_min, y_max = -40.0, 40.0   # lateral range (Velodyne Y)
res = 0.10  # your BEV cell size (m), change if different
W = int(round((x_max - x_min) / res))  # e.g., 700
H = int(round((y_max - y_min) / res))  # e.g., 800

# ----------------- Calibration -----------------
def load_calib(calib_path):
    data = {}
    with open(calib_path, "r") as f:
        for l in f:
            if ":" not in l:
                continue
            k, v = l.split(":", 1)
            data[k.strip()] = np.fromstring(v, sep=" ")
    P2 = data["P2"].reshape(3, 4)
    Tr = data["Tr_velo_to_cam"].reshape(3, 4)
    R0 = data["R0_rect"].reshape(3, 3)
    # Compose rectified velodyne->cam transform
    T = np.vstack([Tr, [0, 0, 0, 1]])
    R_ext = np.eye(4); R_ext[:3, :3] = R0
    velo2cam = R_ext @ T
    R_velo2cam = velo2cam[:3, :3]
    t_velo2cam = velo2cam[:3, 3]
    return P2, velo2cam, R_velo2cam, t_velo2cam

def project_cam_to_img(P, pts3d):
    """pts3d: (N,3) in camera coords -> (N,2) image pixels"""
    pts_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    uvw = (P @ pts_h.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    return uv


def filter_noise_iqr(data):
    """
    Filters out noise from a 1D NumPy array using the Interquartile Range (IQR) method.

    Args:
        data (np.array): A 1D NumPy array of numerical values.

    Returns:
        np.array: The filtered array with outliers removed.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1, q3 = np.percentile(data, [25, 75])

    # Calculate the Interquartile Range
    iqr = q3 - q1

    # Define the upper and lower bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    

    # Filter the data and return the filtered array
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return data
    return filtered_data



# --------- Geometry helpers on BEV (Velodyne XY) ----------
# def bev_norm_to_metric(uv):
#     """(N,2) normalized in [0,1] -> meters in Velodyne (x,y)."""
#     u = np.clip(uv[:, 0], 0.0, 1.0)
#     v = np.clip(uv[:, 1], 0.0, 1.0)
#     x = x_min + u * (x_max - x_min)
#     y = y_min + v * (y_max - y_min)
#     return np.stack([x, y], axis=1)

def bev_norm_to_metric(uv, center=True, y_flip=False):
    """
    (N,2) normalized coords from the BEV image -> meters in Velodyne (x,y).
    center=True  : interpret (u,v) as *pixel centers*; adds +0.5 cell offset.
    y_flip=False : set True if your BEV encoder used row 0 at y_max (common).
    """
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    

    # pixel coords (float)
    # note: multiply by image size (W,H), not (W-1,H-1), because we then add +0.5
    px = u * W
    py = v * H

    if center:
        px = px
        py = py 

    # optional flip: row 0 (top) corresponds to y_max, not y_min
    if y_flip:
        py = (H - 1) - py

    # meters (Velodyne)
    x = x_min + px * res
    y = y_min + py * res
    return np.stack([x, y], axis=1)

def order_corners_ccw(pts):
    """Order 4 points counter-clockwise around centroid."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(ang)
    return pts[idx]

def polygon_scale(pts, scale=1.5):
    """Scale polygon around its centroid by factor."""
    c = pts.mean(axis=0, keepdims=True)
    return (pts - c) * scale + c

def edges_and_axes(pts4):
    """Return edge vectors and (length,width) with main axis vector."""
    e = np.array([pts4[(i+1) % 4] - pts4[i] for i in range(4)])  # 4 edges
    lens = np.linalg.norm(e, axis=1)
    k = int(np.argmax(lens))   # index of long edge
    v_long = e[k] / (lens[k] + 1e-9)
    # length = mean of two opposite long edges; width = mean of the other two
    length = 0.5 * (lens[k] + lens[(k+2) % 4])
    width  = 0.5 * (lens[(k+1) % 4] + lens[(k+3) % 4])
    return v_long, float(length), float(width)

def points_in_polygon(points_xy, poly_xy):
    """Ray casting test, vectorized over points. points_xy: (N,2), poly_xy: (M,2)"""
    x = points_xy[:, 0]; y = points_xy[:, 1]
    px = poly_xy[:, 0];  py = poly_xy[:, 1]
    n = len(poly_xy)
    inside = np.zeros(points_xy.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        # edges that straddle the horizontal ray
        inter = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= inter
        j = i
    return inside

# -------------- Build 3D box in camera coords --------------
def compute_box3d_cam(h, w, l, tx, ty, tz, ry):
    """Return (8,3) box corners in camera coords (KITTI convention, ty is bottom)."""
    # object frame corners
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_c = np.array([   0,    0,    0,    0,   -h,   -h,   -h,   -h])
    z_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])

    R = np.array([[ np.cos(ry), 0, np.sin(ry)],
                  [          0, 1,          0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    corners = (R @ np.vstack([x_c, y_c, z_c])).T
    corners[:, 0] += tx
    corners[:, 1] += ty
    corners[:, 2] += tz
    return corners  # (8,3)

# -------------- Frame conversion --------------
def convert_frame(pred_txt, velo_bin, calib_txt, out_txt,
                  default_heights={"Car":1.6, "Pedestrian":1.7, "Cyclist":1.6},
                  cls_map={"0":"Car","1":"Pedestrian","2":"Cyclist"}):
    P2, velo2cam, R_velo2cam, t_velo2cam = load_calib(calib_txt)
    # Load Velodyne points (X,Y,Z,reflectance)
    pts = np.fromfile(velo_bin, dtype=np.float32).reshape(-1, 4)
    pts_xy = pts[:, :2]
    pts_z  = pts[:, 2]

    lines_out = []

    with open(pred_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 10:  # cls conf + 8 coords (x1 y1 ... x4 y4)
                continue
            cls = parts[0]
            conf = float(parts[1])
            if cls in cls_map:
                kitti_cls = cls_map[cls]
            else:
                # allow already-named classes
                kitti_cls = cls.capitalize()
                if kitti_cls not in ("Car", "Pedestrian", "Cyclist"):
                    continue

            coords = np.array(list(map(float, parts[2:])), dtype=np.float64).reshape(4, 2)
            # normalized → meters (Velodyne XY)
            quad_xy = bev_norm_to_metric(coords)
            # order CCW to stabilize edges
            quad_xy = order_corners_ccw(quad_xy)
            quad_xy_orig = quad_xy.copy()
            # scale +0.5% for zmin region
            center_x, center_y = quad_xy.mean(axis=0)
            dist2ref = math.sqrt(center_x**2 + center_y**2 )
            dist_rel = dist2ref / 80.0
            exp_ratio_prime = 1.0 + dist_rel*2.5
            if (dist_rel<0.4 or kitti_cls=='Pedestrian'):
                exp_ratio_prime = 1.0
            exp_ration_second = 1.05
            
            quad_xy = polygon_scale(quad_xy, exp_ratio_prime)
            quad_xy_exp = polygon_scale(quad_xy, exp_ration_second)

            # pick points inside polygons
            mask_ori_forB = points_in_polygon(pts_xy, quad_xy)
            mask_ori_forT = points_in_polygon(pts_xy, quad_xy_orig)
            mask_exp = points_in_polygon(pts_xy, quad_xy_exp)
            if np.count_nonzero(mask_ori_forB) < 5:
                #print('less points: ',np.count_nonzero(mask_ori), velo_bin)
                quad_xy_exp = polygon_scale(quad_xy, 1.25)
                mask_exp = points_in_polygon(pts_xy, quad_xy_exp)
            if not mask_ori_forB.any() and not mask_exp.any():
                # no points at all → fallback
                cx, cy = quad_xy.mean(axis=0)
                z_bottom = -1.73
                z_top    = z_bottom + default_heights.get(kitti_cls)
            else:                
                #z_bottom = float(np.min(pts_z[mask_exp])) if mask_exp.any() else float(np.min(pts_z[mask_ori]))
                if (mask_ori_forB.any()):
                    #z_bottom = float(np.min(pts_z[mask_ori]))
                    # Apply the mask to get only the relevant z dimensions
                    masked_pts_z = pts_z[mask_ori_forB]
                    # Get the indices that would sort the array in ascending order
                    sorted_indices = np.argsort(masked_pts_z)
                    # Select the indices of the 5 smallest values
                    top_10_indices = sorted_indices[:10]
                    # Use these indices to get the 5 smallest values
                    ten_min_z = masked_pts_z[top_10_indices]
                    # Print the 5 lowest z dimensions
                    filtered_min_z = filter_noise_iqr(ten_min_z)
                    z_bottom = float(np.min(filtered_min_z))
                    
                elif (mask_exp.any()):
                    z_bottom = float(np.min(pts_z[mask_exp]))
                    
                   
                     
                    
                # === TOP Z (symmetric style, no np.partition) ===
                if (mask_ori_forT.any()):
                    # Apply the mask to get only the relevant z dimensions
                    masked_pts_z_top = pts_z[mask_ori_forT]
                elif (mask_exp.any()):
                    masked_pts_z_top = pts_z[mask_exp]
                else:
                    masked_pts_z_top = np.array([])

                if masked_pts_z_top.size > 0:
                    # Get the indices that would sort the array in ascending order
                    sorted_indices_top = np.argsort(masked_pts_z_top)
                    # Select the indices of the 10 largest values
                    top_10_indices_top = sorted_indices_top[-10:]
                    # Use these indices to get the 10 largest values
                    ten_max_z = masked_pts_z_top[top_10_indices_top]
                    # Filter with IQR to remove outliers among the top candidates
                    filtered_max_z = filter_noise_iqr(ten_max_z)
                    
                    z_top = float(np.max(filtered_max_z))
                else:
                    # Fallback if no points are available
                    z_top = z_bottom + default_heights.get(kitti_cls)

                #z_values_ori = pts_z[mask_ori_forT]
                #filtered_z_values = z_values_ori[z_values_ori < (z_bottom + 2.2)]
                #z_top = float(np.max(filtered_z_values)) if filtered_z_values.any() else z_bottom + default_heights.get(kitti_cls)
                
                    
            
                    
            h = float(z_top - z_bottom)
            
            if (h>2.1):
                h = default_heights.get(kitti_cls)
            #1.45
            if (h<1.45 and kitti_cls=="Car"):# and (h>0.9):
                h = default_heights.get(kitti_cls)
            if (h>0.9 and h<1.25 and kitti_cls=="Pedestrian"):# and (h>0.9):
                h = default_heights.get(kitti_cls)-0.1
                z_bottom -= 0.4 
                
                
           
            #print(velo_bin, ': ', h)
            quad_xy = quad_xy_orig
            # main axis & dims from oriented quad
            v_long, length, width = edges_and_axes(quad_xy)
            cx, cy = quad_xy.mean(axis=0)

            # orientation: choose long edge in LIDAR coords
            v_lidar = np.array([v_long[0], v_long[1], 0.0])
            v_cam = R_velo2cam @ v_lidar
            ry = float(np.arctan2(v_cam[0], v_cam[2]))   # yaw around camera Y
            ry -= np.pi / 2  # KITTI convention: +90 degrees
            # normalize to [-pi, pi]
            if ry > np.pi:
                ry -= 2*np.pi
            if ry < -np.pi:
                ry += 2*np.pi

            # location: bottom center in lidar → camera
            bottom_lidar = np.array([cx, cy, z_bottom, 1.0])
            t_cam = (velo2cam @ bottom_lidar)[:3]

            # 3D corners in camera and projection for 2D box
            corners_cam = compute_box3d_cam(h, width, length, t_cam[0], t_cam[1], t_cam[2], ry)
            corners_2d = project_cam_to_img(P2, corners_cam)
            x1, y1 = float(np.min(corners_2d[:, 0])), float(np.min(corners_2d[:, 1]))
            x2, y2 = float(np.max(corners_2d[:, 0])), float(np.max(corners_2d[:, 1]))

            # KITTI label
            out = f"{kitti_cls} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {h:.2f} {width:.2f} {length:.2f} {t_cam[0]:.2f} {t_cam[1]:.2f} {t_cam[2]:.2f} {ry:.2f} {conf:.3f}\n"
            lines_out.append(out)

    with open(out_txt, "w") as fo:
        fo.writelines(lines_out)


# -------------- Multi-threaded Batch runner --------------
def process_file(pred_file, velo_dir, calib_dir, out_dir):
    """Process a single file"""
    stem = pred_file.stem
    velo_bin = velo_dir / f"{stem}.bin"
    calib_txt = calib_dir / f"{stem}.txt"
    out_txt = out_dir / f"{stem}.txt"
    
    if not velo_bin.exists():
        print(f"[skip] no velodyne for {stem}")
        return f"skip-velo: {stem}"
    if not calib_txt.exists():
        print(f"[skip] no calib for {stem}")
        return f"skip-calib: {stem}"
    
    print(f"[convert] {stem}")
    convert_frame(str(pred_file), str(velo_bin), str(calib_txt), str(out_txt))
    return f"success: {stem}"

def main():
    # Your normalized BEV predictions (per frame)
    pred_dir = Path("ds-yolo-3obj-3h2-lowZwideJitterAug-amp-newSET/pred_m(s-mod)-full-s-(3offset)_filtered")#Path("ds-yolo-3obj-3h2-lowZwideJitterAug-amp/pred_m(s-mod)-base+aug_C64")#Path("ds-yolo-3obj-3h2/pred_m(s-mod)-01Conf")
    #pred_dir = Path("pedestrian_detect_bev/txt-test/") #Path("pedestrian_detect_bev/txt/") #Path("ds-yolo-3obj-3h2/pred_m(s-mod)-01Conf")  # your BEV txt predictions
    velo_dir = Path("lidar/training/velodyne")              # KITTI velodyne bin
    calib_dir = Path("calibration/training/calib")          # KITTI calibration txt
    out_dir = Path("results/new-data-full-s-New-3layer")               # output KITTI-format preds
    #out_dir = Path("results/data-s-modified-3h2-new-base+aug_C64")
    
    """
    ## TESTING with a single file
    pred_dir = Path("ds-yolo2dist-aug-3obj/pred_test")  # your BEV txt predictions
    velo_dir = Path("lidar/training/velodyne")              # KITTI velodyne bin
    calib_dir = Path("calibration/training/calib")          # KITTI calibration txt
    out_dir = Path("results/test")  
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all prediction files
    pred_files = sorted(pred_dir.glob("*.txt"))
    
    # Use ThreadPoolExecutor for parallel processing
    # Adjust max_workers based on your system capabilities
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks to the executor
        future_to_file = {
            executor.submit(process_file, pred_file, velo_dir, calib_dir, out_dir): pred_file 
            for pred_file in pred_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            pred_file = future_to_file[future]
            try:
                result = future.result()
                # You can handle results here if needed
            except Exception as exc:
                print(f'{pred_file} generated an exception: {exc}')

if __name__ == "__main__":
    main()