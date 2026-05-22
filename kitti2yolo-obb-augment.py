import os
import glob
import numpy as np
import cv2
import random
from shutil import copyfile
import imageio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor



# BEV params
x_min, x_max, y_min, y_max = 0, 70, -40, 40
res = 0.1
W, H = int((x_max - x_min)/res), int((y_max - y_min)/res)

def read_calib(path):
    data = {}
    with open(path, 'r') as f:
        for l in f:
            if ':' not in l: continue
            k, v = l.split(':',1)
            data[k] = np.fromstring(v, sep=' ')
    Tr = data['Tr_velo_to_cam'].reshape(3,4)
    R0 = data['R0_rect'].reshape(3,3)
    T = np.vstack([Tr, [0,0,0,1]])
    R_ext = np.eye(4); R_ext[:3,:3] = R0
    return np.linalg.inv(R_ext @ T)

def complex_yolo_bev_encoding(path, H, W, extra_offset=0.0):
    """
    Encodes 3D LiDAR points into a 2D BEV map with three height-banded
    intensity channels:

      R: mean intensity of points with z < 0.65
      G: mean intensity of points with 0.65 <= z < 1.30
      B: mean intensity of points with z >= 1.30

    Notes:
      - Uses global/grid params: x_min, x_max, y_min, y_max, res
      - Intensities are scaled to [0, 255] for 8-bit output.

    Args:
        path (str): Path to the LiDAR .bin file (Nx4 float32: x,y,z,intensity).
        H (int): Height (rows) of the BEV image.
        W (int): Width (cols) of the BEV image.

    Returns:
        np.ndarray: (H, W, 3) uint8 BEV image with channels (R,G,B) as above.
    """
    

    # Configuration
    z_offset = 1.73  # adjust if your coordinate frame requires it
    z_offset += extra_offset
    
    

    # Height band thresholds (after z_offset applied)
    band1_max = 0.65
    band2_max = 1.30  # band2 is [0.65, 1.30); band3 is >= 1.30

    # Load points
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts[:, 2] += z_offset  # shift z

    # Crop to evaluation area
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
    )
    p = pts[mask]
    if p.size == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Compute grid indices
    ix = ((p[:, 0] - x_min) / res).astype(int)
    iy = ((p[:, 1] - y_min) / res).astype(int)

    # Prepare BEV channels
    bev_r = np.zeros((H, W), dtype=np.float32)  # mean intensity for z < 0.65
    bev_g = np.zeros((H, W), dtype=np.float32)  # mean intensity for 0.65 <= z < 1.30
    bev_b = np.zeros((H, W), dtype=np.float32)  # mean intensity for z >= 1.30

    # Collect intensities per cell per band
    # cell_bands[(row, col)] -> [list_band1, list_band2, list_band3]
    cell_bands = defaultdict(lambda: [[], [], []])

    z = p[:, 2]
    intensity = p[:, 3]

    for xg, yg, zg, it in zip(ix, iy, z, intensity):
        # bounds check (row = y -> 0..H-1, col = x -> 0..W-1)
        if 0 <= yg < H and 0 <= xg < W:
            if zg < band1_max:
                band_idx = 0  # R
            elif zg < band2_max:
                band_idx = 1  # G
            else:
                band_idx = 2  # B
            cell_bands[(yg, xg)][band_idx].append(it)

    # Compute mean intensity for each band per cell
    for (row, col), bands in cell_bands.items():
        # bands = [list_band1, list_band2, list_band3]
        if bands[0]:  # R
            bev_r[row, col] = (np.max(bands[0])+0.1)*1.3 * 255.0
        if bands[1]:  # G
            bev_g[row, col] = (np.max(bands[1])+0.1)*1.3 * 255.0
        if bands[2]:  # B
            bev_b[row, col] = (np.max(bands[2])+0.1)*1.3 * 255.0

    # Clip and stack to uint8 image
    bev = np.stack([
        np.clip(bev_r, 0, 255),
        np.clip(bev_g, 0, 255),
        np.clip(bev_b, 0, 255),
    ], axis=-1).astype(np.uint8)

    return bev



def complex_yolo_bev_encoding_old(path, H, W):
    """
    Encodes 3D LiDAR points into a 2D BEV map with custom channels.
    
    Args:
        path (str): Path to the LiDAR point cloud file.
        H (int): Height of the BEV map.
        W (int): Width of the BEV map.
        x_min, x_max, y_min, y_max (float): Bounding box for point cloud.
        res (float): Resolution of the BEV grid.

    Returns:
        np.ndarray: The encoded BEV image as a NumPy array (H, W, 3).
    """
    z_offset = 1.73
    max_count = 30.0  # Used for normalization of density
    max_height = 3.0
    
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts[:, 2] += z_offset
    mask = (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
    p = pts[mask]

    bev_r = np.zeros((H, W), dtype=np.float32)
    bev_g = np.zeros((H, W), dtype=np.float32)
    bev_b = np.zeros((H, W), dtype=np.float32)

    # Calculate grid coordinates for each point
    ix = ((p[:, 0] - x_min) / res).astype(int)
    iy = ((p[:, 1] - y_min) / res).astype(int)

    # Group points by grid cell
    cell_points = defaultdict(list)
    for x, y, z, intensity in zip(ix, iy, p[:, 2], p[:, 3]):
        # Ensure coordinates are within map bounds
        if 0 <= y < H and 0 <= x < W:
            cell_points[(y, x)].append((z, intensity))

    for (y, x), pts_in_cell in cell_points.items():
        zs = np.array([pt[0] for pt in pts_in_cell])
        intensities = np.array([pt[1] for pt in pts_in_cell])
        
        # Red Channel: Maximum height
        if len(zs) > 0:
            bev_r[y, x] = np.clip(np.max(zs) / max_height, 0, 1) * 255
        
        # Green Channel: Maximum intensity
        if len(intensities) > 0:
            # I(PΩi→j) is max intensity in cell j
            bev_g[y, x] = np.clip(np.max(intensities) * 255, 0, 255)

        # Blue Channel: Log-normalized point density
        # N is the number of points in the cell
        N = len(pts_in_cell)
        if N > 0:
            # zb(Sj) = min(1.0, log(N + 1) / 64)
            log_density = np.log(N + 1)
            bev_b[y, x] = np.clip(log_density / 3.91, 0, 1) * 255

    bev = np.stack([bev_r, bev_g, bev_b], axis=-1)
    return bev.astype(np.uint8) 

def custom_bev_encoding_dist(path, H, W):
    z_offset = 1.73
    max_count = 30.0
    max_height = 3.0
    y_ref = H // 2
    x_ref = 0

    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts[:, 2] += z_offset
    mask = (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
    p = pts[mask]

    bev_r = np.zeros((H, W), dtype=np.float32)
    bev_g = np.zeros((H, W), dtype=np.float32)
    bev_b = np.zeros((H, W), dtype=np.float32)

    y_coords, x_coords = np.indices((H, W))
    distances = np.sqrt((x_coords - x_ref)**2 + (y_coords - y_ref)**2)
    max_dist = np.max(distances)
    if max_dist > 0:
        normalized_distances = (distances / max_dist) * 255
    else:
        normalized_distances = np.zeros_like(distances)
    bev_b = normalized_distances.astype(np.float32)

    ix = ((p[:, 0] - x_min) / res).astype(int)
    iy = ((p[:, 1] - y_min) / res).astype(int)

    cell_points = defaultdict(list)
    for x, y, z, intensity in zip(ix, iy, p[:, 2], p[:, 3]):
        cell_points[(y, x)].append((z, intensity))

    for (y, x), pts_in_cell in cell_points.items():
        zs = np.array([pt[0] for pt in pts_in_cell])
        intensities = np.array([pt[1] for pt in pts_in_cell])
        if len(zs) > 0:
            bev_g[y, x] = np.clip(np.mean(intensities) * 255, 0, 255)
            bev_r[y, x] = np.clip(np.max(zs) / max_height, 0, 1) * 255
    
    bev = np.stack([bev_r, bev_g, bev_b], axis=-1)
    return bev.astype(np.uint8)

def apply_augmentations(bev, labels, H, W, bin_path=None):
    """
    Augment BEV by (1) shifting scene vertically in z by Δz ~ U(-0.5, +0.5),
    (2) re-encoding the three height-banded mean-intensity channels, and
    (3) adding per-pixel Gaussian noise ONLY on non-zero pixels.

    If bin_path is provided, the BEV is regenerated from points after z-shift.
    If bin_path is None, we keep the input BEV (cannot re-bin without points).

    Returns:
        aug_bev (H, W, 3) uint8, aug_labels (unchanged), aug_params (dict)
    """
   
    # Globals in your script: x_min, x_max, y_min, y_max, res
    z_offset = 1.73
    band1_max = 0.65
    band2_max = 1.30

    # 1) random vertical shift
    z_shift = np.random.uniform(-0.3, 0.3)

    if bin_path is not None:
        # --- regenerate BEV from raw points with the z-shift ---
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        pts[:, 2] += (z_offset + z_shift)

        # crop to area of interest
        mask = (
            (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
            (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
        )
        p = pts[mask]
        if p.size == 0:
            aug_bev = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            ix = ((p[:, 0] - x_min) / res).astype(int)
            iy = ((p[:, 1] - y_min) / res).astype(int)

            bev_r = np.zeros((H, W), dtype=np.float32)  # z < 0.65
            bev_g = np.zeros((H, W), dtype=np.float32)  # 0.65 <= z < 1.30
            bev_b = np.zeros((H, W), dtype=np.float32)  # z >= 1.30

            cell_bands = defaultdict(lambda: [[], [], []])
            z_vals = p[:, 2]
            inten  = p[:, 3]

            for xg, yg, zg, it in zip(ix, iy, z_vals, inten):
                if 0 <= yg < H and 0 <= xg < W:
                    if zg < band1_max:
                        cell_bands[(yg, xg)][0].append(it)  # R
                    elif zg < band2_max:
                        cell_bands[(yg, xg)][1].append(it)  # G
                    else:
                        cell_bands[(yg, xg)][2].append(it)  # B

            for (row, col), bands in cell_bands.items():
                if bands[0]:  # R
                    bev_r[row, col] = (np.max(bands[0]) + 0.1) * 1.3 * 255.0
                if bands[1]:  # G
                    bev_g[row, col] = (np.max(bands[1]) + 0.1) * 1.3 * 255.0
                if bands[2]:  # B
                    bev_b[row, col] = (np.max(bands[2]) + 0.1) * 1.3 * 255.0

            aug_bev = np.stack([
                np.clip(bev_r, 0, 255),
                np.clip(bev_g, 0, 255),
                np.clip(bev_b, 0, 255),
            ], axis=-1).astype(np.uint8)
    else:
        # cannot re-bin without points; keep the original BEV
        print("Warning: bin_path is None, cannot re-bin after z-shift; keeping original BEV")
        aug_bev = bev.copy()

    # 2) per-pixel Gaussian noise (unique per pixel/channel) on non-zero pixels
    NOISE_STD = 20.0  # adjust if you want stronger/weaker perturbation
    noise = np.random.normal(loc=0.0, scale=NOISE_STD, size=aug_bev.shape).astype(np.float32)

    # build a mask of where any channel is nonzero, then apply per-channel
    nonzero_mask = (aug_bev > 0)  # shape (H, W, 3), boolean
    aug_bev = aug_bev.astype(np.float32)
    aug_bev[nonzero_mask] = aug_bev[nonzero_mask] + noise[nonzero_mask]
    aug_bev = np.clip(aug_bev, 0, 255).astype(np.uint8)

    # labels unchanged
    aug_labels = labels.copy()
    aug_params = {'z_offset': float(z_shift), 'jitter': float(NOISE_STD)}




    #image-wide noise:
    
    # 2) image-wide intensity jitter (same value added to all pixels/channels)
    # JITTER_RANGE = 30.0  # +/- range in 8-bit intensity space
    # jitter = float(np.random.uniform(-JITTER_RANGE, JITTER_RANGE))

    # aug_bev = aug_bev.astype(np.float32)
    # aug_bev += jitter
    # aug_bev = np.clip(aug_bev, 0, 255).astype(np.uint8)
    #aug_params = {'z_offset': float(z_shift), 'jitter': jitter}
    
    # labels unchanged
    aug_labels = labels.copy()

    

    return aug_bev, aug_labels, aug_params



def apply_augmentations_old(bev, labels, H, W):
    aug_bev = bev.copy()
    aug_labels = labels.copy()

    dist_offset = random.uniform(-20, 20)
    aug_bev[:, :, 2] = np.clip(aug_bev[:, :, 2] + dist_offset, 0, 255)

    bboxes = []
    for label in labels:
        parts = label.split()
        coords = [float(c) for c in parts[1:]]
        denormalized_coords = np.array(coords).reshape(-1, 2) * np.array([W, H])
        bboxes.append({'coords': denormalized_coords})

    for bbox in bboxes:
        corners = bbox['coords'].astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [corners], 1)
        
        obb_mask = mask.astype(bool)
        data_mask = (aug_bev[:, :, 0] > 0) | (aug_bev[:, :, 1] > 0)
        final_mask = obb_mask & data_mask

        z_offset = random.uniform(-30, 30)
        aug_bev[:, :, 0][final_mask] = np.clip(aug_bev[:, :, 0][final_mask] + z_offset, 0, 255)

        noise = np.random.normal(0, 30, aug_bev.shape[:2])
        aug_bev[:, :, 1][final_mask] = np.clip(aug_bev[:, :, 1][final_mask] + noise[final_mask], 0, 255)

    return aug_bev, aug_labels

def kitti_to_yolo_obb(lab_path, inv_calib):
    labels = []
    with open(lab_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 15:
                continue
            if p[0] == "Car":
                class_idx = 0
            elif p[0] == "Pedestrian":
                class_idx = 1
            elif p[0] == "Cyclist":
                class_idx = 2
            else:
                continue
             
                
            _, w, l = map(float, p[8:11])
            x_c, y_c, z_c, ry = map(float, p[11:15])
            x_lidar, y_lidar, _ = (inv_calib @ [x_c, y_c, z_c, 1])[:3]
            yaw = -ry + np.pi / 2
            x_pix = (x_lidar - x_min) / res
            y_pix = (y_lidar - y_min) / res
            dx = l / 2 / res
            dy = w / 2 / res
            corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            bev_corners = np.dot(corners, R.T) + np.array([x_pix, y_pix])
            bev_corners[:, 0] = np.clip(bev_corners[:, 0], 0, W - 1)
            bev_corners[:, 1] = np.clip(bev_corners[:, 1], 0, H - 1)
            bev_corners[:, 0] /= W
            bev_corners[:, 1] /= H
            coords = bev_corners.flatten()
            label_str = f"{class_idx} " + " ".join(f"{v:.6f}" for v in coords)
            labels.append(label_str)
    return labels

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- NEW: Function to process a single file for multi-threading ---
def process_file(fid, split, img_dir, lbl_dir, bdir, ldir, cdir, H, W):
    print(f"Processing file: {fid}")
    # Process ORIGINAL data
    #bev = custom_bev_encoding_dist(f"{bdir}{fid}.bin", H, W)
    bev = complex_yolo_bev_encoding(f"{bdir}{fid}.bin", H, W, extra_offset= 0.0)
    if split == "val":
        bev_pos30 = complex_yolo_bev_encoding(f"{bdir}{fid}.bin", H, W, extra_offset= 0.3)
        bev_neg30 = complex_yolo_bev_encoding(f"{bdir}{fid}.bin", H, W, extra_offset= -0.3)
        
    invc = read_calib(f"{cdir}{fid}.txt")
    labels = kitti_to_yolo_obb(f"{ldir}{fid}.txt", invc)

    imageio.imwrite(os.path.join(img_dir, f"{fid}.png"), bev)
    with open(os.path.join(lbl_dir, f"{fid}.txt"), "w") as f:
        for line in labels:
            f.write(line + "\n")
            
    if split == "val":
        print(f"  Generating additional val augmentations for file: {fid}")
        imageio.imwrite(os.path.join(img_dir, f"{fid}_pos30.png"), bev_pos30)
        with open(os.path.join(lbl_dir, f"{fid}_pos30.txt"), "w") as f:
            for line in labels:
                f.write(line + "\n")
        imageio.imwrite(os.path.join(img_dir, f"{fid}_neg30.png"), bev_neg30)
        with open(os.path.join(lbl_dir, f"{fid}_neg30.txt"), "w") as f:
            for line in labels:
                f.write(line + "\n")
    
    # Process AUGMENTED data only for the training split
    augment_flag = True
    if split == "train" and augment_flag:
        print(f"  Augmenting file: {fid}")
        # First augmented sample
        aug_bev1, aug_labels1, aug_params = apply_augmentations(bev, labels, H, W, bin_path=f"{bdir}{fid}.bin")
        aug_fid1 = f"{fid}_aug1"
        z_off = int(aug_params['z_offset'])
        n_noise = int(aug_params['jitter'])
        imageio.imwrite(os.path.join(img_dir, f"{aug_fid1}.png"), aug_bev1)
        with open(os.path.join(lbl_dir, f"{aug_fid1}.txt"), "w") as f:
            for line in aug_labels1:
                f.write(line + "\n")
        
        # Second augmented sample
        #aug_bev2, aug_labels2 = apply_augmentations(bev, labels, H, W)
        #aug_fid2 = f"{fid}_aug2"
        #imageio.imwrite(os.path.join(img_dir, f"{aug_fid2}.png"), aug_bev2)
        #with open(os.path.join(lbl_dir, f"{aug_fid2}.txt"), "w") as f:
        #    for line in aug_labels2:
        #        f.write(line + "\n")

# --- NEW HELPER FUNCTION ---
def load_indices_from_file(filepath):
    """Reads file IDs (e.g., '000001') from a text file."""
    try:
        with open(filepath, 'r') as f:
            # Strip whitespace and collect IDs
            indices = [line.strip() for line in f if line.strip()]
        return set(indices)
    except FileNotFoundError:
        print(f"Error: Index file not found at {filepath}")
        return set()
# ---------------------------


if __name__ == "__main__":
    bdir = "lidar/training/velodyne/"
    ldir = "labels/training/label_2/"
    cdir = "calibration/training/calib/"
    out_img_train = "ds-yolo-3obj-3h2-lowZ-PXJitterAug-amp-newSET/images/train"
    #out_img_val = "ds-yolo-3obj-3h2-lowZ-PXJitterAug-amp-newSET/images/val"
    out_img_val = "ds-yolo-3obj-3h2-lowZ-PXJitterAug-amp-newSET/images/val-3"

    out_lbl_train = "ds-yolo-3obj-3h2-lowZ-PXJitterAug-amp-newSET/labels/train"
    #out_lbl_val = "ds-yolo-3obj-3h2-lowZ-PXJitterAug-amp-newSET/labels/val"
    out_lbl_val = "ds-yolo-3obj-3h2-lowZ-PXJitterAug-amp-newSET/labels/val3"


    # Define paths for the index files
    TRAIN_IDX_FILE = 'train_idx.txt'
    VAL_IDX_FILE = 'val_idx.txt'
    
    ensure_dir(out_img_train)
    ensure_dir(out_img_val)
    ensure_dir(out_lbl_train)
    ensure_dir(out_lbl_val)
    
    # 1. Load the indices from the text files
    train_indices = load_indices_from_file(TRAIN_IDX_FILE)
    val_indices = load_indices_from_file(VAL_IDX_FILE)

    if not train_indices and not val_indices:
        print("FATAL: No indices loaded. Check index file paths and content.")
        exit()
        
    # 2. Get all available frame IDs from the source directory
    # Note: We still use glob to ensure we only process existing files
    all_source_frames = {os.path.basename(f)[:-4] for f in glob.glob(bdir + "*.bin")}
    
    # 3. Create the final lists of frames to process
    # Use set intersection to ensure we only try to process files that exist AND are in the list
    train_frames = sorted(list(train_indices.intersection(all_source_frames)))
    val_frames = sorted(list(val_indices.intersection(all_source_frames)))
    
    print(f"Total source files found: {len(all_source_frames)}")
    print(f"Train files to process (based on {TRAIN_IDX_FILE}): {len(train_frames)}")
    print(f"Validation files to process (based on {VAL_IDX_FILE}): {len(val_frames)}")

    #########################
    ### OLD SPLIT:
    # LAST_TRAIN_IDX = 3712
    # frames = sorted([os.path.basename(f)[:-4] for f in glob.glob(bdir + "*.bin")])
    # train_frames = frames[:LAST_TRAIN_IDX]
    # val_frames = frames[LAST_TRAIN_IDX:]

    #########################
    
    for split, split_frames, img_dir, lbl_dir in [
        #("train", train_frames, out_img_train, out_lbl_train),
        ("val", val_frames, out_img_val, out_lbl_val)
    ]:
        print(f"Starting {split} split with {len(split_frames)} files.")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Prepare arguments for each file
            args = [(fid, split, img_dir, lbl_dir, bdir, ldir, cdir, H, W) for fid in split_frames]
            # Submit tasks to the thread pool
            executor.map(lambda p: process_file(*p), args)