#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KITTI Object (2D / Bird's-Eye / 3D) Evaluation — Python port

Faithfully mirrors the logic of the original C++ devkit:
- File formats & folder layout
- Difficulty filtering
- DontCare handling
- IoU thresholds per metric/class
- 41 recall positions (post-2019)
- PR curve max-envelope
- AOS for 2D (if orientations are valid)
- Aggregation across images exactly like the C++ code

Source basis (ported): evaluate_object.cpp, mail.h
"""

import os
import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path


# Optional plotting (to replace gnuplot calls)
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Shapely for robust polygon union/intersection (equivalent to Boost.Geometry)
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# ---------------------- Static parameters (from C++ devkit) ----------------------
N_TESTIMAGES = 7518  # number of test images the C++ loops over
# Difficulty levels
EASY, MODERATE, HARD = 0, 1, 2
# Metrics
IMAGE, GROUND, BOX3D = 0, 1, 2

# Difficulty thresholds (height in px, max occlusion, max truncation)
MIN_HEIGHT = [40, 25, 25]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.3, 0.5]

# Classes and per-metric IoU thresholds
CLASS_NAMES = ["car", "pedestrian", "cyclist"]
CLASS_NAMES_CAP = ["Car", "Pedestrian", "Cyclist"]
NUM_CLASS = 3

# MIN_OVERLAP[metric][class] — same for all three classes in your file
MIN_OVERLAP = {
    IMAGE:  [0.7, 0.5, 0.5],
    GROUND: [0.7, 0.5, 0.5],
    BOX3D:  [0.7, 0.5, 0.5],
}

# 41 recall positions since 2019 change
N_SAMPLE_PTS = 41


# ---------------------- Data structures (mirror C++ structs) ----------------------
@dataclass
class Box:
    type: str
    x1: float
    y1: float
    x2: float
    y2: float
    alpha: float


@dataclass
class GroundTruth:
    box: Box
    truncation: float
    occlusion: int
    ry: float
    t1: float
    t2: float
    t3: float
    h: float
    w: float
    l: float


@dataclass
class Detection:
    box: Box
    thresh: float
    ry: float
    t1: float
    t2: float
    t3: float
    h: float
    w: float
    l: float


@dataclass
class PrData:
    v: List[float]            # detection scores (for threshold selection)
    similarity: float         # orientation similarity sum (AOS)
    tp: int
    fp: int
    fn: int

    @staticmethod
    def empty():
        return PrData(v=[], similarity=0.0, tp=0, fp=0, fn=0)



############### added ###########

def report_missed_objects(current_class: int,
                          groundtruth: List[List[GroundTruth]],
                          detections: List[List[Detection]],
                          difficulty: int,
                          metric: int,
                          file_names: List[str]):
    """
    Print missed GT objects (not matched by any detection) for a given class/metric.
    Only works for BOX3D (3D evaluation).
    """
    assert metric == BOX3D, "report_missed_objects is only for 3D eval"

    min_ov = MIN_OVERLAP[metric][current_class]
    cname = CLASS_NAMES[current_class]

    for idx, (gts, dets) in enumerate(zip(groundtruth, detections)):
        file_name = file_names[idx]
        for g in gts:
            if g.box.type.lower() != cname:
                continue

            # filter difficulty
            height = g.box.y2 - g.box.y1
            if (g.occlusion > MAX_OCCLUSION[difficulty] or
                g.truncation > MAX_TRUNCATION[difficulty] or
                height <= MIN_HEIGHT[difficulty]):
                continue

            matched = False
            for d in dets:
                if d.box.type.lower() != cname:
                    continue
                if box3d_overlap(d, g) >= min_ov:
                    matched = True
                    break

            if not matched:
                # print key info to help locate object
                print(f"[MISSED] {file_name} – GT {cname} "
                      f"2Dbox=({g.box.x1:.1f},{g.box.y1:.1f},{g.box.x2:.1f},{g.box.y2:.1f}) "
                      f"3Dloc=({g.t1:.2f},{g.t2:.2f},{g.t3:.2f}) hwl=({g.h:.2f},{g.w:.2f},{g.l:.2f})")
                
                
                
###############################

                

# ---------------------- I/O helpers (KITTI formats) ----------------------
def load_detections(path: str,
                    eval_image: List[bool],
                    eval_ground: List[bool],
                    eval_3d: List[bool]) -> Tuple[List[Detection], bool, bool]:
    """
    Returns (detections, compute_aos, success)
    compute_aos is turned False if any detection has alpha == -10 (invalid)
    """
    detections: List[Detection] = []
    compute_aos = True
    if not os.path.isfile(path):
        return detections, compute_aos, False

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 16:
                # skip malformed lines
                continue
            # KITTI detection line format:
            # type, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, t1, t2, t3, ry, score
            typ = parts[0]
            alpha = float(parts[3])
            x1, y1, x2, y2 = map(float, parts[4:8])
            h, w, l = map(float, parts[8:11])
            t1, t2, t3 = map(float, parts[11:14])
            ry = float(parts[14])
            score = float(parts[15])
            d = Detection(
                box=Box(typ, x1, y1, x2, y2, alpha),
                thresh=score, ry=ry, t1=t1, t2=t2, t3=t3, h=h, w=w, l=l
            )
            detections.append(d)

            if alpha == -10:
                compute_aos = False

            # note which metrics we can evaluate for this class
            for c in range(NUM_CLASS):
                if typ.lower() == CLASS_NAMES[c] or typ == CLASS_NAMES_CAP[c]:
                    if (x1 >= 0) and (not eval_image[c]):
                        eval_image[c] = True
                    if (t1 != -1000 and t3 != -1000 and w > 0 and l > 0) and (not eval_ground[c]):
                        eval_ground[c] = True
                    if (t1 != -1000 and t2 != -1000 and t3 != -1000 and h > 0 and w > 0 and l > 0) and (not eval_3d[c]):
                        eval_3d[c] = True
                    break

    return detections, compute_aos, True


def load_groundtruth(path: str) -> Tuple[List[GroundTruth], bool]:
    """
    KITTI GT line format:
    type, truncation, occlusion, alpha, x1, y1, x2, y2, h, w, l, t1, t2, t3, ry
    """
    gts: List[GroundTruth] = []
    if not os.path.isfile(path):
        return gts, False

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            typ = parts[0]
            trunc = float(parts[1])
            occ = int(float(parts[2]))  # robustness to "2.0" style
            alpha = float(parts[3])
            x1, y1, x2, y2 = map(float, parts[4:8])
            h, w, l = map(float, parts[8:11])
            t1, t2, t3 = map(float, parts[11:14])
            ry = float(parts[14])
            gts.append(GroundTruth(
                box=Box(typ, x1, y1, x2, y2, alpha),
                truncation=trunc, occlusion=occ, ry=ry, t1=t1, t2=t2, t3=t3, h=h, w=w, l=l
            ))
    return gts, True


# ---------------------- Geometry helpers (mirror Boost.Geometry code) ----------------------
def oriented_box_polygon_xy(l: float, w: float, ry: float, tx: float, tz: float) -> ShapelyPolygon:
    """
    Construct BEV (x-z plane) oriented rectangle polygon from (l, w, yaw ry, center tx, tz).
    Matches C++ toPolygon(): rotate axis-aligned corners by ry, then translate by (t1, t3).
    """
    # corners in local box frame: (+l/2, +w/2), (+l/2, -w/2), (-l/2, -w/2), (-l/2, +w/2)
    corners = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2],
    ], dtype=float)

    c, s = math.cos(ry), math.sin(ry)
    R = np.array([[c, s], [-s, c]], dtype=float)  # same as C++ (note sign pattern)
    rotated = (R @ corners.T).T
    rotated[:, 0] += tx
    rotated[:, 1] += tz

    # close polygon
    return ShapelyPolygon(rotated)


def iou_polygon(a: ShapelyPolygon, b: ShapelyPolygon, criterion: int = -1) -> float:
    """
    IoU / IoA variants on polygons:
    criterion: -1 -> IoU; 0 -> inter/area(a); 1 -> inter/area(b)
    """
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b)
    if inter.is_empty:
        inter_area = 0.0
    else:
        inter_area = inter.area

    if criterion == -1:
        union_area = a.union(b).area
        return inter_area / union_area if union_area > 0 else 0.0
    elif criterion == 0:
        aa = a.area
        return inter_area / aa if aa > 0 else 0.0
    else:
        ba = b.area
        return inter_area / ba if ba > 0 else 0.0


def image_box_overlap(det_box: Box, gt_box: Box, criterion: int = -1) -> float:
    # axis-aligned IoU in image plane
    x1 = max(det_box.x1, gt_box.x1)
    y1 = max(det_box.y1, gt_box.y1)
    x2 = min(det_box.x2, gt_box.x2)
    y2 = min(det_box.y2, gt_box.y2)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0.0

    inter = w * h
    a_area = (det_box.x2 - det_box.x1) * (det_box.y2 - det_box.y1)
    b_area = (gt_box.x2 - gt_box.x1) * (gt_box.y2 - gt_box.y1)

    if criterion == -1:   # IoU
        return inter / (a_area + b_area - inter) if (a_area + b_area - inter) > 0 else 0.0
    elif criterion == 0:  # IoA wrt det
        return inter / a_area if a_area > 0 else 0.0
    else:                 # IoA wrt gt
        return inter / b_area if b_area > 0 else 0.0


def ground_box_overlap(d: Detection, g: GroundTruth, criterion: int = -1) -> float:
    dp = oriented_box_polygon_xy(d.l, d.w, d.ry, d.t1, d.t3)
    gp = oriented_box_polygon_xy(g.l, g.w, g.ry, g.t1, g.t3)
    return iou_polygon(dp, gp, criterion)


def box3d_overlap(d: Detection, g: GroundTruth, criterion: int = -1) -> float:
    # base polygons in x-z
    dp = oriented_box_polygon_xy(d.l, d.w, d.ry, d.t1, d.t3)
    gp = oriented_box_polygon_xy(g.l, g.w, g.ry, g.t1, g.t3)
    inter_area = dp.intersection(gp).area if (not dp.is_empty and not gp.is_empty) else 0.0

    # vertical overlap along y
    ymax = min(d.t2, g.t2)
    ymin = max(d.t2 - d.h, g.t2 - g.h)
    inter_h = max(0.0, ymax - ymin)
    inter_vol = inter_area * inter_h

    det_vol = d.h * d.l * d.w
    gt_vol = g.h * g.l * g.w

    if criterion == -1:
        denom = det_vol + gt_vol - inter_vol
        return inter_vol / denom if denom > 0 else 0.0
    elif criterion == 0:
        return inter_vol / det_vol if det_vol > 0 else 0.0
    else:
        return inter_vol / gt_vol if gt_vol > 0 else 0.0


# ---------------------- KITTI evaluation helpers (ported logic) ----------------------
def get_thresholds(scores: List[float], n_gt: int) -> List[float]:
    """
    Select score thresholds giving ~uniform recall steps (N_SAMPLE_PTS),
    matching C++ getThresholds() behavior (descending scores, left-approx).
    """
    if n_gt <= 0:
        return []
    v = sorted(scores, reverse=True)
    thresholds = []
    current_recall = 0.0
    for i in range(len(v)):
        l_recall = float(i + 1) / n_gt
        r_recall = float(i + 2) / n_gt if i < (len(v) - 1) else l_recall
        # choose left if closer to current_recall
        if (r_recall - current_recall) < (current_recall - l_recall) and i < (len(v) - 1):
            continue
        thresholds.append(v[i])
        current_recall += 1.0 / (N_SAMPLE_PTS - 1.0)
    return thresholds


def clean_data(current_class: int,
               gt: List[GroundTruth],
               det: List[Detection],
               difficulty: int) -> Tuple[List[int], List[GroundTruth], List[int], int]:
    """
    Build ignored_gt, dontcare (list of GT entries with type DontCare),
    ignored_det, and count of valid GT (n_gt) for denominator of recall.
    Mirrors C++ cleanData().
    """
    ignored_gt: List[int] = []
    dontcare: List[GroundTruth] = []
    ignored_det: List[int] = []
    n_gt = 0

    # GT side
    for g in gt:
        height = g.box.y2 - g.box.y1

        # class check (neighbor/ignore rules)
        if g.box.type.lower() == CLASS_NAMES[current_class]:
            valid_class = 1
        elif (CLASS_NAMES[current_class].lower() == "pedestrian" and g.box.type.lower() == "person_sitting") or \
             (CLASS_NAMES[current_class].lower() == "car" and g.box.type.lower() == "van"):
            valid_class = 0
        else:
            valid_class = -1

        # difficulty filtering
        ignore = (g.occlusion > MAX_OCCLUSION[difficulty] or
                  g.truncation > MAX_TRUNCATION[difficulty] or
                  height <= MIN_HEIGHT[difficulty])

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)  # valid GT used for recall denominator
            n_gt += 1
        elif valid_class == 0 or (ignore and valid_class == 1):
            ignored_gt.append(1)  # GT present but ignored
        else:
            ignored_gt.append(-1) # other classes → FN if missed

    # DontCare
    for g in gt:
        if g.box.type.lower() == "dontcare":
            dontcare.append(g)

    # Detection side
    for d in det:
        height = abs(d.box.y2 - d.box.y1)
        if d.box.type.lower() == CLASS_NAMES[current_class]:
            valid_class = 1
        else:
            valid_class = -1

        if height < MIN_HEIGHT[difficulty]:
            ignored_det.append(1)  # short→ignored
        elif valid_class == 1:
            ignored_det.append(0)
        else:
            ignored_det.append(-1)

    return ignored_gt, dontcare, ignored_det, n_gt


def compute_statistics(current_class: int,
                       gt: List[GroundTruth],
                       det: List[Detection],
                       dc: List[GroundTruth],
                       ignored_gt: List[int],
                       ignored_det: List[int],
                       compute_fp: bool,
                       metric: int,
                       compute_aos: bool = False,
                       thresh: float = 0.0) -> PrData:
    """
    Core matching & counting logic per image, ported from C++ computeStatistics().
    """
    stat = PrData.empty()
    NO_DET = -1e10
    assigned_detection = [False] * len(det)
    ignored_threshold = [False] * len(det)

    # choose overlap function & IoU threshold
    if metric == IMAGE:
        def ovl(D, G, crit=-1): return image_box_overlap(D.box, G.box, crit)
    elif metric == GROUND:
        def ovl(D, G, crit=-1): return ground_box_overlap(D, G, crit)
    else:
        def ovl(D, G, crit=-1): return box3d_overlap(D, G, crit)

    min_ov = MIN_OVERLAP[metric][current_class]

    # low-score detections are ignored for FP counting
    if compute_fp:
        for i in range(len(det)):
            if det[i].thresh < thresh:
                ignored_threshold[i] = True

    # Evaluate each GT
    deltas = []  # for AOS
    for i_g, g in enumerate(gt):
        if ignored_gt[i_g] == -1:
            continue

        det_idx = -1
        valid_detection = NO_DET
        max_overlap = 0.0
        assigned_ignored_det = False

        for j, d in enumerate(det):
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue

            overlap = ovl(d, g, -1)

            if not compute_fp:
                # for recall thresholds, pick highest-score candidate above IoU
                if overlap > min_ov and d.thresh > valid_detection:
                    det_idx = j
                    valid_detection = d.thresh
            else:
                # for PR, pick greatest-overlap valid detection; if only ignored_det overlap, keep it until a valid one appears
                if overlap > min_ov and (overlap > max_overlap or assigned_ignored_det) and ignored_det[j] == 0:
                    max_overlap = overlap
                    det_idx = j
                    valid_detection = 1.0
                    assigned_ignored_det = False
                elif overlap > min_ov and valid_detection == NO_DET and ignored_det[j] == 1:
                    det_idx = j
                    valid_detection = 1.0
                    assigned_ignored_det = True

        # Count TP/FN/assignments
        if valid_detection == NO_DET and ignored_gt[i_g] == 0:
            stat.fn += 1
        elif valid_detection != NO_DET and (ignored_gt[i_g] == 1 or (det_idx >= 0 and ignored_det[det_idx] == 1)):
            if det_idx >= 0:
                assigned_detection[det_idx] = True
        elif valid_detection != NO_DET:
            stat.tp += 1
            if det_idx >= 0:
                stat.v.append(det[det_idx].thresh)
                if compute_aos:
                    deltas.append(g.box.alpha - det[det_idx].box.alpha)
                assigned_detection[det_idx] = True

    # FP / DontCare / AOS accumulation
    if compute_fp:
        # raw FP
        for i in range(len(det)):
            if not (assigned_detection[i] or ignored_det[i] in (-1, 1) or ignored_threshold[i]):
                stat.fp += 1

        # DontCare suppression (criterion=0 → inter/area(det))
        nstuff = 0
        for g in dc:
            for j, d in enumerate(det):
                if assigned_detection[j]:
                    continue
                if ignored_det[j] in (-1, 1):
                    continue
                if ignored_threshold[j]:
                    continue
                overlap = 0.0
                if metric == IMAGE:
                    overlap = image_box_overlap(d.box, g.box, 0)
                elif metric == GROUND:
                    overlap = ground_box_overlap(d, g, 0)
                else:
                    overlap = box3d_overlap(d, g, 0)
                if overlap > min_ov:
                    assigned_detection[j] = True
                    nstuff += 1

        stat.fp -= nstuff

        # AOS sum (FP contribute 0 similarity)
        if compute_aos:
            tmp = [0.0] * stat.fp + [(1.0 + math.cos(dd)) / 2.0 for dd in deltas]
            if stat.tp > 0 or stat.fp > 0:
                stat.similarity = sum(tmp)
            else:
                stat.similarity = -1.0

    return stat


def eval_class(current_class: int,
               groundtruth: List[List[GroundTruth]],
               detections: List[List[Detection]],
               compute_aos: bool,
               difficulty: int,
               metric: int) -> Tuple[List[float], List[float]]:
    """
    End-to-end class-wise evaluation producing precision[41] (and aos[41] if applicable),
    saving the exact PR construction as the C++ code.
    """
    assert len(groundtruth) == len(detections)
    n_images = len(groundtruth)

    # Collect ignored flags and DontCare + score list
    ignored_gts = []
    ignored_dets = []
    dcs = []
    n_gt_total = 0
    scores_all = []

    for i in range(n_images):
        ig, dc, idet, n_gt = clean_data(current_class, groundtruth[i], detections[i], difficulty)
        ignored_gts.append(ig)
        ignored_dets.append(idet)
        dcs.append(dc)
        n_gt_total += n_gt

        # collect scores for threshold selection
        tmp = compute_statistics(current_class, groundtruth[i], detections[i], dc, ig, idet,
                                 compute_fp=False, metric=metric, compute_aos=False)
        scores_all.extend(tmp.v)

    thresholds = get_thresholds(scores_all, n_gt_total)
    if len(thresholds) == 0:
        # no valid GTs → return zeros
        prec = [0.0] * N_SAMPLE_PTS
        aos = [0.0] * N_SAMPLE_PTS if compute_aos else []
        return prec, aos

    pr = [PrData.empty() for _ in range(len(thresholds))]

    # Aggregate TP/FP/FN over images for each threshold
    for i in range(n_images):
        for t_idx, thr in enumerate(thresholds):
            tmp = compute_statistics(current_class, groundtruth[i], detections[i], dcs[i],
                                     ignored_gts[i], ignored_dets[i],
                                     compute_fp=True, metric=metric,
                                     compute_aos=compute_aos, thresh=thr)
            pr[t_idx].tp += tmp.tp
            pr[t_idx].fp += tmp.fp
            pr[t_idx].fn += tmp.fn
            if tmp.similarity != -1:
                pr[t_idx].similarity += tmp.similarity

    # Build recall/precision arrays at 41 points
    precision = [0.0] * len(thresholds)
    aos = [0.0] * len(thresholds) if compute_aos else []

    recall_vals = []
    for i, v in enumerate(pr):
        denom = (v.tp + v.fn)
        r = (v.tp / denom) if denom > 0 else 0.0
        recall_vals.append(r)
        denom_p = (v.tp + v.fp)
        precision[i] = (v.tp / denom_p) if denom_p > 0 else 0.0
        if compute_aos:
            aos[i] = (v.similarity / denom_p) if denom_p > 0 else 0.0

    # Apply max-envelope from right to left
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
        if compute_aos:
            aos[i] = max(aos[i], aos[i + 1])

    # The C++ saves exactly N_SAMPLE_PTS rows (thresholds length should match 41 via getThresholds)
    # Return arrays sized to N_SAMPLE_PTS for saving/plotting
    if len(precision) < N_SAMPLE_PTS:
        # pad (rare) to 41 if numerical corner case
        precision += [precision[-1]] * (N_SAMPLE_PTS - len(precision))
        if compute_aos:
            aos += [aos[-1]] * (N_SAMPLE_PTS - len(aos))

    return precision[:N_SAMPLE_PTS], (aos[:N_SAMPLE_PTS] if compute_aos else [])


def save_stats(precision: List[float], aos: List[float], fp_det, fp_ori):
    if not precision:
        return
    fp_det.write(" ".join(f"{p:.6f}" for p in precision) + "\n")
    if aos:
        fp_ori.write(" ".join(f"{a:.6f}" for a in aos) + "\n")


def save_and_plot(dir_name: str, file_name: str, obj_type: str, curves: List[List[float]], is_aos: bool):
    """
    Save plot data in the same text format the C++ creates (Recall, Easy, Moderate, Hard),
    and (optionally) produce a PNG using matplotlib (instead of gnuplot).
    """
    os.makedirs(dir_name, exist_ok=True)
    out_txt = os.path.join(dir_name, f"{file_name}.txt")
    print(f"save {out_txt}")
    with open(out_txt, "w") as f:
        for i in range(N_SAMPLE_PTS):
            recall = i / (N_SAMPLE_PTS - 1.0)
            f.write(f"{recall:.6f} {curves[0][i]:.6f} {curves[1][i]:.6f} {curves[2][i]:.6f}\n")

    # Optional: PNG (the C++ uses gnuplot+ps2pdf+pdfcrop via system())
    if HAVE_MPL:
        plt.figure(figsize=(6, 4.5))
        x = np.linspace(0, 1, N_SAMPLE_PTS)
        labels = ["Easy", "Moderate", "Hard"]
        for idx in range(3):
            plt.plot(x, curves[idx], label=labels[idx])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Recall")
        plt.ylabel("Orientation Similarity" if is_aos else "Precision")
        title = obj_type[:1].upper() + obj_type[1:]
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(dir_name, f"{file_name}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()


def evaluate(pred_dir: Path, gt_dir: Path) -> bool:
    """
    Evaluate predictions (KITTI format) against ground truth labels.
    pred_dir: directory with prediction .txt files (KITTI format)
    gt_dir:   directory with ground truth .txt files (KITTI format)
    """
    # Create results/plot next to pred_dir
    result_dir = pred_dir.parent
    plot_dir = result_dir / "plot-new-data-full-s-New-3layer"
    os.makedirs(plot_dir, exist_ok=True)

    # Collect frames (only those available in pred_dir ∩ gt_dir)
    pred_files = sorted([f for f in pred_dir.glob("*.txt")])
    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return False

    groundtruth: List[List[GroundTruth]] = []
    detections: List[List[Detection]] = []

    compute_aos = True
    eval_image = [False] * NUM_CLASS
    eval_ground = [False] * NUM_CLASS
    eval_3d = [False] * NUM_CLASS

    print(f"Loading detections from {pred_dir} ...")
    for pred_path in pred_files:
        file_name = pred_path.name
        gt_path = gt_dir / file_name

        gt, gt_ok = load_groundtruth(str(gt_path))
        det, aos_ok, det_ok = load_detections(str(pred_path), eval_image, eval_ground, eval_3d)
        compute_aos = compute_aos and aos_ok

        groundtruth.append(gt)
        detections.append(det)

        if not gt_ok:
            print(f"WARNING: GT missing {file_name}, skipping.")
        if not det_ok:
            print(f"WARNING: Detection missing {file_name}, skipping.")

    print("  done.")

    # ----- 2D evaluation (IMAGE) -----
    for c in range(NUM_CLASS):
        if eval_image[c]:
            cname = CLASS_NAMES[c]
            print(f"Starting 2D evaluation ({cname}) ...")
            det_out = open(result_dir / f"stats_{cname}_detection.txt", "w")
            ori_out = open(result_dir / f"stats_{cname}_orientation.txt", "w") if compute_aos else None

            curves = []
            aos_curves = []
            for diff in (EASY, MODERATE, HARD):
                prec, aos = eval_class(c, groundtruth, detections, compute_aos, diff, IMAGE)
                curves.append(prec)
                if compute_aos:
                    aos_curves.append(aos)
                save_stats(prec, aos, det_out, ori_out)

            det_out.close()
            if compute_aos and ori_out:
                ori_out.close()

            save_and_plot(str(plot_dir), f"{cname}_detection", cname, curves, is_aos=False)
            if compute_aos:
                save_and_plot(str(plot_dir), f"{cname}_orientation", cname, aos_curves, is_aos=True)
            print("  done.")

    # ----- BEV evaluation (GROUND) -----
    for c in range(NUM_CLASS):
        if eval_ground[c]:
            cname = CLASS_NAMES[c]
            print(f"Starting bird's eye evaluation ({cname}) ...")
            det_out = open(result_dir / f"stats_{cname}_detection_ground.txt", "w")

            curves = []
            for diff in (EASY, MODERATE, HARD):
                prec, _ = eval_class(c, groundtruth, detections, False, diff, GROUND)
                curves.append(prec)
                save_stats(prec, [], det_out, None)

            det_out.close()
            save_and_plot(str(plot_dir), f"{cname}_detection_ground", cname, curves, is_aos=False)
            print("  done.")

    # ----- 3D evaluation (BOX3D) -----
    file_names = [f.name for f in pred_files]

    for c in range(NUM_CLASS):
        if eval_3d[c]:
            cname = CLASS_NAMES[c]
            print(f"Starting 3D evaluation ({cname}) ...")
            det_out = open(result_dir / f"stats_{cname}_detection_3d.txt", "w")

            curves = []
            for diff in (EASY, MODERATE, HARD):
                prec, _ = eval_class(c, groundtruth, detections, False, diff, BOX3D)
                curves.append(prec)
                save_stats(prec, [], det_out, None)

            det_out.close()
            save_and_plot(str(plot_dir), f"{cname}_detection_3d", cname, curves, is_aos=False)
            if cname == "car":  # only report missed cars
                report_missed_objects(c, groundtruth, detections,
                                    difficulty=MODERATE,  # or EASY/HARD
                                    metric=BOX3D,
                                    file_names=file_names)
            print("  done.")

    return True


def main():
    # predictions you generated (KITTI format)
    pred_dir = Path("results/new-data-full-s-New-3layer")
    # ground truth KITTI-style labels (validation subset)
    gt_dir   = Path("labels/training/label_2")

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir {pred_dir} not found")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth dir {gt_dir} not found")

    print(f"Evaluating predictions in {pred_dir} against {gt_dir}...")
    ok = evaluate(pred_dir, gt_dir)
    


    if not ok:
        print("Evaluation failed")
    else:
        print("Evaluation done")
        
if __name__ == "__main__":
    main()
