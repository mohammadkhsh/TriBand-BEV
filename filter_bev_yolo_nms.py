#!/usr/bin/env python3
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple


@dataclass
class Detection:
    class_id: int
    confidence: float
    polygon: List[Tuple[float, float]]
    line: str


def polygon_signed_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return area * 0.5


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    return abs(polygon_signed_area(poly))


def is_inside(
    point: Tuple[float, float],
    edge_start: Tuple[float, float],
    edge_end: Tuple[float, float],
    orientation_sign: float,
) -> bool:
    cross = (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (point[0] - edge_start[0])
    return cross * orientation_sign >= -1e-12


def line_intersection(
    s: Tuple[float, float],
    e: Tuple[float, float],
    cp1: Tuple[float, float],
    cp2: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    x1, y1 = s
    x2, y2 = e
    x3, y3 = cp1
    x4, y4 = cp2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def polygon_clip(
    subject: List[Tuple[float, float]],
    clip: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    output = subject
    if not output or len(clip) < 3:
        return []
    orientation_sign = 1.0 if polygon_signed_area(clip) >= 0 else -1.0
    for i in range(len(clip)):
        cp1 = clip[i]
        cp2 = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if is_inside(e, cp1, cp2, orientation_sign):
                if not is_inside(s, cp1, cp2, orientation_sign):
                    inter = line_intersection(s, e, cp1, cp2)
                    if inter is not None:
                        output.append(inter)
                output.append(e)
            elif is_inside(s, cp1, cp2, orientation_sign):
                inter = line_intersection(s, e, cp1, cp2)
                if inter is not None:
                    output.append(inter)
            s = e
    return output


def polygon_iou(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0
    inter_poly = polygon_clip(poly1, poly2)
    inter_area = polygon_area(inter_poly)
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    for det in ordered:
        if any(polygon_iou(det.polygon, kept_det.polygon) >= iou_threshold for kept_det in kept):
            continue
        kept.append(det)
    return kept


def parse_line(line: str, path: Path, line_num: int) -> Optional[Detection]:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) != 10:
        print(
            f"Warning: {path} line {line_num} has {len(parts)} fields; expected 10. Skipping.",
            file=sys.stderr,
        )
        return None
    try:
        class_id = int(float(parts[0]))
        confidence = float(parts[1])
        coords = list(map(float, parts[2:]))
    except ValueError:
        print(f"Warning: {path} line {line_num} has non-numeric values. Skipping.", file=sys.stderr)
        return None
    polygon = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
    return Detection(class_id=class_id, confidence=confidence, polygon=polygon, line=line.strip())


def read_detections(paths: List[Path]) -> List[Detection]:
    detections: List[Detection] = []
    for path in paths:
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines, start=1):
            det = parse_line(line, path, idx)
            if det is not None:
                detections.append(det)
    return detections


def write_detections(path: Path, detections: List[Detection]) -> None:
    with path.open("w", newline="\n") as handle:
        for det in detections:
            handle.write(det.line)
            handle.write("\n")


def collect_groups(input_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.txt")):
        base = path.stem.split("_")[0]
        groups[base].append(path)
    return groups


def run(input_dir: Path, output_dir: Path, iou_threshold: float, quiet: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = collect_groups(input_dir)
    for base, paths in sorted(groups.items()):
        detections = read_detections(paths)
        if not detections:
            write_detections(output_dir / f"{base}.txt", [])
            if not quiet:
                print(f"{base}: 0 -> 0")
            continue
        by_class: Dict[int, List[Detection]] = defaultdict(list)
        for det in detections:
            by_class[det.class_id].append(det)
        kept: List[Detection] = []
        for class_id in sorted(by_class.keys()):
            kept.extend(nms(by_class[class_id], iou_threshold))
        kept.sort(key=lambda d: d.confidence, reverse=True)
        write_detections(output_dir / f"{base}.txt", kept)
        if not quiet:
            print(f"{base}: {len(detections)} -> {len(kept)}")


def main() -> int:
    default_input = Path(
        "ds-yolo-3obj-3h2-lowZwideJitterAug-amp-newSET/pred_m(s-mod)-full-s-(3offset)"
    )
    parser = argparse.ArgumentParser(
        description="Merge BEV YOLO prediction files and apply polygon NMS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Folder with prediction .txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="ds-yolo-3obj-3h2-lowZwideJitterAug-amp-newSET/pred_m(s-mod)-full-s-(3offset)_filtered",
        help="Output folder for filtered predictions (default: <input>/filtered).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for suppression (default: 0.5).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file logging.")
    args = parser.parse_args()

    input_dir = args.input
    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1
    output_dir = args.output if args.output is not None else input_dir / "filtered"
    run(input_dir, output_dir, args.iou_threshold, args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
