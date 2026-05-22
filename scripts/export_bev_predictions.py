from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "yolo_bev_pred_saver.py"), run_name="__main__")
