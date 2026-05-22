from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "bev-to-3Ds_multiT.py"), run_name="__main__")
