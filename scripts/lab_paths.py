"""Where the derived data files live (tracked in the repo under data/).

  data/leaderboard.json   import_results.py   (read by bootstrap_ci.py, variance_stats.py, import_toolmode.py, weight_sensitivity.py)
  data/trialspread.json   import_results.py   (written next to the leaderboard)
  data/variance.json      variance_stats.py   (bootstrap_ci.py --pool-variance adds "pooled" / "pooled_ties")
  data/toolmode.json      import_toolmode.py

Override the directory with $LAB_DATA_DIR, or only the leaderboard file with $LAB_LEADERBOARD_OUT (kept for compatibility).
The site build (11ty, separate private repo) reads the same files: its site/src/_data/*.json are symlinks to data/."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LAB_DATA_DIR") or ROOT / "data")
def data(name): return DATA_DIR / name
def leaderboard(): return Path(os.environ.get("LAB_LEADERBOARD_OUT") or data("leaderboard.json"))
