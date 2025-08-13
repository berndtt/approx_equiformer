# ensure the repo root is importable: from nets.foo import Bar
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
