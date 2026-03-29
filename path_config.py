import sys
from pathlib import Path

# find where the project is
PROJECT_DIR = Path(__file__).resolve().parent

# useful paths for the rest of the repo
WORKSPACE_DIR = PROJECT_DIR / "workspace"
TRAINED_CHECKPOINT_DIR = WORKSPACE_DIR / "models"
DATASET_DIR = WORKSPACE_DIR / "datasets"
CONFIG_DIR = WORKSPACE_DIR / "configs"
EXTERNAL_DIR = PROJECT_DIR / "external"

# add SRC dir to path for imports
SRC_DIR = str(PROJECT_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)