# Import required libraries
import sys
from pathlib import Path

# Ensure the 'src' directory is in sys.path for module imports
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))