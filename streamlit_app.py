"""Streamlit Cloud Root Entrypoint for ORCA Platform.

Ensures proper path resolution and launches the Streamlit Control Tower.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import runpy

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure root working directory for model artifacts and data loading
os.chdir(str(ROOT_DIR))

# Execute main dashboard app
dashboard_app_path = SRC_DIR / "delay_intelligence" / "dashboard" / "app.py"
runpy.run_path(str(dashboard_app_path), run_name="__main__")
