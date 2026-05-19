# Databricks notebook source
# Calibre VN2 Smoke Test — Databricks
# MAGIC %md
# MAGIC ## Calibre VN2 Smoke Test on Databricks
# MAGIC
# MAGIC This notebook runs the Calibre forecasting engine on VN2 fixture data on the Databricks driver.
# MAGIC
# MAGIC Prerequisites:
# MAGIC - Repo cloned into Databricks Repos (or benchmark files on DBFS)
# MAGIC - Calibre wheel uploaded to DBFS at `/dbfs/mnt/calibre/calibre-0.1.0-py3-none-any.whl`

# COMMAND ----------

import os
import sys

# If running inside Databricks Repos, add repo root to path so benchmarks/ is importable.
# Adjust REPO_ROOT to match your clone path.
REPO_ROOT = "/Workspace/Repos/calibre"
if os.path.isdir(REPO_ROOT):
    sys.path.insert(0, REPO_ROOT)
    print(f"Using repo at {REPO_ROOT}")
else:
    print("Warning: REPO_ROOT not found; assuming benchmarks/ are on PYTHONPATH already.")

# COMMAND ----------

# Install Calibre with benchmark extras.
# If the wheel is not yet on DBFS, build it locally and upload:
#   cd /path/to/calibre && uv build --wheel
#   dbfs cp dist/calibre-0.1.0-py3-none-any.whl dbfs:/mnt/calibre/

%pip install /dbfs/mnt/calibre/calibre-0.1.0-py3-none-any.whl[benchmarks]

# COMMAND ----------

# Unset the hardcoded home MLflow tracking URI so Databricks-managed MLflow is used.
# If you prefer to disable MLflow entirely, set CALIBRE_NO_MLFLOW=1 instead.
os.environ.pop("MLFLOW_TRACKING_URI", None)

# Optional: disable MLflow if you do not want any tracking for the smoke test.
# os.environ["CALIBRE_NO_MLFLOW"] = "1"

print("MLFLOW_TRACKING_URI:", os.environ.get("MLFLOW_TRACKING_URI", "(not set — will use default)"))

# COMMAND ----------

from pathlib import Path

# Copy VN2 fixture data to DBFS if not already present.
local_fixture = Path(REPO_ROOT) / "benchmarks" / "vn2" / "fixture"
dbfs_fixture = Path("/dbfs/mnt/calibre/vn2-fixture")

dbfs_fixture.mkdir(parents=True, exist_ok=True)

if local_fixture.exists():
    for src in local_fixture.iterdir():
        dst = dbfs_fixture / src.name
        if src.is_file() and not dst.exists():
            dst.write_bytes(src.read_bytes())
            print(f"Copied {src.name} -> {dst}")
else:
    print(f"Warning: local fixture not found at {local_fixture}")

print(f"Fixture files in DBFS: {sorted(p.name for p in dbfs_fixture.iterdir())}")

# COMMAND ----------

# Run the Databricks smoke test. The fixture was already copied above,
# so main() will detect it and skip its own copy step.
from scripts.databricks_smoke import main

result = main()
print(f"Smoke test exit code: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: run the full winning benchmark
# MAGIC
# MAGIC After the smoke test passes, run the VN2 winning benchmark:
# MAGIC
# MAGIC ```python
# MAGIC from benchmarks.vn2.run_winning import run_winning
# MAGIC
# MAGIC summary = run_winning(
# MAGIC     data_dir="/dbfs/mnt/calibre/vn2-fixture",
# MAGIC     results_dir="/dbfs/mnt/calibre/results",
# MAGIC     verbose=True,
# MAGIC )
# MAGIC display(summary)
# MAGIC ```
