"""End-to-end pipeline entrypoint.

Runs every stage in order, writes ocean_cube.zarr + QC outputs.
"""
from __future__ import annotations

from export_csv import main as export_csv
from pipeline.stage6_assemble import assemble
from pipeline.stage7_qc import run as run_qc
from pipeline.utils import get_logger

log = get_logger("run_pipeline")


def main() -> None:
    ds = assemble()
    run_qc(ds)
    export_csv()
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
