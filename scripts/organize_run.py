"""Move heavy log artifacts into data/_heavy/<run_name>/ and symlink back."""

import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from burst.core.train_utils import resolve_logs_dir, resolve_results_dir

logger = logging.getLogger(__name__)

EXPECTED_PARTS = 2
HEAVY_EXTENSIONS = {".pkl", ".pt"}
HEAVY_DIRS = {"checkpoints"}


def organize(run_dir: Path) -> None:
    """Move heavy files from logs into _heavy/ and symlink back."""
    data_dir = run_dir.parent
    run_name = run_dir.name
    logs_dir = resolve_logs_dir(run_dir)
    results_dir = resolve_results_dir(run_dir)
    heavy_dir = data_dir / "_heavy" / run_name
    heavy_dir.mkdir(parents=True, exist_ok=True)

    for src_dir in (logs_dir, results_dir):
        if not src_dir.exists():
            continue
        _move_heavy_in(src_dir, heavy_dir, src_dir)

    light_size = sum(
        _file_size(f)
        for d in (logs_dir, results_dir)
        if d.exists()
        for f in d.rglob("*")
        if f.is_file() and not f.is_symlink()
    )
    heavy_size = sum(f.stat().st_size for f in heavy_dir.rglob("*") if f.is_file())
    logger.info("\n  Download folder: %.1f MB", light_size / 1e6)
    logger.info("  Heavy (excluded): %.1f MB", heavy_size / 1e6)


def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _move_heavy_in(root: Path, heavy_dir: Path, base: Path) -> None:
    """Recursively move heavy files/dirs from root into heavy_dir, symlinking back."""
    for p in sorted(root.iterdir()):
        if p.is_symlink():
            continue
        if p.is_dir() and p.name in HEAVY_DIRS:
            rel = p.relative_to(base)
            dest = heavy_dir / rel
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            p.symlink_to(dest.resolve())
            logger.info("  moved dir  %s/ -> _heavy/%s/", rel, rel)
        elif p.is_dir():
            _move_heavy_in(p, heavy_dir, base)
        elif p.is_file() and p.suffix in HEAVY_EXTENSIONS:
            rel = p.relative_to(base)
            dest = heavy_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            p.symlink_to(dest.resolve())
            logger.info("  moved file %s -> _heavy/%s", rel, rel)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < EXPECTED_PARTS:
        logger.info("Usage: %s <run_dir>", sys.argv[0])
        sys.exit(1)
    rd = Path(sys.argv[1])
    logs_exists = resolve_logs_dir(rd).exists()
    results_exists = resolve_results_dir(rd).exists()
    if not logs_exists and not results_exists:
        logger.info("Not found: %s (no logs or results dirs)", rd)
        sys.exit(1)
    logger.info("Organizing %s...", rd)
    organize(rd)
    logger.info("Done.")
