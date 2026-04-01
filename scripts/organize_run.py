"""Move heavy run artifacts into _heavy/ and symlink results, logs, and _heavy under data/."""

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

EXPECTED_PARTS = 2
HEAVY_EXTENSIONS = {".pkl", ".pt"}
HEAVY_DIRS = {"checkpoints"}


def organize(run_dir: Path) -> None:
    """Move heavy files into _heavy/ and create top-level symlinks."""
    heavy_dir = run_dir / "_heavy"
    heavy_dir.mkdir(exist_ok=True)

    for p in sorted(run_dir.iterdir()):
        if p.name == "_heavy" or p.is_symlink():
            continue

        if p.is_dir() and p.name in HEAVY_DIRS:
            dest = heavy_dir / p.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(p), str(dest))
            p.symlink_to(dest.resolve())
            logger.info("  moved dir  %s/ -> _heavy/%s/", p.name, p.name)

        elif p.is_file() and p.suffix in HEAVY_EXTENSIONS:
            dest = heavy_dir / p.name
            shutil.move(str(p), str(dest))
            p.symlink_to(dest.resolve())
            logger.info("  moved file %s -> _heavy/%s", p.name, p.name)

    for sub in run_dir.iterdir():
        if sub.name.startswith("_") or not sub.is_dir() or sub.is_symlink():
            continue
        for p in sorted(sub.rglob("*")):
            if p.is_file() and not p.is_symlink() and p.suffix in HEAVY_EXTENSIONS:
                rel = p.relative_to(run_dir)
                dest = heavy_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
                p.symlink_to(dest.resolve())
                logger.info("  moved file %s -> _heavy/%s", rel, rel)

    light_size = sum(
        f.stat().st_size
        for f in run_dir.rglob("*")
        if f.is_file() and not f.is_symlink() and "_heavy" not in f.parts
    )
    heavy_size = sum(f.stat().st_size for f in heavy_dir.rglob("*") if f.is_file())
    logger.info("\n  Download folder: %.1f MB", light_size / 1e6)
    logger.info("  Heavy (excluded): %.1f MB", heavy_size / 1e6)

    _mirror_to_top_level(run_dir)


def _mirror_to_top_level(run_dir: Path) -> None:
    """Create top-level data/{results,logs,_heavy}/<run_name>/ symlinks."""
    data_dir = run_dir.parent
    run_name = run_dir.name

    for folder_name in ("results", "logs", "_heavy"):
        src = run_dir / folder_name
        if not src.exists():
            continue
        top_level_parent = data_dir / folder_name
        top_level_parent.mkdir(exist_ok=True)
        link = top_level_parent / run_name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())
        logger.info("  mirrored %s/%s/ -> %s", folder_name, run_name, src)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < EXPECTED_PARTS:
        logger.info("Usage: %s <run_dir>", sys.argv[0])
        sys.exit(1)
    rd = Path(sys.argv[1])
    if not rd.exists():
        logger.info("Not found: %s", rd)
        sys.exit(1)
    logger.info("Organizing %s...", rd)
    organize(rd)
    logger.info("Done.")
