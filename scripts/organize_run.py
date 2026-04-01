"""Move heavy files (pickles, checkpoints) into _heavy/ and leave symlinks.

Also mirrors results/ and logs/ into a top-level data/{results,logs,_heavy}/<run_name>/
structure so users can download only the top-level results/ folder to get all runs.

Usage: python scripts/organize_run.py <run_dir>

After running:
  <run_dir>/                    (original, symlinks to _heavy for heavy files)
  data/results/<run_name>/      (symlink to <run_dir>/results/)
  data/logs/<run_name>/         (symlink to <run_dir>/logs/)
  data/_heavy/<run_name>/       (symlink to <run_dir>/_heavy/)
"""

import shutil
import sys
from pathlib import Path

HEAVY_EXTENSIONS = {".pkl", ".pt"}
HEAVY_DIRS = {"checkpoints"}


def organize(run_dir: Path):
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
            print(f"  moved dir  {p.name}/ -> _heavy/{p.name}/")

        elif p.is_file() and p.suffix in HEAVY_EXTENSIONS:
            dest = heavy_dir / p.name
            shutil.move(str(p), str(dest))
            p.symlink_to(dest.resolve())
            print(f"  moved file {p.name} -> _heavy/{p.name}")

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
                print(f"  moved file {rel} -> _heavy/{rel}")

    light_size = sum(
        f.stat().st_size
        for f in run_dir.rglob("*")
        if f.is_file() and not f.is_symlink() and "_heavy" not in f.parts
    )
    heavy_size = sum(f.stat().st_size for f in heavy_dir.rglob("*") if f.is_file())
    print(f"\n  Download folder: {light_size / 1e6:.1f} MB")
    print(f"  Heavy (excluded): {heavy_size / 1e6:.1f} MB")

    _mirror_to_top_level(run_dir)


def _mirror_to_top_level(run_dir: Path):
    """Create top-level data/{results,logs,_heavy}/<run_name>/ symlinks.

    This lets users download only data/results/ to get all run outputs.
    """
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
        print(f"  mirrored {folder_name}/{run_name}/ -> {src}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir>")
        sys.exit(1)
    rd = Path(sys.argv[1])
    if not rd.exists():
        print(f"Not found: {rd}")
        sys.exit(1)
    print(f"Organizing {rd}...")
    organize(rd)
    print("Done.")
