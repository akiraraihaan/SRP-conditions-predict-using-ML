"""Sample N random images from FINAL-pipeline/dataset (all classes pooled together).

Sampling uses random.SystemRandom (OS entropy, not the reproducible Mersenne
Twister), so the selection is genuinely random on every run.
"""

import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent.parent / "FINAL-pipeline" / "dataset"
OUTPUT_DIR = BASE_DIR / "output"


def collect_images(dataset_dir: Path) -> list[Path]:
    """Collect every image file across all class subfolders."""
    return sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_random_images(
    count: int = 100,
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = OUTPUT_DIR,
    clean: bool = True,
) -> list[Path]:
    """Copy `count` randomly chosen images from the dataset into the output folder.

    Returns: list of paths to the copied files.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    all_images = collect_images(dataset_dir)
    if not all_images:
        raise RuntimeError(f"No image files found in {dataset_dir}")
    if len(all_images) < count:
        raise ValueError(
            f"Only {len(all_images)} images available, not enough for {count}."
        )

    rng = random.SystemRandom()
    selected = rng.sample(all_images, count)

    if clean and output_dir.is_dir():
        for stale in output_dir.iterdir():
            if stale.is_file():
                stale.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for i, src in enumerate(selected, start=1):
        class_name = src.relative_to(dataset_dir).parts[0]
        dst = output_dir / f"{i:03d}__{class_name}__{src.name}"
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


if __name__ == "__main__":
    from collections import Counter

    copied = sample_random_images(100)
    print(f"Copied {len(copied)} images to: {OUTPUT_DIR}")
    print("\nClass distribution in the sample:")
    for class_name, n in sorted(Counter(p.name.split("__")[1] for p in copied).items()):
        print(f"  {class_name:45s} {n}")
