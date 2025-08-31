import argparse
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

from utils import ensure_dir


def load_dataset(data_dir: Path) -> datasets.ImageFolder:
    tfm = transforms.ToTensor()
    return datasets.ImageFolder(root=str(data_dir), transform=tfm)


def plot_class_distribution(dataset: datasets.ImageFolder, output_dir: Path) -> Path:
    # ImageFolder exposes samples as list of (path, class_idx)
    label_indices = [label for _, label in dataset.samples]
    counts = Counter(label_indices)
    labels = [dataset.classes[i] for i in sorted(counts.keys())]
    values = [counts[i] for i in sorted(counts.keys())]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="#4c78a8")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = output_dir / "class_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_sample_grid(dataset: datasets.ImageFolder, output_dir: Path, rows: int = 3, cols: int = 6) -> Path:
    total = rows * cols
    indices = random.sample(range(len(dataset)), k=min(total, len(dataset)))
    images: List[np.ndarray] = []
    titles: List[str] = []
    for idx in indices:
        img_t, y = dataset[idx]
        images.append(img_t.permute(1, 2, 0).numpy())
        titles.append(dataset.classes[y])

    n = len(images)
    grid_rows = rows
    grid_cols = int(np.ceil(n / rows)) if n < total else cols

    plt.figure(figsize=(grid_cols * 3, grid_rows * 3))
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(grid_rows, grid_cols, i + 1)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    out_path = output_dir / "sample_grid.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def compute_image_size_stats(dataset: datasets.ImageFolder, sample_size: int = 512) -> Tuple[float, float]:
    # Sample a subset of file paths to avoid loading all
    subset = random.sample(dataset.samples, k=min(sample_size, len(dataset.samples)))
    widths: List[int] = []
    heights: List[int] = []
    for path, _ in subset:
        try:
            with Image.open(path) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue
    mean_w = float(np.mean(widths)) if widths else 0.0
    mean_h = float(np.mean(heights)) if heights else 0.0
    return mean_w, mean_h


def compute_class_balance(dataset: datasets.ImageFolder) -> Tuple[List[str], List[int]]:
    counts = Counter([y for _, y in dataset.samples])
    labels = [dataset.classes[i] for i in sorted(counts.keys())]
    values = [counts[i] for i in sorted(counts.keys())]
    return labels, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset EDA for garbage classification")
    parser.add_argument("--data_dir", type=str, default=str(Path("dataset") / "garbage_classification"))
    parser.add_argument("--output_dir", type=str, default=str(Path("outputs")))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    ds = load_dataset(data_dir)
    print(f"Found {len(ds)} images across {len(ds.classes)} classes: {ds.classes}")

    dist_path = plot_class_distribution(ds, output_dir)
    grid_path = plot_sample_grid(ds, output_dir)
    mean_w, mean_h = compute_image_size_stats(ds)
    print(f"Saved class distribution to: {dist_path}")
    print(f"Saved sample grid to: {grid_path}")
    print(f"Approx. mean image size: {mean_w:.1f} x {mean_h:.1f} (W x H)")


if __name__ == "__main__":
    main()


