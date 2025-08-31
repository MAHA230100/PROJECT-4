import argparse
from pathlib import Path
from typing import Optional

from PIL import Image


def process_image(input_path: Path, output_path: Path, size: int = 224, quality: int = 90) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as im:
        im = im.convert("RGB")
        im = im.resize((size, size), Image.BILINEAR)
        im.save(output_path, format="JPEG", quality=quality, optimize=True)


def preprocess_folder(input_dir: Path, output_dir: Path, size: int = 224, quality: int = 90, limit: Optional[int] = None) -> None:
    count = 0
    for class_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        for img_path in class_dir.rglob("*.jpg"):
            rel = img_path.relative_to(input_dir)
            out_path = output_dir / rel
            process_image(img_path, out_path, size=size, quality=quality)
            count += 1
            if limit is not None and count >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline preprocessing: resize/compress dataset")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    preprocess_folder(Path(args.input_dir), Path(args.output_dir), size=args.size, quality=args.quality, limit=args.limit)


if __name__ == "__main__":
    main()


