import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training and visualization pipeline")
    parser.add_argument("--data_dir", type=str, default=str(Path("dataset") / "garbage_classification"))
    parser.add_argument("--output_dir", type=str, default=str(Path("outputs")))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--backbones", type=str, nargs="*", default=["resnet18"], help="List of backbones to train")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    # Run EDA first to generate basic artifacts
    run([
        "python",
        str(project_root / "src" / "eda.py"),
        "--data_dir",
        args.data_dir,
        "--output_dir",
        args.output_dir,
    ])

    # Train one or multiple backbones sequentially
    for backbone in args.backbones:
        out_dir_bb = str(Path(args.output_dir) / backbone)
        train_cmd = [
            "python",
            str(project_root / "src" / "train.py"),
            "--data_dir",
            args.data_dir,
            "--output_dir",
            out_dir_bb,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--image_size",
            str(args.image_size),
            "--backbone",
            backbone,
            "--use_amp",
        ]
        run(train_cmd)

    # Visualize for each backbone
    for backbone in args.backbones:
        viz_cmd = [
            "python",
            str(project_root / "src" / "visualize.py"),
            "--artifacts_dir",
            str(Path(args.output_dir) / backbone),
        ]
        run(viz_cmd)


if __name__ == "__main__":
    main()


