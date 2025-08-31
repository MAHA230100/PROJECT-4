import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

from utils import ensure_dir, save_json, set_seed
from config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_IMAGE_SIZE


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, val_tfms


def create_dataloaders(data_dir: Path, image_size: int, batch_size: int, num_workers: int, val_split: float) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_tfms, val_tfms = build_transforms(image_size)
    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
    class_names = full_dataset.classes

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Apply validation transforms to val subset
    val_dataset.dataset.transform = val_tfms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int, backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, np.array(all_targets), np.array(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a garbage image classifier (ResNet-18)")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0", "baseline_cnn"])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience based on val F1 (macro)")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    model = build_model(num_classes=len(class_names), backbone=args.backbone, pretrained=True).to(device)

    # Compute class weights to mitigate imbalance
    class_counts = np.zeros(len(class_names), dtype=np.int64)
    # full_dataset is inside create_dataloaders, so rebuild lightweight ImageFolder to count
    count_ds = datasets.ImageFolder(root=str(data_dir))
    for _, y in count_ds.samples:
        class_counts[y] += 1
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    class_weights = class_weights * (len(class_names) / class_weights.sum())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1_macro": []}

    best_val_f1 = -1.0
    best_model_path = output_dir / "best_model.pth"
    epochs_without_improve = 0

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    save_json(class_to_idx, output_dir / "class_to_idx.json")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        # Train epoch (with optional AMP)
        model.train()
        running_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        for images, targets in tqdm(train_loader, desc="Train", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)
        val_f1_macro = f1_score(y_true, y_pred, average="macro")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(float(val_f1_macro))

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1(macro): {val_f1_macro:.4f}"
        )

        if val_f1_macro > best_val_f1:
            best_val_f1 = float(val_f1_macro)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "backbone": args.backbone,
            }, best_model_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"Early stopping triggered. No improvement in {args.patience} epochs.")
            break

        # Save per-epoch predictions for last epoch only
        if epoch == args.epochs:
            preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            preds_df.to_csv(output_dir / "val_predictions.csv", index=False)

    # Save overall history and final reports
    save_json(history, output_dir / "training_history.json")

    # Final evaluation report and confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(report).to_csv(output_dir / "classification_report.csv")

    print(f"Best val F1(macro): {best_val_f1:.4f}. Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()


