#!/usr/bin/env python3
"""
train_maskrcnn.py — Fine-tune Mask R-CNN on bubble instance masks.

Uses torchvision's Mask R-CNN (ResNet-50 + FPN) starting from COCO pre-trained
weights, following the architecture in Cui et al. 2022 and Kim & Park 2021.

Note on BubMask weights: the pre-staged mask_rcnn_bubble.h5 (Kim & Park 2021)
is in TensorFlow/Keras format and cannot be loaded directly into torchvision.
This script instead uses PyTorch torchvision COCO weights as the starting point.
Converting the TF weights to PyTorch is a separate task.

All dependencies are already in bubbly-train-env (torch + torchvision).

Interface matches the manage_bubbly.py training contract:
    --dataset PATH    root with images/ and labels/ subdirectories
    --name    STR     experiment name (used for checkpoint folder)
    --epochs  INT     training epochs (default 50)
    --save_root PATH  where to save checkpoints (passed by manage_bubbly.py)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True, type=Path)
    p.add_argument("--name",      required=True, type=str)
    p.add_argument("--epochs",    type=int, default=50)
    p.add_argument("--save_root", type=Path, default=None)
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────
class BubbleDataset(Dataset):
    """
    Loads images as float tensors [C, H, W] in [0,1].
    Converts uint16 instance masks into per-instance boxes, labels, and masks
    in the format expected by torchvision Mask R-CNN.
    """

    def __init__(self, image_paths: list, labels_dir: Path):
        self.image_paths = image_paths
        self.labels_dir  = labels_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import cv2
        import tifffile

        img_path = self.image_paths[idx]
        lbl_path = self.labels_dir / (img_path.stem + ".tif")

        # Image → float tensor [C, H, W] in [0, 1]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        img = (img.astype(np.float32) / np.iinfo(img.dtype).max
               if img.dtype != np.float32 else img)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # [C, H, W]

        # Label mask → per-instance arrays
        mask = tifffile.imread(str(lbl_path)).astype(np.int32)
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]

        boxes, labels, masks = [], [], []
        for iid in instance_ids:
            m = (mask == iid)
            rows = np.where(m.any(axis=1))[0]
            cols = np.where(m.any(axis=0))[0]
            if rows.size == 0 or cols.size == 0:
                continue
            x1, y1 = int(cols[0]), int(rows[0])
            x2, y2 = int(cols[-1]) + 1, int(rows[-1]) + 1
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(1)          # class 1 = bubble
            masks.append(m)

        if not boxes:
            # No instances: return empty tensors (torchvision handles this)
            H, W = mask.shape
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)
            masks_t  = torch.zeros((0, H, W), dtype=torch.bool)
            area_t   = torch.zeros(0, dtype=torch.float32)
            crowd_t  = torch.zeros(0, dtype=torch.int64)
        else:
            boxes_arr = np.array(boxes, dtype=np.float32)
            boxes_t  = torch.from_numpy(boxes_arr)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            masks_t  = torch.from_numpy(np.stack(masks, axis=0))
            area_t   = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            crowd_t  = torch.zeros(len(boxes), dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "masks":    masks_t,
            "image_id": torch.tensor([idx]),
            "area":     area_t,
            "iscrowd":  crowd_t,
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes=2):
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace box predictor head (background + bubble)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def evaluate_loss(model, loader, device):
    """Compute mean training loss in eval mode (for checkpoint decisions)."""
    model.train()  # loss dict only available in train mode
    total = 0.0
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total += sum(v.item() for v in loss_dict.values())
            count += 1
    return total / max(count, 1)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.save_root:
        save_dir = args.save_root / args.name
    else:
        save_dir = Path(__file__).resolve().parent.parent / "microsam" / "models" / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"STARTING Mask R-CNN TRAINING: {args.name}")
    print(f"Dataset:  {args.dataset}")
    print(f"Epochs:   {args.epochs}")
    print(f"Output:   {save_dir}")
    print("--------------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be extremely slow.")

    # ── Load data ─────────────────────────────────────────────────────────────
    images_dir = args.dataset / "images"
    labels_dir = args.dataset / "labels"

    image_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.tif")))
    if not image_paths:
        print("ERROR: No images found in images/")
        sys.exit(1)

    # Filter to those with a paired label
    image_paths = [p for p in image_paths if (labels_dir / (p.stem + ".tif")).exists()]
    if not image_paths:
        print("ERROR: No image/label pairs found.")
        sys.exit(1)

    print(f"Found {len(image_paths)} image/label pairs.")

    split_idx = max(1, int(len(image_paths) * 0.9))
    train_paths = image_paths[:split_idx]
    val_paths   = image_paths[split_idx:]
    print(f"Split: {len(train_paths)} train / {len(val_paths)} val")

    train_ds = BubbleDataset(train_paths, labels_dir)
    val_ds   = BubbleDataset(val_paths,   labels_dir)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # ── Build model ───────────────────────────────────────────────────────────
    print("Loading Mask R-CNN (ResNet-50 + FPN, COCO pretrained) ...")
    model = build_model(num_classes=2).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    # ── Train ─────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_path = save_dir / "best.pt"
    last_path  = save_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        lr_scheduler.step()
        avg_train = epoch_loss / max(len(train_loader), 1)

        val_loss = evaluate_loss(model, val_loader, device)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        torch.save(model.state_dict(), last_path)
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={val_loss:.4f}"
              + ("  [best]" if improved else ""))

    print("--------------------------------")
    print("TRAINING COMPLETE.")
    print(f"Best model: {best_path}  (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
