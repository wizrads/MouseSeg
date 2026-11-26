"""Train a MouseSeg 3D U-Net from scratch on CT scans and segmentation masks.

Example:
    python train.py --images ./data/images --labels ./data/labels \
        --mode unified --epochs 50 --output models/custom_unified.onnx
"""

import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mouseseg.model import build_model, export_onnx


def nifti_stem(path):
    """Return filename without .nii.gz or .nii extension."""
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else (name[:-4] if name.endswith(".nii") else path.stem)


def find_nifti_pairs(images_dir, labels_dir):
    """Match CT scans in images_dir with masks in labels_dir (supports <stem>_0000.nii.gz too)."""
    label_lookup = {}
    for path in sorted(Path(labels_dir).iterdir()):
        if not path.name.startswith(".") and path.name.endswith((".nii.gz", ".nii")):
            stem = nifti_stem(path)
            label_lookup[stem] = path
            for suffix in ("_seg", "_label", "_labels", "_mask"):
                if stem.endswith(suffix):
                    label_lookup.setdefault(stem[: -len(suffix)], path)

    pairs = []
    for image_path in sorted(Path(images_dir).iterdir()):
        if not image_path.name.startswith(".") and image_path.name.endswith((".nii.gz", ".nii")):
            stem = nifti_stem(image_path)
            label_path = label_lookup.get(stem) or (label_lookup.get(stem[:-5]) if stem.endswith("_0000") else None)
            if label_path is not None:
                pairs.append((image_path, label_path))
    if not pairs:
        raise FileNotFoundError(f"No matching NIfTI pairs found in {images_dir} and {labels_dir}.")
    return pairs


def reorient_to_ras(path):
    """Load a 3D NIfTI volume and reorient it to canonical RAS (Right-Anterior-Superior)."""
    img = nib.load(str(path))
    header_spacing = img.header.get_zooms()[:3]
    if np.allclose(np.abs(np.diag(img.affine)[:3]), 1.0, atol=1e-3) and not np.allclose(header_spacing, 1.0, atol=1e-3):
        img = nib.Nifti1Image(np.asanyarray(img.dataobj), np.diag([header_spacing[0], header_spacing[1], header_spacing[2], 1.0]))
    ras_img = nib.as_closest_canonical(img)
    volume = np.asarray(ras_img.get_fdata(), dtype=np.float32)
    while volume.ndim > 3 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    voxel_spacing_mm = ras_img.header.get_zooms()[:3]
    return volume, voxel_spacing_mm


def resample_volume_pair(image_path, label_path):
    """Resample CT (physical HU) and integer label mask to 0.20 mm RAS in (Z, X, Y) order."""
    ct_ras, voxel_spacing_mm = reorient_to_ras(image_path)
    label_ras, _ = reorient_to_ras(label_path)
    if -405.0 <= float(ct_ras.min()) <= -398.5:
        ct_ras[(ct_ras >= -405.0) & (ct_ras <= -398.5)] = -1000.0

    zoom_factors = [sp / 0.20 for sp in voxel_spacing_mm]
    ct_resampled = ndimage.zoom(ct_ras, zoom=zoom_factors, order=1, prefilter=False)
    label_resampled = ndimage.zoom(np.rint(label_ras).astype(np.int16), zoom=zoom_factors, order=0, prefilter=False)
    ct_volume = np.ascontiguousarray(np.transpose(ct_resampled, (2, 0, 1)), dtype=np.float32)
    label_volume = np.ascontiguousarray(np.transpose(label_resampled, (2, 0, 1)), dtype=np.int16)
    return ct_volume, label_volume


def normalize_hu(ct_volume, mode):
    """Anchor soft tissue toward +45 HU and linearly scale HU windows to [0, 1]."""
    hu = np.nan_to_num(ct_volume.astype(np.float32), nan=-1000.0, posinf=3000.0, neginf=-1000.0)
    soft_tissue = (hu >= -60.0) & (hu <= 160.0)
    if np.count_nonzero(soft_tissue) >= 64:
        shift = np.clip(45.0 - float(hu[soft_tissue].mean()), -25.0, 25.0)
        hu = hu + np.clip((hu + 350.0) / 150.0, 0.0, 1.0) * shift
    hu_windows = [(-1000.0, 3000.0), (-400.0, 1000.0)] if mode == "unified" else [(-400.0, 1000.0)]
    return np.stack([np.clip((hu - lo) / (hi - lo), 0.0, 1.0) for lo, hi in hu_windows], axis=0).astype(np.float32)


class MouseTrainDataset(Dataset):
    """Random 3D patch dataset with flips and physical HU contrast/shift/noise augmentation."""

    def __init__(self, volumes, patch_size, mode, patches_per_scan=4, augment=True):
        self.volumes = volumes
        self.patch_size = patch_size
        self.mode = mode
        self.patches_per_scan = patches_per_scan
        self.augment = augment
        self.foreground_coords = [np.argwhere(label_vol > 0) for _, label_vol in volumes]

    def __len__(self):
        return len(self.volumes) * self.patches_per_scan

    def __getitem__(self, index):
        scan_index = index % len(self.volumes)
        hu_vol, label_vol = self.volumes[scan_index]
        patch_d, patch_h, patch_w = self.patch_size

        pad_d = max(0, patch_d - hu_vol.shape[0])
        pad_h = max(0, patch_h - hu_vol.shape[1])
        pad_w = max(0, patch_w - hu_vol.shape[2])
        if pad_d or pad_h or pad_w:
            hu_vol = np.pad(hu_vol, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=-1000.0)
            label_vol = np.pad(label_vol, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")

        fg = self.foreground_coords[scan_index]
        if len(fg) > 0 and random.random() < 0.5:
            cz, cx, cy = fg[random.randrange(len(fg))]
            z0 = int(np.clip(cz - patch_d // 2, 0, hu_vol.shape[0] - patch_d))
            x0 = int(np.clip(cx - patch_h // 2, 0, hu_vol.shape[1] - patch_h))
            y0 = int(np.clip(cy - patch_w // 2, 0, hu_vol.shape[2] - patch_w))
        else:
            z0 = random.randint(0, hu_vol.shape[0] - patch_d)
            x0 = random.randint(0, hu_vol.shape[1] - patch_h)
            y0 = random.randint(0, hu_vol.shape[2] - patch_w)

        hu_patch = hu_vol[z0 : z0 + patch_d, x0 : x0 + patch_h, y0 : y0 + patch_w].copy()
        label_patch = label_vol[z0 : z0 + patch_d, x0 : x0 + patch_h, y0 : y0 + patch_w].copy()

        if self.augment:
            flip_axes = (0, 1, 2) if self.mode == "tumor_only" else (0, 2)
            for axis in flip_axes:
                if random.random() < 0.5:
                    hu_patch = np.flip(hu_patch, axis=axis)
                    label_patch = np.flip(label_patch, axis=axis)
            hu_patch = hu_patch * random.uniform(0.90, 1.10) + random.uniform(-25.0, 25.0)
            noise_std = random.uniform(0.0, 25.0)
            if noise_std > 0:
                hu_patch = hu_patch + np.random.normal(0.0, noise_std, size=hu_patch.shape).astype(np.float32)

        ct_patch = torch.from_numpy(normalize_hu(hu_patch, self.mode))
        if self.mode == "tumor_only":
            tumor_mask = (label_patch == 12) if np.any(label_patch == 12) else (label_patch > 0)
            organ_mask = np.zeros_like(label_patch, dtype=np.int64)
        else:
            tumor_mask = label_patch == 12
            organ_mask = np.where((label_patch >= 1) & (label_patch <= 11), label_patch, 0).astype(np.int64)
        return ct_patch, torch.from_numpy(np.ascontiguousarray(organ_mask)), torch.from_numpy(np.ascontiguousarray(tumor_mask, dtype=np.float32))


def compute_loss(organ_logits, tumor_logits, organ_target, tumor_target, supervise_organs, supervise_tumor):
    """Combined Cross-Entropy + Soft Dice for organs and BCE + Soft Dice for tumor."""
    loss = tumor_logits.sum() * 0.0
    if supervise_organs:
        ce_loss = F.cross_entropy(organ_logits, organ_target)
        organ_probs = torch.softmax(organ_logits, dim=1)
        dice_terms = []
        for class_id in range(1, organ_logits.shape[1]):
            gt_class = (organ_target == class_id).float()
            if gt_class.sum() > 0:
                intersection = (organ_probs[:, class_id] * gt_class).sum()
                dice_terms.append(1.0 - (2.0 * intersection + 1.0) / (organ_probs[:, class_id].sum() + gt_class.sum() + 1.0))
        loss = loss + ce_loss + (torch.stack(dice_terms).mean() if dice_terms else 0.0)

    if supervise_tumor:
        tumor_logits_3d = tumor_logits.squeeze(1)
        bce_loss = F.binary_cross_entropy_with_logits(tumor_logits_3d, tumor_target)
        tumor_probs = torch.sigmoid(tumor_logits_3d)
        intersection = (tumor_probs * tumor_target).sum()
        loss = loss + bce_loss + (1.0 - (2.0 * intersection + 1.0) / (tumor_probs.sum() + tumor_target.sum() + 1.0))
    return loss


def main():
    parser = argparse.ArgumentParser(description="Train a MouseSeg 3D U-Net from scratch.")
    parser.add_argument("--images", type=Path, required=True, help="Folder of CT NIfTI scans (.nii or .nii.gz)")
    parser.add_argument("--labels", type=Path, required=True, help="Folder of matching label NIfTI masks")
    parser.add_argument("--mode", choices=["unified", "tumor_only"], default="unified")
    parser.add_argument("--output", type=Path, default=Path("models/trained_unified.onnx"), help="Output .onnx path")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Output .pt checkpoint path (default: matches --output)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 128, 128])
    parser.add_argument("--patches-per-scan", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or args.output.with_suffix(".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    patch_size = [max(16, ((s + 15) // 16) * 16) for s in args.patch_size]

    pairs = find_nifti_pairs(args.images, args.labels)
    print(f"Found {len(pairs)} scan/label pair(s). Resampling to 0.20 mm RAS...")
    volumes = [resample_volume_pair(img_p, lbl_p) for img_p, lbl_p in pairs]

    n_val = max(1, int(round(len(volumes) * args.val_fraction))) if len(volumes) >= 4 and args.val_fraction > 0 else 0
    train_vols = volumes[:-n_val] if n_val > 0 else volumes
    val_vols = volumes[-n_val:] if n_val > 0 else []

    supervise_organs = args.mode == "unified" and any(np.any((label_vol >= 1) & (label_vol <= 11)) for _, label_vol in train_vols)
    supervise_tumor = args.mode == "tumor_only" or any(np.any(label_vol == 12) for _, label_vol in train_vols)

    device = torch.device(args.device)
    model = build_model(args.mode, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-5)

    train_loader = DataLoader(MouseTrainDataset(train_vols, patch_size, args.mode, args.patches_per_scan, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MouseTrainDataset(val_vols, patch_size, args.mode, 2, augment=False), batch_size=1) if val_vols else None

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for ct_patch, organ_target, tumor_target in train_loader:
            ct_patch = ct_patch.to(device)
            organ_target = organ_target.to(device)
            tumor_target = tumor_target.to(device)

            optimizer.zero_grad()
            organ_logits, tumor_logits = model(ct_patch)
            loss = compute_loss(organ_logits, tumor_logits, organ_target, tumor_target, supervise_organs, supervise_tumor)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        scheduler.step()
        train_loss /= len(train_loader)

        eval_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for ct_patch, organ_target, tumor_target in val_loader:
                    organ_logits, tumor_logits = model(ct_patch.to(device))
                    val_total += float(
                        compute_loss(organ_logits, tumor_logits, organ_target.to(device), tumor_target.to(device), supervise_organs, supervise_tumor).item()
                    )
            eval_loss = val_total / len(val_loader)

        if eval_loss <= best_loss:
            best_loss = eval_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
        print(f"Epoch {epoch:03d}/{args.epochs:03d}  train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    export_onnx(model, args.output, args.mode)
    print(f"Saved best PyTorch weights -> {checkpoint_path}")
    print(f"Exported ONNX model        -> {args.output}")


if __name__ == "__main__":
    main()
