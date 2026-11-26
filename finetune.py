"""Fine-tune a pretrained MouseSeg 3D model on scans from your own CT scanner.

Example:
    python finetune.py --images ./my_data/images --labels ./my_data/labels \
        --mode unified --output models/my_scanner_unified.onnx
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

REPO_DIR = Path(__file__).resolve().parent


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


def preprocess_pair(image_path, label_path, mode):
    """Resample CT and label to 0.20 mm RAS, transpose to (Z, X, Y), and normalize HU windows."""
    ct_ras, voxel_spacing_mm = reorient_to_ras(image_path)
    label_ras, _ = reorient_to_ras(label_path)

    # Restore -400 HU pre-clipped outside air to -1000 HU before resampling
    if -405.0 <= float(ct_ras.min()) <= -398.5:
        ct_ras[(ct_ras >= -405.0) & (ct_ras <= -398.5)] = -1000.0

    zoom_factors = [sp / 0.20 for sp in voxel_spacing_mm]
    ct_volume = np.transpose(ndimage.zoom(ct_ras, zoom=zoom_factors, order=1, prefilter=False), (2, 0, 1))
    label_volume = np.ascontiguousarray(
        np.transpose(ndimage.zoom(np.rint(label_ras).astype(np.int16), zoom=zoom_factors, order=0, prefilter=False), (2, 0, 1))
    )

    # Anchor soft tissue toward +45 HU and scale HU windows linearly to [0, 1]
    hu = np.nan_to_num(ct_volume.astype(np.float32), nan=-1000.0, posinf=3000.0, neginf=-1000.0)
    soft_tissue = (hu >= -60.0) & (hu <= 160.0)
    if np.count_nonzero(soft_tissue) >= 64:
        shift = np.clip(45.0 - float(hu[soft_tissue].mean()), -25.0, 25.0)
        hu = hu + np.clip((hu + 350.0) / 150.0, 0.0, 1.0) * shift
    hu_windows = [(-1000.0, 3000.0), (-400.0, 1000.0)] if mode == "unified" else [(-400.0, 1000.0)]
    normalized_ct = np.stack([np.clip((hu - lo) / (hi - lo), 0.0, 1.0) for lo, hi in hu_windows], axis=0).astype(np.float32)
    return normalized_ct, label_volume


class MousePatchDataset(Dataset):
    """Random 3D patch dataset sampling foreground-centered and random crops."""

    def __init__(self, volumes, patch_size, mode, patches_per_scan=4):
        self.volumes = volumes
        self.patch_size = patch_size
        self.mode = mode
        self.patches_per_scan = patches_per_scan
        self.foreground_coords = [np.argwhere(label_vol > 0) for _, label_vol in volumes]

    def __len__(self):
        return len(self.volumes) * self.patches_per_scan

    def __getitem__(self, index):
        scan_index = index % len(self.volumes)
        image_vol, label_vol = self.volumes[scan_index]
        patch_d, patch_h, patch_w = self.patch_size

        pad_d = max(0, patch_d - image_vol.shape[1])
        pad_h = max(0, patch_h - image_vol.shape[2])
        pad_w = max(0, patch_w - image_vol.shape[3])
        if pad_d or pad_h or pad_w:
            image_vol = np.pad(image_vol, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
            label_vol = np.pad(label_vol, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")

        fg = self.foreground_coords[scan_index]
        if len(fg) > 0 and random.random() < 0.5:
            cz, cx, cy = fg[random.randrange(len(fg))]
            z0 = int(np.clip(cz - patch_d // 2, 0, image_vol.shape[1] - patch_d))
            x0 = int(np.clip(cx - patch_h // 2, 0, image_vol.shape[2] - patch_h))
            y0 = int(np.clip(cy - patch_w // 2, 0, image_vol.shape[3] - patch_w))
        else:
            z0 = random.randint(0, image_vol.shape[1] - patch_d)
            x0 = random.randint(0, image_vol.shape[2] - patch_h)
            y0 = random.randint(0, image_vol.shape[3] - patch_w)

        ct_patch = torch.from_numpy(image_vol[:, z0 : z0 + patch_d, x0 : x0 + patch_h, y0 : y0 + patch_w].copy())
        label_patch = label_vol[z0 : z0 + patch_d, x0 : x0 + patch_h, y0 : y0 + patch_w]
        if self.mode == "tumor_only":
            tumor_mask = (label_patch == 12) if np.any(label_patch == 12) else (label_patch > 0)
            organ_mask = np.zeros_like(label_patch, dtype=np.int64)
        else:
            tumor_mask = label_patch == 12
            organ_mask = np.where((label_patch >= 1) & (label_patch <= 11), label_patch, 0).astype(np.int64)
        return ct_patch, torch.from_numpy(organ_mask), torch.from_numpy(tumor_mask.astype(np.float32))


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
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained MouseSeg 3D model on your own CT scans.")
    parser.add_argument("--images", type=Path, required=True, help="Folder of CT NIfTI scans (.nii or .nii.gz)")
    parser.add_argument("--labels", type=Path, required=True, help="Folder of matching label NIfTI masks")
    parser.add_argument("--mode", choices=["unified", "tumor_only"], default="unified")
    parser.add_argument("--pretrained", type=Path, default=None, help="Optional custom .pt weights path")
    parser.add_argument("--output", type=Path, default=Path("models/my_scanner_unified.onnx"), help="Output .onnx path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 128, 128])
    parser.add_argument("--patches-per-scan", type=int, default=4)
    parser.add_argument("--unfreeze-all", action="store_true", help="Update all encoder stages instead of freezing early layers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    default_pt = "mouseseg_unified.pt" if args.mode == "unified" else "mouseseg_tumor_specialist.pt"
    weights_path = args.pretrained or (REPO_DIR / "models" / default_pt)
    patch_size = [max(16, ((s + 15) // 16) * 16) for s in args.patch_size]

    pairs = find_nifti_pairs(args.images, args.labels)
    print(f"Found {len(pairs)} scan/label pair(s). Resampling to 0.20 mm RAS...")
    volumes = [preprocess_pair(img_p, lbl_p, args.mode) for img_p, lbl_p in pairs]

    supervise_organs = args.mode == "unified" and any(np.any((label_vol >= 1) & (label_vol <= 11)) for _, label_vol in volumes)
    supervise_tumor = args.mode == "tumor_only" or any(np.any(label_vol == 12) for _, label_vol in volumes)

    model = build_model(args.mode)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)

    frozen_modules = [] if args.unfreeze_all else [model.stem, model.enc0, model.encoders[0]]
    for module in frozen_modules:
        module.requires_grad_(False)

    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(MousePatchDataset(volumes, patch_size, args.mode, args.patches_per_scan), batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for module in frozen_modules:
            module.eval()
        epoch_loss = 0.0
        for ct_patch, organ_target, tumor_target in loader:
            ct_patch = ct_patch.to(device)
            organ_target = organ_target.to(device)
            tumor_target = tumor_target.to(device)

            optimizer.zero_grad()
            organ_logits, tumor_logits = model(ct_patch)
            loss = compute_loss(organ_logits, tumor_logits, organ_target, tumor_target, supervise_organs, supervise_tumor)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        print(f"Epoch {epoch:02d}/{args.epochs:02d}  loss={epoch_loss / len(loader):.4f}")

    export_onnx(model, args.output, args.mode)
    print(f"Exported fine-tuned ONNX model -> {args.output}")


if __name__ == "__main__":
    main()
