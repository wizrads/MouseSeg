#!/usr/bin/env python3
"""Command-line tool for pure-CPU 3D mouse CT organ and tumor segmentation.

Examples:
    python segment_mouse.py -i mouse_ct.nii.gz -o mouse_seg.nii.gz
    python segment_mouse.py -i mouse_ct.nii.gz -o mouse_seg.nii.gz --body-mask mouse_body.nii.gz --csv volumes.csv
    python segment_mouse.py -i ./scans_dir -o ./segs_dir --mode tumor_only --body-mask --csv ./segs_dir/volumes.csv
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from mouseseg import load_onnx_model, segment_ct_volume  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "unified": REPO_DIR / "models" / "mouseseg_unified_int8.onnx",
    "tumor_only": REPO_DIR / "models" / "mouseseg_tumor_specialist_int8.onnx",
}
SKIP_SUFFIXES = ("_seg.nii.gz", "_seg.nii", "_body_mask.nii.gz", "_body_mask.nii")


def strip_nifti_ext(filename):
    """Strip .nii.gz or .nii extension from a filename."""
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return filename[:-7]
    if lower.endswith(".nii") or lower.endswith(".hdr"):
        return filename[:-4]
    return Path(filename).stem


def collect_scan_jobs(input_arg, output_arg):
    """Return `(jobs, is_batch)` where each job is `(input_path, output_mask_path)`."""
    in_path = Path(input_arg)
    out_path = Path(output_arg)
    if in_path.is_dir():
        scans = [
            p for p in sorted(in_path.iterdir())
            if p.is_file()
            and not p.name.startswith(".")
            and p.name.lower().endswith((".nii.gz", ".nii"))
            and not p.name.lower().endswith(SKIP_SUFFIXES)
        ]
        if not scans:
            raise FileNotFoundError(f"No NIfTI scans (.nii or .nii.gz) found in directory: {in_path}")
        out_path.mkdir(parents=True, exist_ok=True)
        return [(p, out_path / f"{strip_nifti_ext(p.name)}_seg.nii.gz") for p in scans], True

    if not in_path.is_file():
        raise FileNotFoundError(f"Input CT scan not found: {in_path}")
    if out_path.exists() and out_path.is_dir():
        return [(in_path, out_path / f"{strip_nifti_ext(in_path.name)}_seg.nii.gz")], False
    return [(in_path, out_path)], False


def resolve_body_mask_path(body_arg, in_file, out_file, is_batch):
    """Resolve optional `--body-mask` flag or path for single-file and batch modes."""
    if body_arg is None:
        return None
    stem = strip_nifti_ext(in_file.name)
    if body_arg is True:
        return out_file.parent / f"{stem}_body_mask.nii.gz"
    target = Path(body_arg)
    if is_batch or (target.exists() and target.is_dir()) or not target.name.lower().endswith((".nii.gz", ".nii")):
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{stem}_body_mask.nii.gz"
    return target


def write_csv_rows(csv_path, rows, include_scan=False):
    """Write structure volume and mean HU rows to a CSV file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (["scan"] if include_scan else []) + ["id", "structure", "voxels", "volume_mm3", "volume_cm3", "mean_hu"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {
                "id": row["id"],
                "structure": row["structure"],
                "voxels": row["voxels"],
                "volume_mm3": f"{row['volume_mm3']:.4f}",
                "volume_cm3": f"{row['volume_cm3']:.6f}",
                "mean_hu": "" if row["mean_hu"] is None else f"{row['mean_hu']:.2f}",
            }
            if include_scan:
                out_row["scan"] = row["scan"]
            writer.writerow(out_row)


def build_parser():
    parser = argparse.ArgumentParser(
        description="MouseSeg: Pure-CPU 3D mouse CT organ and subcutaneous tumor segmentation."
    )
    parser.add_argument("-i", "--input", required=True, help="Input CT NIfTI file (.nii/.nii.gz) or directory of scans")
    parser.add_argument("-o", "--output", required=True, help="Output mask NIfTI path (single scan) or output directory (batch)")
    parser.add_argument(
        "--mode",
        choices=["unified", "tumor_only"],
        default="unified",
        help="Segmentation mode: 'unified' (10 organs + tumor) or 'tumor_only' (default: unified)",
    )
    parser.add_argument("-m", "--model", default=None, help="Optional path to a custom or fine-tuned .onnx model (overrides --mode)")
    parser.add_argument(
        "--body-mask",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help="Also save binary (0/1) cradle-stripped mouse body mask (optional file/folder path, or auto <stem>_body_mask.nii.gz)",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help="Optional .csv path (or auto <stem>_volumes.csv) for structure volumes and mean HU",
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for ONNX Runtime (default: 4)")
    parser.add_argument(
        "--orientation-mode",
        choices=["auto", "header"],
        default="auto",
        help="Orientation handling: 'auto' detects mouse pose from CT anatomy, 'header' trusts NIfTI affine (default: auto)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console summary output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_path = Path(args.model) if args.model else DEFAULT_MODELS[args.mode]
    jobs, is_batch = collect_scan_jobs(args.input, args.output)
    session, config = load_onnx_model(model_path, num_threads=args.threads)

    batch_csv_rows = []
    for in_file, out_file in jobs:
        body_out = resolve_body_mask_path(args.body_mask, in_file, out_file, is_batch)
        res = segment_ct_volume(
            input_path=in_file,
            session=session,
            config=config,
            output_path=out_file,
            body_mask_path=body_out,
            orientation_mode=args.orientation_mode,
        )
        if not args.quiet:
            extra = f" + {body_out.name}" if body_out is not None else ""
            print(f"[MouseSeg] {in_file.name} -> {out_file}{extra} ({res['total_seconds']:.2f} s)")

        if args.csv is not None:
            stem = strip_nifti_ext(in_file.name)
            if args.csv is True and not is_batch:
                write_csv_rows(out_file.parent / f"{stem}_volumes.csv", res["stats"])
            elif isinstance(args.csv, str) and not is_batch:
                write_csv_rows(Path(args.csv), res["stats"])
            else:
                for row in res["stats"]:
                    batch_csv_rows.append({"scan": in_file.name, **row})

    if args.csv is not None and is_batch:
        csv_out = (Path(args.output) / "volumes.csv") if args.csv is True else Path(args.csv)
        if csv_out.exists() and csv_out.is_dir():
            csv_out = csv_out / "volumes.csv"
        write_csv_rows(csv_out, batch_csv_rows, include_scan=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
