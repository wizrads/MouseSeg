"""Whole-mouse binary Body mask generation (stripping scanner beds and cradles) and NIfTI saving."""

import gzip
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def lateral_border_values(volume_3d):
    """Return voxels on the four lateral (X, Y) faces of a 3D volume (excluding Z end caps)."""
    return np.concatenate([
        volume_3d[0, :, :].ravel(),
        volume_3d[-1, :, :].ravel(),
        volume_3d[:, 0, :].ravel(),
        volume_3d[:, -1, :].ravel(),
    ])


def is_preclipped_air_hu(hu_volume, border_p10_hu=None, wide_padded=False):
    """Return True when exterior air was pre-clipped around -400 HU instead of -1000 HU."""
    if hu_volume.size == 0:
        return False
    min_val = float(np.min(hu_volume))
    unpadded_clip = (-405.0 <= min_val <= -398.5) and (
        float(np.mean((hu_volume >= -402.5) & (hu_volume <= -398.5))) >= 0.005
    )
    spike_low = -405.0 if wide_padded else -401.0
    padded_spike = (
        float(np.mean((hu_volume >= spike_low) & (hu_volume <= -398.5))) >= 0.010
        and float(np.mean((hu_volume > -980.0) & (hu_volume < -405.0))) < 0.003
    )
    if unpadded_clip or padded_spike:
        return True
    if border_p10_hu is not None and (-415.0 <= float(border_p10_hu) <= -350.0):
        if float(np.percentile(hu_volume, 1.0)) >= -420.0:
            return True
    return False


def fill_2d_slice_holes(mask_3d):
    """Fill enclosed 2D holes inside each axial (X, Y) slice of a 3D boolean mask."""
    inverted = ~mask_3d
    planar_struct = np.zeros((3, 3, 3), dtype=bool)
    planar_struct[1, 1, 1] = True
    planar_struct[0, 1, 1] = planar_struct[2, 1, 1] = True
    planar_struct[1, 0, 1] = planar_struct[1, 2, 1] = True
    labeled, num_features = ndimage.label(inverted, structure=planar_struct)
    if num_features == 0:
        return mask_3d.copy()
    exterior_ids = np.unique(lateral_border_values(labeled))
    is_exterior = np.isin(labeled, exterior_ids)
    return ~(is_exterior & inverted)


def fill_3d_holes(mask_3d):
    """Fill enclosed 3D interior cavities (lungs, trachea, GI gas) inside a 3D boolean mask."""
    inverted = ~mask_3d
    labeled, num_features = ndimage.label(inverted, structure=ndimage.generate_binary_structure(3, 1))
    if num_features == 0:
        return mask_3d.copy()
    border_ids = np.unique(np.concatenate([
        labeled[0, :, :].ravel(),
        labeled[-1, :, :].ravel(),
        labeled[:, 0, :].ravel(),
        labeled[:, -1, :].ravel(),
        labeled[:, :, 0].ravel(),
        labeled[:, :, -1].ravel(),
    ]))
    is_exterior = np.isin(labeled, border_ids)
    return ~(is_exterior & inverted)


def compute_body_mask(
    ct_hu_ras,
    seg_mask_ras,
    voxel_spacing_mm,
    is_calibrated_hu=True,
    open_radius_mm=1.15,
    pre_harmonized_air=False,
):
    """Build a cradle-stripped binary mouse Body mask on the resampled RAS grid."""
    sx = 2 if (voxel_spacing_mm[0] <= 0.25 and ct_hu_ras.shape[0] >= 48) else 1
    sy = 2 if (voxel_spacing_mm[1] <= 0.25 and ct_hu_ras.shape[1] >= 48) else 1
    sz = 2 if (voxel_spacing_mm[2] <= 0.25 and ct_hu_ras.shape[2] >= 48) else 1
    coarse_spacing = (voxel_spacing_mm[0] * sx, voxel_spacing_mm[1] * sy, voxel_spacing_mm[2] * sz)

    work_hu = ct_hu_ras
    if is_calibrated_hu and not pre_harmonized_air:
        hu_probe = ct_hu_ras[::sx, ::sy, ::sz]
        border_vals = lateral_border_values(hu_probe)
        border_p10 = float(np.percentile(border_vals, 10.0)) if border_vals.size > 0 else -1000.0
        if is_preclipped_air_hu(hu_probe, border_p10_hu=border_p10, wide_padded=True):
            work_hu = np.asarray(ct_hu_ras, dtype=np.float32).copy()
            work_hu[(work_hu >= -415.0) & (work_hu <= -395.0)] = -650.0

    if is_calibrated_hu:
        body_thresh = -450.0
        skin_thresh = -500.0
    else:
        p05, p98 = np.percentile(work_hu, [5.0, 98.0])
        span = max(float(p98 - p05), 1.0)
        body_thresh = float(p05) + 0.22 * span
        skin_thresh = float(p05) + 0.18 * span

    anatomy_mask = seg_mask_ras > 0
    internal_organs = (seg_mask_ras >= 1) & (seg_mask_ras <= 10)

    hu_coarse = work_hu[::sx, ::sy, ::sz]
    fg_coarse = (work_hu > body_thresh)[::sx, ::sy, ::sz].copy()
    anat_coarse = anatomy_mask[::sx, ::sy, ::sz]
    internal_coarse = internal_organs[::sx, ::sy, ::sz]

    # 1. Strip stationary plastic scanner beds and side rails extruded along Z
    active_z = np.flatnonzero(fg_coarse.any(axis=(0, 1)))
    cradle_strip = None
    if is_calibrated_hu and active_z.size >= 6:
        z0, z1 = int(active_z[0]), int(active_z[-1]) + 1
        hu_sub = hu_coarse[:, :, z0:z1]
        fg_sub = fg_coarse[:, :, z0:z1]
        solid_frac = np.mean(fg_sub, axis=2)
        anat_any = np.any(anat_coarse[:, :, z0:z1], axis=2)
        bone_any = np.any(hu_sub > 300.0, axis=2)
        cnt_z = np.maximum(1, np.sum(fg_sub, axis=2))
        mean_z = np.sum(np.where(fg_sub, hu_sub, 0.0), axis=2) / cnt_z
        var_z = np.sum(np.where(fg_sub, (hu_sub - mean_z[:, :, None]) ** 2, 0.0), axis=2) / cnt_z
        std_z = np.sqrt(np.maximum(0.0, var_z))

        plastic_xy = (solid_frac >= 0.45) & (~anat_any) & (~bone_any) & (mean_z >= -350.0) & (mean_z <= 260.0) & (std_z <= 85.0)
        rail_xy = (solid_frac >= 0.85) & (~anat_any) & (std_z <= 140.0)
        if np.any(plastic_xy) or np.any(rail_xy):
            seed_xy = plastic_xy | rail_xy
            bed_xy = seed_xy.copy()
            uniform_nonbone_xy = (solid_frac >= 0.45) & (~bone_any) & (mean_z >= -350.0) & (mean_z <= 260.0) & (std_z <= 85.0)
            row_frac = np.mean(seed_xy, axis=0)
            col_frac = np.mean(seed_xy, axis=1)
            if np.any(row_frac >= 0.18):
                bed_xy |= (row_frac >= 0.18)[None, :] & uniform_nonbone_xy
            if np.any(col_frac >= 0.18):
                bed_xy |= (col_frac >= 0.18)[:, None] & uniform_nonbone_xy

            protected = (
                ndimage.distance_transform_edt(~anat_coarse, sampling=coarse_spacing) <= 0.45
                if np.any(anat_coarse) else np.zeros_like(fg_coarse)
            )
            cradle_strip = ((bed_xy[:, :, None] & (hu_coarse <= 280.0)) | rail_xy[:, :, None]) & (~protected)
            fg_coarse &= ~cradle_strip

    # 2. Morphological opening on coarse grid to detach remaining bed contact points
    eroded = ndimage.distance_transform_edt(fg_coarse, sampling=coarse_spacing) >= open_radius_mm
    labeled, num_cc = ndimage.label(eroded, structure=ndimage.generate_binary_structure(3, 2))
    keep_ids = []
    if num_cc > 0:
        ref_mask = internal_coarse if int(internal_coarse.sum()) >= 12 else anat_coarse
        skeleton_mask = hu_coarse > 350.0
        if np.any(ref_mask):
            organ_overlap = np.bincount(labeled[ref_mask], minlength=num_cc + 1)[1:]
            bone_overlap = np.bincount(labeled[skeleton_mask], minlength=num_cc + 1)[1:] if np.any(skeleton_mask) else np.zeros_like(organ_overlap)
            scores = organ_overlap + 2 * bone_overlap
            max_score = int(scores.max()) if scores.size > 0 else 0
            if max_score > 0:
                keep_ids = (np.flatnonzero((organ_overlap >= max(4, int(0.002 * ref_mask.sum()))) & (scores >= int(0.01 * max_score))) + 1).tolist()
                if not keep_ids:
                    keep_ids = [int(np.argmax(scores)) + 1]
        if not keep_ids:
            counts = np.bincount(labeled.ravel())[1:]
            keep_ids = [int(np.argmax(counts)) + 1]
        core_coarse = np.isin(labeled, keep_ids)
    else:
        core_coarse = anat_coarse.copy() if np.any(anat_coarse) else fg_coarse.copy()

    dilated_coarse = ndimage.distance_transform_edt(~core_coarse, sampling=coarse_spacing) <= (open_radius_mm + 0.25)
    if cradle_strip is not None:
        dilated_coarse &= ~cradle_strip

    # 3. Upsample back to resampled grid, gate by skin HU, and fill internal air cavities
    ux = np.minimum(np.arange(work_hu.shape[0]) // sx, dilated_coarse.shape[0] - 1)
    uy = np.minimum(np.arange(work_hu.shape[1]) // sy, dilated_coarse.shape[1] - 1)
    uz = np.minimum(np.arange(work_hu.shape[2]) // sz, dilated_coarse.shape[2] - 1)
    body_mask = (dilated_coarse[np.ix_(ux, uy, uz)] & (work_hu > skin_thresh)) | anatomy_mask
    body_mask = fill_2d_slice_holes(body_mask)
    body_mask = fill_3d_holes(body_mask)
    return body_mask | anatomy_mask


def save_nifti_mask(mask_3d, reference_img, output_path):
    """Save a 3D uint8 mask NIfTI preserving the input image's shape, affine, and header."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = np.asarray(mask_3d, dtype=np.uint8)
    if len(reference_img.shape) > 3:
        out_data = out_data[(Ellipsis,) + (None,) * (len(reference_img.shape) - 3)]

    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    out_img = nib.Nifti1Image(out_data, reference_img.affine, header)

    if output_path.name.lower().endswith(".nii.gz"):
        with gzip.open(output_path, "wb", compresslevel=1) as gz_file:
            out_img.to_file_map({"image": nib.FileHolder(fileobj=gz_file)})
    else:
        nib.save(out_img, str(output_path))
    return output_path
