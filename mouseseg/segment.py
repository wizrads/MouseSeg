"""Pure-CPU 3D mouse CT segmentation pipeline using ONNX Runtime."""

import gzip
import io
import json
import math
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import onnxruntime as ort
from scipy import ndimage

from .body import compute_body_mask, fill_2d_slice_holes, is_preclipped_air_hu, lateral_border_values, save_nifti_mask
from .postprocess import (
    STRUCTURE_NAMES,
    clean_organ_masks,
    clean_tumor_mask,
    finalize_organ_predictions,
    keep_largest_components,
    resolve_organ_tumor_overlap,
)


def load_onnx_model(onnx_path, num_threads=4):
    """Load an ONNX model on CPU and return `(session, config_dict)`."""
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(num_threads)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    custom_map = session.get_modelmeta().custom_metadata_map or {}
    config = {}
    for key in ("mouseseg_config", "mouseseg_config_json"):
        if key in custom_map:
            config = json.loads(custom_map[key])
            break

    in_shape = session.get_inputs()[0].shape
    out_shapes = [o.shape for o in session.get_outputs()]
    in_channels = int(config.get("in_channels", in_shape[1] if isinstance(in_shape[1], int) else 2))
    default_num_oar = 1 if len(out_shapes) == 1 else (int(out_shapes[0][1]) if isinstance(out_shapes[0][1], int) else 12)
    num_oar_classes = int(config.get("num_oar_classes", default_num_oar))
    tumor_only = bool(config.get("tumor_only", num_oar_classes <= 1))

    config.setdefault("in_channels", in_channels)
    config.setdefault("num_oar_classes", num_oar_classes)
    config.setdefault("tumor_only", tumor_only)
    config.setdefault("tumor_label_id", 12)
    config.setdefault("spatial_divisor", 16)
    config.setdefault("target_spacing_mm", [0.2, 0.2, 0.2])
    config.setdefault("hu_windows", [[-400.0, 1000.0]] if tumor_only else [[-1000.0, 3000.0], [-400.0, 1000.0]])
    config.setdefault("harmonize_air_floor", True)
    config.setdefault("soft_tissue_calib_target_hu", 45.0)
    config.setdefault("soft_tissue_max_shift_hu", 25.0)
    config.setdefault("window_slices", int(config.get("recommended_max_3d_depth_voxels", 128)))
    config.setdefault("window_overlap", float(config.get("recommended_window_overlap", 0.25)))
    return session, config


def load_ct_nifti(path):
    """Load a 3D NIfTI CT image (including singleton trailing dims), repair unit sform if needed, and return HU scaling."""
    path = Path(path)
    try:
        img = nib.load(str(path))
    except Exception:
        with gzip.open(path, "rb") as gz_file:
            buf = io.BytesIO(gz_file.read())
        img = nib.Nifti1Image.from_file_map({"image": nib.FileHolder(filename=str(path), fileobj=buf)})

    if len(img.shape) < 3 or any(s != 1 for s in img.shape[3:]):
        raise ValueError(f"Expected a 3D CT scan, got shape {img.shape}")

    geom_affine = np.array(img.affine, dtype=np.float64)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if hasattr(img, "get_sform") and hasattr(img, "get_qform"):
        sform, scode = img.get_sform(coded=True)
        qform, qcode = img.get_qform(coded=True)
        if scode > 0 and np.allclose(np.abs(np.diag(sform)[:3]), 1.0, atol=1e-3) and not np.allclose(zooms, 1.0, atol=1e-3):
            if qcode > 0 and not np.allclose(np.abs(np.diag(qform)[:3]), 1.0, atol=1e-3):
                geom_affine = np.array(qform, dtype=np.float64)
            else:
                geom_affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
                geom_affine[:3, 3] = sform[:3, 3]

    if nib.is_proxy(img.dataobj) and hasattr(img.dataobj, "get_unscaled"):
        raw_3d = np.asanyarray(img.dataobj.get_unscaled()).reshape(img.shape[:3])
        raw_slope = getattr(img.dataobj, "slope", 1.0)
        raw_inter = getattr(img.dataobj, "inter", 0.0)
        slope = float(raw_slope) if raw_slope is not None and np.isfinite(raw_slope) and float(raw_slope) != 0.0 else 1.0
        inter = float(raw_inter) if raw_inter is not None and np.isfinite(raw_inter) else 0.0
    else:
        raw_3d = np.asanyarray(img.dataobj).reshape(img.shape[:3])
        slope, inter = 1.0, 0.0

    voxel_spacing_mm = tuple(float(v) for v in nib.affines.voxel_sizes(geom_affine)[:3])
    return img, raw_3d, slope, inter, geom_affine, voxel_spacing_mm


def reorient_to_ras(volume_3d, affine):
    """Reorient a 3D volume to canonical RAS (Right, Anterior, Superior) orientation."""
    in_ornt = nib.orientations.io_orientation(affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    to_ras = nib.orientations.ornt_transform(in_ornt, ras_ornt)
    from_ras = nib.orientations.ornt_transform(ras_ornt, in_ornt)
    ras_affine = affine @ nib.orientations.inv_ornt_aff(to_ras, volume_3d.shape)
    ras_spacing_mm = tuple(float(v) for v in nib.affines.voxel_sizes(ras_affine)[:3])
    return nib.orientations.apply_orientation(volume_3d, to_ras), ras_spacing_mm, from_ras


def _soft_enclosed_mouse_core(sub_hu, sp):
    """Build soft-tissue-enclosed mouse mask and 1.8 mm eroded interior core on a coarse grid."""
    soft_ring = (sub_hu > -280.0) & (sub_hu < 320.0)
    soft_enc = fill_2d_slice_holes(soft_ring)
    padded = np.pad(soft_enc, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=False)
    core = ndimage.distance_transform_edt(padded, sampling=sp)[1:-1, 1:-1, :] >= 1.8
    if not np.any(core):
        core = soft_enc
    lbl, num_cc = ndimage.label(core, structure=ndimage.generate_binary_structure(3, 2))
    if num_cc > 1:
        core = lbl == (int(np.argmax(np.bincount(lbl.ravel())[1:])) + 1)
    return soft_enc, core


def estimate_mouse_pose_and_mask_cradle(raw_ras, spacing_mm, slope=1.0, inter=0.0, auto_pose=True):
    """Strip external high-HU Z-persistent scanner cradles and optionally estimate canonical prone RAS pose."""
    strides = [max(1, raw_ras.shape[i] // 160, int(round(0.45 / max(spacing_mm[i], 1e-4)))) for i in range(3)]
    eff_slope = slope if slope != 0.0 else 1.0
    sub = np.asarray(raw_ras[::strides[0], ::strides[1], ::strides[2]], dtype=np.float32) * np.float32(eff_slope) + np.float32(inter)
    sp = tuple(spacing_mm[i] * strides[i] for i in range(3))

    border_vals = lateral_border_values(sub)
    border_p10 = float(np.percentile(border_vals, 10.0)) if border_vals.size > 0 else -1000.0
    is_calibrated_hu = border_p10 < -350.0
    if not is_calibrated_hu:
        if not auto_pose:
            return raw_ras, spacing_mm, slope, inter, (0, 1, 2), 0, False
        p05, p995 = np.percentile(sub, [5.0, 99.5])
        if p995 - p05 > 1e-3:
            sub = -1000.0 + (sub - float(p05)) * (2800.0 / float(p995 - p05))

    soft_enc, core = _soft_enclosed_mouse_core(sub, sp)

    # 1. Detect if the mouse long axis lies along X or Y instead of Z
    perm = (0, 1, 2)
    if auto_pose and sub.shape[2] * sp[2] >= 18.0:
        spans = [float(sub.shape[i] * sp[i]) for i in range(3)]
        for ax in range(3):
            idx = np.flatnonzero(core.any(axis=tuple(j for j in range(3) if j != ax)))
            if idx.size > 1:
                spans[ax] = float((idx[-1] - idx[0] + 1) * sp[ax])
        max_ax = int(np.argmax(spans))
        if max_ax != 2 and spans[max_ax] > max(20.0, 1.25 * spans[2]):
            perm = (1, 2, 0) if max_ax == 0 else (0, 2, 1)
            raw_ras = np.transpose(raw_ras, perm)
            spacing_mm = tuple(spacing_mm[p] for p in perm)
            strides = [strides[p] for p in perm]
            sub = np.transpose(sub, perm).copy()
            sp = tuple(sp[p] for p in perm)
            soft_enc, core = _soft_enclosed_mouse_core(sub, sp)

    # 2. Mask external high-HU (> 300 HU) Z-persistent (>= 60% Z span) composite cradles to -1000 HU
    if is_calibrated_hu and sub.shape[2] >= 24:
        hi_z_frac = np.mean(sub > 300.0, axis=2)
        dense_z_frac = np.mean(sub > 450.0, axis=2)
        enc_xy_frac = np.mean(soft_enc, axis=2)
        core_xy_frac = np.mean(core, axis=2)
        seed_cradle_xy = (hi_z_frac >= 0.60) & (dense_z_frac >= 0.25) & (enc_xy_frac < 0.20)
        if np.any(seed_cradle_xy):
            grow_cradle_xy = (hi_z_frac >= 0.50) & (core_xy_frac < 0.20)
            lbl_xy, _ = ndimage.label(seed_cradle_xy | grow_cradle_xy, structure=ndimage.generate_binary_structure(2, 2))
            cradle_xy = np.isin(lbl_xy, np.unique(lbl_xy[seed_cradle_xy])) & (lbl_xy > 0)
            cradle_xy_pad = ndimage.binary_dilation(cradle_xy, iterations=2) & (core_xy_frac < 0.15)
            mouse_env = ndimage.distance_transform_edt(~core, sampling=sp) <= 2.8

            sub_strip = (
                (cradle_xy_pad[:, :, None] & (~core) & (sub > 200.0))
                | (cradle_xy[:, :, None] & (~soft_enc) & (sub > -350.0))
                | ((~mouse_env) & (sub > -350.0))
            )
            sub = np.where(sub_strip, np.float32(-1000.0), sub)

            ux = np.minimum(np.arange(raw_ras.shape[0]) // strides[0], sub.shape[0] - 1)
            uy = np.minimum(np.arange(raw_ras.shape[1]) // strides[1], sub.shape[1] - 1)
            uz = np.minimum(np.arange(raw_ras.shape[2]) // strides[2], sub.shape[2] - 1)
            if eff_slope != 1.0 or inter != 0.0:
                raw_ras = raw_ras.astype(np.float32, copy=True) * np.float32(eff_slope) + np.float32(inter)
                eff_slope = slope = 1.0
                inter = 0.0
            else:
                raw_ras = raw_ras.copy()
            fill_raw = raw_ras.dtype.type(-1000) if np.issubdtype(raw_ras.dtype, np.integer) else np.float32(-1000.0)
            cx_full = cradle_xy_pad[np.ix_(ux, uy)]
            cseed_full = cradle_xy[np.ix_(ux, uy)]
            for z0 in range(0, raw_ras.shape[2], 64):
                z1 = min(raw_ras.shape[2], z0 + 64)
                uz_sl = uz[z0:z1]
                sl = raw_ras[:, :, z0:z1]
                core_sl = core[np.ix_(ux, uy, uz_sl)]
                enc_sl = soft_enc[np.ix_(ux, uy, uz_sl)]
                env_sl = mouse_env[np.ix_(ux, uy, uz_sl)]
                drop = (
                    (cx_full[:, :, None] & (~core_sl) & (sl > 200))
                    | (cseed_full[:, :, None] & (~enc_sl) & (sl > -350))
                    | ((~env_sl) & (sl > -350))
                )
                sl[drop] = fill_raw

    if not auto_pose or sub.shape[2] * sp[2] < 18.0:
        return raw_ras, spacing_mm, slope, inter, perm, 0, False

    # 3. Cranial-caudal polarity (z_flip): aerated thoracic lungs (+Z) vs solid abdomen (-Z)
    lungs = (soft_enc & (~(sub > -350.0))) & (sub >= -880.0) & (sub <= -220.0)
    abdomen = soft_enc & (sub > -150.0) & (sub < 700.0)
    lung_z = lungs.sum(axis=(0, 1)).astype(np.float64)
    abd_z = abdomen.sum(axis=(0, 1)).astype(np.float64)
    z_coords = np.arange(sub.shape[2], dtype=np.float64) * sp[2]
    z_flip = bool(
        lung_z.sum() >= 4
        and abd_z.sum() > 0
        and ((lung_z * z_coords).sum() / lung_z.sum() - (abd_z * z_coords).sum() / abd_z.sum()) < -1.5
    )

    # 4. Axial roll k_rot90 in {0, 1, 2, 3}: per-slice soft-tissue minus largest-2D-bone centroid over trunk
    sub_z = np.flip(sub, axis=2) if z_flip else sub
    enc_z = np.flip(soft_enc, axis=2) if z_flip else soft_enc
    act_z = np.flatnonzero(enc_z.any(axis=(0, 1)))
    za0, za1 = (int(act_z[0]), int(act_z[-1]) + 1) if act_z.size > 0 else (0, sub_z.shape[2])
    tz0 = za0 + int(0.25 * (za1 - za0))
    tz1 = za0 + max(tz0 - za0 + 1, int(0.70 * (za1 - za0)))

    dxs, dys = [], []
    conn_2d = ndimage.generate_binary_structure(2, 2)
    for z in range(tz0, tz1):
        sl_hu = sub_z[:, :, z]
        sl_enc = enc_z[:, :, z]
        soft_m = sl_enc & (sl_hu >= -200.0) & (sl_hu < 350.0)
        bone_m = sl_enc & (sl_hu >= 450.0)
        if np.count_nonzero(bone_m) < 3:
            bone_m = sl_enc & (sl_hu >= 320.0)
        if np.count_nonzero(soft_m) < 20 or np.count_nonzero(bone_m) < 3:
            continue
        lbl_b, num_b = ndimage.label(bone_m, structure=conn_2d)
        big_bone = lbl_b == (int(np.argmax(np.bincount(lbl_b.ravel())[1:])) + 1)
        sx_i, sy_i = np.nonzero(soft_m)
        bx_i, by_i = np.nonzero(big_bone)
        dxs.append((float(sx_i.mean()) - float(bx_i.mean())) * sp[0])
        dys.append((float(sy_i.mean()) - float(by_i.mean())) * sp[1])

    med_dx = float(np.median(dxs)) if dxs else 0.0
    med_dy = float(np.median(dys)) if dys else 0.0
    scores = {0: med_dy, 1: med_dx, 2: -med_dy, 3: -med_dx}
    best_k = 0 if scores[0] >= 1.8 else max(scores, key=scores.get)
    if scores[best_k] < 1.0:
        best_k = 0

    if best_k:
        raw_ras = np.rot90(raw_ras, k=best_k, axes=(0, 1))
        if best_k % 2 == 1:
            spacing_mm = (spacing_mm[1], spacing_mm[0], spacing_mm[2])
    if z_flip:
        raw_ras = np.flip(raw_ras, axis=2)

    return raw_ras, spacing_mm, slope, inter, perm, best_k, z_flip


def estimate_mouse_pose(raw_ras, spacing_mm, slope=1.0, inter=0.0):
    """Estimate canonical prone `(perm, k_rot90, z_flip)` pose from CT anatomy."""
    _, _, _, _, perm, k_rot90, z_flip = estimate_mouse_pose_and_mask_cradle(
        raw_ras, spacing_mm, slope=slope, inter=inter, auto_pose=True
    )
    return perm, k_rot90, z_flip


def find_mouse_bounding_box(raw_ras, spacing_mm, slope=1.0, inter=0.0, margin_mm=3.5):
    """Find a padded 3D bounding box `(x0, x1, y0, y1, z0, z1)` around the mouse body."""
    nx, ny, nz = raw_ras.shape
    sx, sy, sz = [max(1, raw_ras.shape[i] // 160, int(round(0.40 / max(spacing_mm[i], 1e-4)))) for i in range(3)]
    sub = raw_ras[::sx, ::sy, ::sz]
    eff_slope = slope if slope != 0.0 else 1.0
    sub_hu = np.asarray(sub, dtype=np.float32) * eff_slope + inter
    border_p10 = float(np.percentile(lateral_border_values(sub_hu), 10.0))

    if is_preclipped_air_hu(sub_hu, border_p10_hu=border_p10, wide_padded=True):
        sub_hu = sub_hu.copy()
        sub_hu[(sub_hu >= -415.0) & (sub_hu <= -395.0)] = -1000.0
        border_p10 = float(np.percentile(lateral_border_values(sub_hu), 10.0))

    if border_p10 < -350.0:
        fg_coarse = sub_hu > -500.0
    else:
        p05, p98 = np.percentile(sub_hu, [5.0, 98.0])
        if p98 - p05 <= 1e-3:
            return (0, nx, 0, ny, 0, nz)
        fg_coarse = sub_hu > (float(p05) + 0.20 * float(p98 - p05))

    if not np.any(fg_coarse):
        return (0, nx, 0, ny, 0, nz)

    # Suppress thin scanner cradle rails extruded along Z
    if border_p10 < -350.0 and fg_coarse.shape[2] >= 40:
        nz_c = fg_coarse.shape[2]
        z_count = fg_coarse.sum(axis=2)
        persistent = z_count >= int(math.ceil(0.80 * nz_c))
        if persistent.any() and not persistent.all():
            sp_xy = max(min(spacing_mm[0] * sx, spacing_mm[1] * sy), 1e-4)
            erode_iters = max(1, int(round(2.6 / sp_xy)))
            cross_2d = ndimage.generate_binary_structure(2, 1)
            torso = ndimage.binary_erosion(z_count >= int(math.ceil(0.50 * nz_c)), structure=cross_2d, iterations=erode_iters)
            torso = ndimage.binary_dilation(torso, structure=cross_2d, iterations=erode_iters + 2)
            rail = persistent & (~torso)
            if rail.any():
                rail = ndimage.binary_dilation(rail, structure=cross_2d, iterations=2) & (~torso)
                if np.any(fg_coarse & (~rail[:, :, None])):
                    fg_coarse = fg_coarse & (~rail[:, :, None])

    labeled, num_cc = ndimage.label(fg_coarse, structure=ndimage.generate_binary_structure(3, 2))
    if num_cc > 1:
        counts = np.bincount(labeled.ravel())[1:]
        keep_ids = np.flatnonzero(counts >= max(8, int(0.04 * int(counts.max())))) + 1
        fg_coarse = np.isin(labeled, keep_ids)

    xs = np.flatnonzero(fg_coarse.any(axis=(1, 2)))
    ys = np.flatnonzero(fg_coarse.any(axis=(0, 2)))
    zs = np.flatnonzero(fg_coarse.any(axis=(0, 1)))
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        return (0, nx, 0, ny, 0, nz)

    pad = [int(math.ceil(margin_mm / max(s, 1e-4))) for s in spacing_mm]
    return (
        max(0, int(xs[0] * sx) - pad[0]), min(nx, int((xs[-1] + 1) * sx) + pad[0]),
        max(0, int(ys[0] * sy) - pad[1]), min(ny, int((ys[-1] + 1) * sy) + pad[1]),
        max(0, int(zs[0] * sz) - pad[2]), min(nz, int((zs[-1] + 1) * sz) + pad[2]),
    )


def _axis_resample_weights(in_len, out_len):
    """Build 1D resampling matrix using anti-aliased box averaging when downsampling >1.25x."""
    weights = np.zeros((out_len, in_len), dtype=np.float32)
    scale = in_len / float(out_len)
    if scale > 1.25:
        lo = np.arange(out_len, dtype=np.float64)[:, None] * scale
        hi = np.minimum(float(in_len), lo + scale)
        idx = np.arange(in_len, dtype=np.float64)[None, :]
        weights[:] = (np.maximum(0.0, np.minimum(hi, idx + 1.0) - np.maximum(lo, idx)) / scale).astype(np.float32)
    else:
        pos = np.clip((np.arange(out_len, dtype=np.float64) + 0.5) * scale - 0.5, 0.0, max(0.0, in_len - 1.0))
        i0 = np.minimum(np.floor(pos).astype(np.int32), max(0, in_len - 1))
        i1 = np.minimum(i0 + 1, max(0, in_len - 1))
        frac = (pos - i0).astype(np.float32)
        rows = np.arange(out_len, dtype=np.int32)
        weights[rows, i0] += 1.0 - frac
        weights[rows, i1] += frac
    return weights


def resample_volume(volume_3d, out_shape, slope=1.0, inter=0.0):
    """Resample a 3D CT array to `out_shape` in float32 physical HU."""
    if volume_3d.shape == out_shape:
        return volume_3d.astype(np.float32, copy=True) * np.float32(slope) + np.float32(inter)

    cur = np.ascontiguousarray(volume_3d, dtype=np.float32)
    for axis in sorted(range(3), key=lambda a: out_shape[a] / max(cur.shape[a], 1)):
        if cur.shape[axis] == out_shape[axis]:
            continue
        w = _axis_resample_weights(cur.shape[axis], out_shape[axis])
        if axis == 0:
            cur = (w @ cur.reshape(cur.shape[0], -1)).reshape(out_shape[0], cur.shape[1], cur.shape[2])
        elif axis == 1:
            cur = np.matmul(w, cur)
        else:
            cur = (cur.reshape(-1, cur.shape[2]) @ np.ascontiguousarray(w.T)).reshape(cur.shape[0], cur.shape[1], out_shape[2])
    return cur * np.float32(slope) + np.float32(inter)


def resize_nearest_3d(mask_3d, target_shape):
    """Nearest-neighbor 3D resize for sharp thin structures (Bone and Trachea)."""
    if mask_3d.shape == target_shape:
        return mask_3d
    indices = [
        np.clip(np.floor((np.arange(target_shape[i], dtype=np.float32) + 0.5) * np.float32(mask_3d.shape[i] / max(target_shape[i], 1))).astype(np.int32), 0, mask_3d.shape[i] - 1)
        for i in range(3)
    ]
    return mask_3d[np.ix_(indices[0], indices[1], indices[2])]


def _smooth_binomial_3d(field_3d):
    """Apply 1 pass of separable 3D [0.25, 0.5, 0.25] binomial smoothing with clamped borders."""
    px = np.pad(field_3d, ((1, 1), (0, 0), (0, 0)), mode="edge")
    fx = np.float32(0.25) * (px[:-2] + px[2:]) + np.float32(0.5) * px[1:-1]
    py = np.pad(fx, ((0, 0), (1, 1), (0, 0)), mode="edge")
    fy = np.float32(0.25) * (py[:, :-2] + py[:, 2:]) + np.float32(0.5) * py[:, 1:-1]
    pz = np.pad(fy, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return np.float32(0.25) * (pz[:, :, :-2] + pz[:, :, 2:]) + np.float32(0.5) * pz[:, :, 1:-1]


def _trilinear_axis_map(fg_lo, fg_hi, res_len, target_len):
    """Map foreground bounds [fg_lo, fg_hi) onto tight target indices and linear interpolation weights."""
    r0, r1 = max(0, fg_lo - 2), min(res_len, fg_hi + 2)
    sub_len = r1 - r0
    pos_all = np.clip(
        (np.arange(target_len, dtype=np.float64) + 0.5) * (res_len / float(max(target_len, 1))) - 0.5,
        0.0,
        max(0.0, res_len - 1.0),
    )
    active = np.flatnonzero((pos_all >= max(0.0, fg_lo - 1.5)) & (pos_all <= min(res_len - 1.0, (fg_hi - 1) + 1.5)))
    if active.size == 0:
        return r0, r1, 0, 0, None
    t0, t1 = int(active[0]), int(active[-1]) + 1
    loc = np.clip(pos_all[t0:t1] - r0, 0.0, max(0.0, sub_len - 1.0))
    i0 = np.minimum(np.floor(loc).astype(np.int32), max(0, sub_len - 1))
    i1 = np.minimum(i0 + 1, max(0, sub_len - 1))
    frac = (loc - i0).astype(np.float32)
    w = np.zeros((t1 - t0, sub_len), dtype=np.float32)
    rows = np.arange(t1 - t0)
    w[rows, i0] += np.float32(1.0) - frac
    w[rows, i1] += frac
    return r0, r1, t0, t1, w


def _smooth_trilinear_blocks(bin_3d, target_shape):
    """Smooth a 3D boolean mask on its tight bbox and yield 64-slice trilinearly upsampled blocks."""
    xs = np.flatnonzero(bin_3d.any(axis=(1, 2)))
    if xs.size == 0:
        return
    fx0, fx1 = int(xs[0]), int(xs[-1]) + 1
    ys = np.flatnonzero(bin_3d[fx0:fx1].any(axis=(0, 2)))
    fy0, fy1 = int(ys[0]), int(ys[-1]) + 1
    zs = np.flatnonzero(bin_3d[fx0:fx1, fy0:fy1].any(axis=(0, 1)))
    fz0, fz1 = int(zs[0]), int(zs[-1]) + 1

    rx0, rx1, tx0, tx1, wx = _trilinear_axis_map(fx0, fx1, bin_3d.shape[0], target_shape[0])
    ry0, ry1, ty0, ty1, wy = _trilinear_axis_map(fy0, fy1, bin_3d.shape[1], target_shape[1])
    rz0, rz1, tz0, tz1, wz = _trilinear_axis_map(fz0, fz1, bin_3d.shape[2], target_shape[2])
    if wx is None or wy is None or wz is None:
        return

    sm = _smooth_binomial_3d(bin_3d[rx0:rx1, ry0:ry1, rz0:rz1].astype(np.float32))
    sx, sy, sz = sm.shape
    tx, ty, tz = tx1 - tx0, ty1 - ty0, tz1 - tz0
    sm_flat = sm.reshape(-1, sz)
    wz_t = np.ascontiguousarray(wz.T)
    for z_s in range(0, tz, 64):
        z_e = min(tz, z_s + 64)
        bz = z_e - z_s
        cz = (sm_flat @ wz_t[:, z_s:z_e]).reshape(sx, sy, bz)
        cy = np.matmul(wy, cz)
        val = (wx @ cy.reshape(sx, -1)).reshape(tx, ty, bz)
        yield tx0, tx1, ty0, ty1, tz0 + z_s, tz0 + z_e, val


def smooth_and_upsample_binary(mask_res_ras, target_shape):
    """Smooth a binary mask with 1-pass [0.25, 0.5, 0.25] 3D kernel and trilinearly upsample (>= 0.5)."""
    out_bool = np.zeros(target_shape, dtype=bool)
    for x0, x1, y0, y1, z0, z1, val in _smooth_trilinear_blocks(mask_res_ras, target_shape):
        out_bool[x0:x1, y0:y1, z0:z1] = val >= np.float32(0.5)
    return out_bool


def smooth_and_upsample_masks(seg_res_ras, target_shape, tumor_id=12):
    """Upsample label map keeping Trachea (9) & Bone (10) crisp and smoothing soft-tissue organs & tumor."""
    out_seg = np.zeros(target_shape, dtype=np.uint8)
    present = {int(c) for c in np.unique(seg_res_ras).tolist() if c > 0}
    if not present:
        return out_seg

    # Keep thin Trachea (9) and Bone (10) crisp via tight-bbox nearest-neighbor upsampling
    if 9 in present or 10 in present:
        sharp_res = (seg_res_ras == 9) | (seg_res_ras == 10)
        xs = np.flatnonzero(sharp_res.any(axis=(1, 2)))
        if xs.size > 0:
            ys = np.flatnonzero(sharp_res[xs[0] : xs[-1] + 1].any(axis=(0, 2)))
            zs = np.flatnonzero(sharp_res[xs[0] : xs[-1] + 1, ys[0] : ys[-1] + 1].any(axis=(0, 1)))
            idx = [
                np.clip(np.floor((np.arange(target_shape[i], dtype=np.float32) + 0.5) * np.float32(seg_res_ras.shape[i] / max(target_shape[i], 1))).astype(np.int32), 0, seg_res_ras.shape[i] - 1)
                for i in range(3)
            ]
            tx = np.flatnonzero((idx[0] >= xs[0]) & (idx[0] <= xs[-1]))
            ty = np.flatnonzero((idx[1] >= ys[0]) & (idx[1] <= ys[-1]))
            tz = np.flatnonzero((idx[2] >= zs[0]) & (idx[2] <= zs[-1]))
            if tx.size and ty.size and tz.size:
                x0, x1, y0, y1, z0, z1 = int(tx[0]), int(tx[-1]) + 1, int(ty[0]), int(ty[-1]) + 1, int(tz[0]), int(tz[-1]) + 1
                nn_sub = seg_res_ras[np.ix_(idx[0][x0:x1], idx[1][y0:y1], idx[2][z0:z1])]
                sh = (nn_sub == 9) | (nn_sub == 10)
                out_seg[x0:x1, y0:y1, z0:z1][sh] = nn_sub[sh]

    soft_oar_ids = [cid for cid in range(1, 9) if cid in present]
    if soft_oar_ids:
        best_score = np.zeros(target_shape, dtype=np.uint8)
        for cid in soft_oar_ids:
            for x0, x1, y0, y1, z0, z1, val in _smooth_trilinear_blocks(seg_res_ras == cid, target_shape):
                score_u8 = np.rint(np.clip(val, np.float32(0.0), np.float32(1.0)) * np.float32(255.0)).astype(np.uint8)
                out_sub = out_seg[x0:x1, y0:y1, z0:z1]
                best_sub = best_score[x0:x1, y0:y1, z0:z1]
                win = (score_u8 >= 128) & (score_u8 > best_sub) & (out_sub != 9) & (out_sub != 10)
                out_sub[win] = np.uint8(cid)
                best_sub[win] = score_u8[win]
        del best_score

    if tumor_id in present:
        for x0, x1, y0, y1, z0, z1, val in _smooth_trilinear_blocks(seg_res_ras == tumor_id, target_shape):
            score_u8 = np.rint(np.clip(val, np.float32(0.0), np.float32(1.0)) * np.float32(255.0)).astype(np.uint8)
            out_sub = out_seg[x0:x1, y0:y1, z0:z1]
            out_sub[(score_u8 >= 128) & (out_sub == 0)] = np.uint8(tumor_id)
    return out_seg


def restore_preclipped_air(crop_raw, spacing_mm, slope, inter, fill_hu=-1000.0):
    """Restore -400 HU pre-clipped exterior air to `fill_hu` prior to trilinear resampling."""
    strides = [max(1, crop_raw.shape[i] // 160, int(round(0.40 / max(spacing_mm[i], 1e-4)))) for i in range(3)]
    eff_slope = slope if slope != 0.0 else 1.0
    sub_hu = np.asarray(crop_raw[::strides[0], ::strides[1], ::strides[2]], dtype=np.float32) * eff_slope + inter
    border_p10 = float(np.percentile(lateral_border_values(sub_hu), 10.0)) if sub_hu.size > 0 else -1000.0
    if not is_preclipped_air_hu(sub_hu, border_p10_hu=border_p10, wide_padded=True):
        return crop_raw, slope, inter, False
    hu = crop_raw.astype(np.float32, copy=True) * np.float32(eff_slope) + np.float32(inter)
    hu[(hu >= -415.0) & (hu <= -395.0)] = np.float32(fill_hu)
    return hu, 1.0, 0.0, True


def normalize_hu(ct_dhw, hu_windows, harmonize_air=True, soft_tissue_target_hu=45.0, max_shift_hu=25.0):
    """Harmonize air floor, anchor soft tissue toward +45 HU, and scale HU windows to [0, 1]."""
    hu = np.nan_to_num(np.asarray(ct_dhw, dtype=np.float32), copy=True, nan=-1000.0, posinf=5000.0, neginf=-1000.0)
    if harmonize_air and is_preclipped_air_hu(hu):
        hu[(hu >= -405.0) & (hu <= -398.5)] = -1000.0
    if soft_tissue_target_hu is not None and max_shift_hu > 0.0:
        soft_mask = (hu >= -60.0) & (hu <= 160.0)
        if int(np.count_nonzero(soft_mask)) >= 64:
            shift = float(np.clip(float(soft_tissue_target_hu) - float(hu[soft_mask].mean(dtype=np.float64)), -float(max_shift_hu), float(max_shift_hu)))
            gate = np.clip((hu + np.float32(350.0)) * np.float32(1.0 / 150.0), np.float32(0.0), np.float32(1.0))
            hu += gate * np.float32(shift)

    channels = []
    for lo, hi in hu_windows:
        inv_span = np.float32(1.0 / max(float(hi) - float(lo), 1e-6))
        scaled = (np.clip(hu, np.float32(lo), np.float32(hi)) - np.float32(lo)) * inv_span
        channels.append(np.ascontiguousarray(scaled, dtype=np.float32))
    return np.stack(channels, axis=0)


def predict_sliding_window(session, input_channels, num_oar_classes=12, tumor_only=False, window_slices=128, overlap=0.25, divisor=16):
    """Run 3D U-Net inference across overlapping axial sliding windows with raised-cosine blending."""
    input_name = session.get_inputs()[0].name
    c_in, num_slices, h_in, w_in = input_channels.shape
    h_pad = int(math.ceil(h_in / divisor) * divisor)
    w_pad = int(math.ceil(w_in / divisor) * divisor)

    padded = np.zeros((c_in, num_slices, h_pad, w_pad), dtype=np.float32)
    padded[:, :, :h_in, :w_in] = input_channels

    def run_window(batch_5d):
        outputs = session.run(None, {input_name: batch_5d})
        return (np.zeros_like(outputs[0]), outputs[0]) if len(outputs) == 1 else (outputs[0], outputs[1])

    # Single-pass fast path when volume fits inside one window
    d_single = int(math.ceil(num_slices / divisor) * divisor)
    if d_single <= window_slices:
        batch_in = np.zeros((1, c_in, d_single, h_pad, w_pad), dtype=np.float32)
        batch_in[0, :, :num_slices] = padded
        oar_out, tum_out = run_window(batch_in)
        organ_logits = None if tumor_only else oar_out[0, :, :num_slices, :h_in, :w_in]
        return organ_logits, np.ascontiguousarray(tum_out[0, 0, :num_slices, :h_in, :w_in], dtype=np.float32)

    # Overlapping axial sliding-window schedule
    win_d = max(divisor * 4, int(math.floor(min(256, window_slices) / divisor) * divisor))
    ov_d = 0 if overlap <= 0.0 else min(max(divisor, int(round((win_d * overlap) / divisor) * divisor)), win_d - divisor)
    step = max(divisor, win_d - ov_d)
    starts = list(range(0, max(1, num_slices - win_d + 1), step))
    if not starts or (starts[-1] + win_d) < num_slices:
        starts.append(max(0, num_slices - win_d))

    organ_logits = None if tumor_only else np.zeros((num_oar_classes, num_slices, h_in, w_in), dtype=np.float32)
    tumor_logits = np.zeros((num_slices, h_in, w_in), dtype=np.float32)
    weight_sum = np.zeros((num_slices, 1, 1), dtype=np.float32)
    win_buf = np.zeros((1, c_in, win_d, h_pad, w_pad), dtype=np.float32)

    for idx, z_start in enumerate(starts):
        z_end = min(num_slices, z_start + win_d)
        cur_d = z_end - z_start
        win_buf.fill(0.0)
        win_buf[0, :, :cur_d] = padded[:, z_start:z_end]
        oar_win, tum_win = run_window(win_buf)

        wz = np.ones(cur_d, dtype=np.float32)
        ramp = min(ov_d, cur_d // 2)
        if len(starts) > 1 and ramp > 1:
            t = 0.5 * (1.0 - np.cos(np.pi * (np.arange(ramp, dtype=np.float32) + 0.5) / float(ramp)))
            if idx > 0:
                wz[:ramp] = t
            if idx < len(starts) - 1:
                wz[-ramp:] = t[::-1]

        w3d = wz[:, None, None]
        if organ_logits is not None:
            organ_logits[:, z_start:z_end] += oar_win[0, :, :cur_d, :h_in, :w_in] * w3d[None]
        tumor_logits[z_start:z_end] += tum_win[0, 0, :cur_d, :h_in, :w_in] * w3d
        weight_sum[z_start:z_end] += w3d

    inv_w = 1.0 / np.maximum(weight_sum, 1e-6)
    if organ_logits is not None:
        organ_logits *= inv_w[None]
    tumor_logits *= inv_w
    return organ_logits, tumor_logits


def compute_structure_stats(raw_3d, slope, inter, seg_mask_3d, voxel_spacing_mm, tumor_only=False, body_mask_3d=None):
    """Compute voxel counts, physical volumes (mm^3, cm^3), and mean HU on the native CT grid."""
    counts = np.zeros(13, dtype=np.int64)
    raw_sums = np.zeros(13, dtype=np.float64)
    body_vox = 0
    body_raw_sum = 0.0
    nz = seg_mask_3d.shape[2]
    for z0 in range(0, nz, 32):
        z1 = min(nz, z0 + 32)
        s_flat = seg_mask_3d[:, :, z0:z1].ravel()
        v_flat = raw_3d[:, :, z0:z1].ravel().astype(np.float64, copy=False)
        counts += np.bincount(s_flat, minlength=13)[:13]
        raw_sums += np.bincount(s_flat, weights=v_flat, minlength=13)[:13]
        if body_mask_3d is not None:
            b_flat = body_mask_3d[:, :, z0:z1].ravel()
            body_vox += int(np.count_nonzero(b_flat))
            body_raw_sum += float(v_flat[b_flat].sum())

    eff_slope = slope if slope != 0.0 else 1.0
    vox_mm3 = float(np.prod(voxel_spacing_mm))
    structure_ids = [12] if tumor_only else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

    rows = []
    for sid in structure_ids:
        vox = int(counts[sid])
        mm3 = vox * vox_mm3
        mean_hu = (float(raw_sums[sid]) / vox) * eff_slope + inter if vox > 0 else None
        rows.append({
            "id": sid,
            "structure": STRUCTURE_NAMES[sid],
            "voxels": vox,
            "volume_mm3": mm3,
            "volume_cm3": mm3 / 1000.0,
            "mean_hu": mean_hu,
        })

    if body_mask_3d is not None:
        b_mm3 = body_vox * vox_mm3
        b_hu = (body_raw_sum / body_vox) * eff_slope + inter if body_vox > 0 else None
        rows.append({
            "id": "body",
            "structure": "Body",
            "voxels": body_vox,
            "volume_mm3": b_mm3,
            "volume_cm3": b_mm3 / 1000.0,
            "mean_hu": b_hu,
        })
    return rows


def segment_ct_volume(
    input_path,
    session,
    config,
    output_path=None,
    body_mask_path=None,
    compute_body=False,
    orientation_mode="auto",
):
    """Run the 5-step 3D mouse CT segmentation pipeline on a single NIfTI volume."""
    t_start = time.perf_counter()
    need_body = (body_mask_path is not None) or bool(compute_body)
    tumor_only = bool(config.get("tumor_only", False))
    tumor_id = int(config.get("tumor_label_id", 12))

    # 1. Load NIfTI, reorient to RAS, strip external composite cradles, and align mouse pose
    ct_img, raw_3d, orig_slope, orig_inter, geom_affine, orig_spacing_mm = load_ct_nifti(input_path)
    raw_ras, ras_spacing_mm, from_ras = reorient_to_ras(raw_3d, geom_affine)
    raw_ras, ras_spacing_mm, slope, inter, perm, k_rot90, z_flip = estimate_mouse_pose_and_mask_cradle(
        raw_ras, ras_spacing_mm, slope=orig_slope, inter=orig_inter, auto_pose=(orientation_mode == "auto")
    )

    # 2. Crop exterior air around the mouse and resample to 0.20 mm
    x0, x1, y0, y1, z0, z1 = find_mouse_bounding_box(raw_ras, ras_spacing_mm, slope=slope, inter=inter)
    crop_raw = raw_ras[x0:x1, y0:y1, z0:z1]
    target_sp = config.get("target_spacing_mm", [0.2, 0.2, 0.2])
    res_shape = tuple(
        int(dim) if abs(sp_in - sp_tgt) / max(sp_tgt, 1e-6) <= 0.03 else max(1, int(round(dim * sp_in / sp_tgt)))
        for dim, sp_in, sp_tgt in zip(crop_raw.shape, ras_spacing_mm, target_sp)
    )
    eff_spacing = tuple(crop_raw.shape[i] * ras_spacing_mm[i] / max(res_shape[i], 1) for i in range(3))

    if config.get("harmonize_air_floor", True):
        crop_net, net_slope, net_inter, did_net_harm = restore_preclipped_air(crop_raw, ras_spacing_mm, slope, inter, fill_hu=-1000.0)
    else:
        crop_net, net_slope, net_inter, did_net_harm = crop_raw, slope, inter, False
    ct_res_ras = resample_volume(crop_net, res_shape, slope=net_slope, inter=net_inter)

    body_hu_ras = None
    did_body_harm = False
    if need_body:
        crop_b, b_slope, b_inter, did_body_harm = restore_preclipped_air(crop_raw, ras_spacing_mm, slope, inter, fill_hu=-650.0)
        body_hu_ras = resample_volume(crop_b, res_shape, slope=b_slope, inter=b_inter) if did_body_harm else ct_res_ras.copy()

    # 3. Normalize HU windows in (Z, X, Y) order for the 3D U-Net
    ct_dhw = np.ascontiguousarray(np.transpose(ct_res_ras, (2, 0, 1)), dtype=np.float32)
    del ct_res_ras
    input_channels = normalize_hu(
        ct_dhw,
        hu_windows=config["hu_windows"],
        harmonize_air=bool(config.get("harmonize_air_floor", True)),
        soft_tissue_target_hu=config.get("soft_tissue_calib_target_hu", 45.0),
        max_shift_hu=float(config.get("soft_tissue_max_shift_hu", 25.0)),
    )
    if input_channels.shape[0] != int(config["in_channels"]):
        input_channels = np.repeat(input_channels[:1], int(config["in_channels"]), axis=0)

    # 4. Run 3D U-Net inference across overlapping axial sliding windows
    t_ort = time.perf_counter()
    organ_logits, tumor_logits = predict_sliding_window(
        session,
        input_channels,
        num_oar_classes=int(config["num_oar_classes"]),
        tumor_only=tumor_only,
        window_slices=int(config.get("window_slices", 128)),
        overlap=float(config.get("window_overlap", 0.25)),
        divisor=int(config.get("spatial_divisor", 16)),
    )
    ort_seconds = time.perf_counter() - t_ort
    del input_channels

    # 5. Clean connected components and map back to the original CT voxel grid
    tumor_mask_dhw, tumor_probs_dhw = clean_tumor_mask(tumor_logits, eff_spacing, ct_dhw, config)
    del tumor_logits
    if tumor_only:
        seg_dhw = np.where(tumor_mask_dhw, np.uint8(tumor_id), np.uint8(0))
    else:
        organ_mask_dhw, organ_max_prob_dhw = finalize_organ_predictions(organ_logits, config, ct_hu_dhw=ct_dhw)
        del organ_logits
        organ_mask_dhw = clean_organ_masks(organ_mask_dhw, (eff_spacing[2], eff_spacing[0], eff_spacing[1]), ct_dhw)
        seg_dhw = resolve_organ_tumor_overlap(organ_mask_dhw, organ_max_prob_dhw, tumor_mask_dhw, tumor_probs_dhw, tumor_id=tumor_id)
    del ct_dhw, tumor_probs_dhw

    seg_res_ras = np.ascontiguousarray(np.transpose(seg_dhw, (1, 2, 0)))
    if not tumor_only:
        keep_largest_components(seg_res_ras, exclude_labels=(tumor_id,))

    full_ras_seg = np.zeros(raw_ras.shape, dtype=np.uint8)
    full_ras_seg[x0:x1, y0:y1, z0:z1] = smooth_and_upsample_masks(seg_res_ras, crop_raw.shape, tumor_id=tumor_id)
    if not tumor_only:
        keep_largest_components(full_ras_seg, exclude_labels=(tumor_id,))

    def invert_to_input_grid(arr_ras):
        out = arr_ras
        if z_flip:
            out = np.flip(out, axis=2)
        if k_rot90:
            out = np.rot90(out, k=(-k_rot90) % 4, axes=(0, 1))
        if perm != (0, 1, 2):
            out = np.transpose(out, tuple(int(i) for i in np.argsort(perm)))
        return nib.orientations.apply_orientation(out, from_ras)

    seg_native_3d = invert_to_input_grid(full_ras_seg)

    body_native_3d = None
    if need_body and body_hu_ras is not None:
        strides = [max(1, raw_ras.shape[i] // 160, int(round(0.40 / max(ras_spacing_mm[i], 1e-4)))) for i in range(3)]
        border_vals = lateral_border_values(raw_ras[::strides[0], ::strides[1], ::strides[2]])
        is_cal_hu = bool((float(np.percentile(border_vals, 10.0)) * slope + inter) < -350.0) if border_vals.size > 0 else True
        body_res_ras = compute_body_mask(
            body_hu_ras,
            seg_res_ras,
            eff_spacing,
            is_calibrated_hu=is_cal_hu,
            pre_harmonized_air=bool(did_net_harm or did_body_harm),
        )
        full_ras_body = np.zeros(raw_ras.shape, dtype=bool)
        full_ras_body[x0:x1, y0:y1, z0:z1] = smooth_and_upsample_binary(body_res_ras, crop_raw.shape)
        body_native_3d = invert_to_input_grid(full_ras_body) | (seg_native_3d > 0)

    if output_path is not None:
        save_nifti_mask(seg_native_3d, ct_img, output_path)
    if body_mask_path is not None and body_native_3d is not None:
        save_nifti_mask(body_native_3d, ct_img, body_mask_path)

    stats = compute_structure_stats(
        raw_3d, orig_slope, orig_inter, seg_native_3d, orig_spacing_mm, tumor_only=tumor_only, body_mask_3d=body_native_3d
    )
    return {
        "input_path": str(input_path),
        "output_path": str(output_path) if output_path is not None else None,
        "body_mask_path": str(body_mask_path) if body_mask_path is not None else None,
        "pose": {"perm": list(perm), "k_rot90": int(k_rot90), "z_flip": bool(z_flip)},
        "total_seconds": time.perf_counter() - t_start,
        "ort_seconds": ort_seconds,
        "stats": stats,
    }
