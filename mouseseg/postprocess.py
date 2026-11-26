"""Postprocessing for 3D mouse organ and subcutaneous tumor segmentation masks."""

import math
import numpy as np
from scipy import ndimage

STRUCTURE_NAMES = {
    1: "Bladder",
    2: "Lung",
    3: "Heart",
    4: "Liver",
    5: "Intestine",
    6: "Kidney",
    7: "Spleen",
    8: "Stomach",
    9: "Trachea",
    10: "Bone",
    12: "Subcutaneous tumor",
}

# Maximum connected components per organ (0 = unlimited for skeletal bone)
ORGAN_MAX_COMPONENTS = {1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 0}

# Per-organ component rules: (min_volume_mm3, min_fraction_of_largest, max_components)
ORGAN_COMPONENT_RULES = {
    1: (2.0, 0.20, 1),   # Bladder
    2: (3.0, 0.05, 2),   # Lung (left + right)
    3: (2.0, 0.20, 1),   # Heart
    4: (2.0, 0.00, 1),   # Liver
    5: (6.0, 0.02, 1),   # Intestine
    6: (3.0, 0.00, 2),   # Kidney (left + right)
    7: (2.0, 0.20, 1),   # Spleen
    8: (2.0, 0.20, 1),   # Stomach
    9: (0.5, 0.10, 1),   # Trachea
    10: (1.0, 0.00, 0),  # Bone
}

CONN_6 = ndimage.generate_binary_structure(3, 1)
CONN_26 = ndimage.generate_binary_structure(3, 3)


def tight_bbox_slices(mask_3d):
    """Return (z_slice, x_slice, y_slice) tightly bounding True voxels, or None if empty."""
    zs = np.flatnonzero(mask_3d.any(axis=(1, 2)))
    if zs.size == 0:
        return None
    z0, z1 = int(zs[0]), int(zs[-1]) + 1
    sub_z = mask_3d[z0:z1]
    xs = np.flatnonzero(sub_z.any(axis=(0, 2)))
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    ys = np.flatnonzero(sub_z[:, x0:x1].any(axis=(0, 1)))
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    return slice(z0, z1), slice(x0, x1), slice(y0, y1)


def keep_largest_components(mask_3d, max_components=None, exclude_labels=(12,)):
    """Keep at most top-k largest 6-connected components per organ label."""
    rules = ORGAN_MAX_COMPONENTS if max_components is None else max_components
    present = set(int(c) for c in np.unique(mask_3d).tolist())
    for organ_id, max_k in rules.items():
        if max_k <= 0 or organ_id in exclude_labels or organ_id not in present:
            continue
        organ_binary = mask_3d == organ_id
        bbox = tight_bbox_slices(organ_binary)
        if bbox is None:
            continue
        sub_mask = organ_binary[bbox]
        labeled, num_cc = ndimage.label(sub_mask, structure=CONN_6)
        if num_cc <= max_k:
            continue
        counts = np.bincount(labeled.ravel())[1:]
        keep_ids = np.argsort(-counts, kind="stable")[:max_k] + 1
        drop_voxels = sub_mask & (~np.isin(labeled, keep_ids))
        mask_3d[bbox][drop_voxels] = 0
    return mask_3d


def finalize_organ_predictions(organ_logits, config, ct_hu_dhw=None):
    """Convert (C, Z, X, Y) organ logits into uint8 organ labels and max softmax probabilities."""
    num_classes, nz, nx, ny = organ_logits.shape
    organ_mask = np.argmax(organ_logits, axis=0).astype(np.uint8)
    organ_max_prob = np.empty((nz, nx, ny), dtype=np.float16)

    bladder_thresh = config.get("recommended_bladder_prob_threshold")
    stomach_seed = config.get("recommended_stomach_seed_threshold")
    stomach_grow = config.get("recommended_stomach_grow_threshold")
    use_stomach_hyst = stomach_grow is not None and 0.0 < float(stomach_grow) < 1.0 and num_classes > 8
    stomach_flags = np.zeros((nz, nx, ny), dtype=np.uint8) if use_stomach_hyst else None

    min_probs = config.get("recommended_oar_min_prob") or {}
    lung_floor = float(min_probs.get("2", min_probs.get(2, 0.0)))
    lung_grow = 0.18
    use_lung_hyst = ct_hu_dhw is not None and num_classes > 2 and lung_floor > lung_grow
    lung_flags = np.zeros((nz, nx, ny), dtype=np.uint8) if use_lung_hyst else None

    # Compute softmax in 32-slice chunks to keep peak memory small
    for z0 in range(0, nz, 32):
        z1 = min(nz, z0 + 32)
        chunk = organ_logits[:, z0:z1].astype(np.float32, copy=False)
        exp_logits = np.exp(chunk - np.max(chunk, axis=0, keepdims=True))
        denom = np.sum(exp_logits, axis=0)
        max_prob = np.max(exp_logits, axis=0) / denom
        cls_chunk = organ_mask[z0:z1]

        if bladder_thresh is not None and 0.0 < float(bladder_thresh) < 1.0 and num_classes > 1:
            bladder_prob = exp_logits[1] / denom
            promote = ((cls_chunk == 0) | (cls_chunk == 5)) & (bladder_prob >= float(bladder_thresh))
            cls_chunk[promote] = 1
            max_prob[promote] = bladder_prob[promote]

        if stomach_flags is not None:
            stomach_prob = exp_logits[8] / denom
            eligible = (cls_chunk == 0) | (cls_chunk == 5) | (cls_chunk == 8)
            grow_mask = eligible & (stomach_prob >= float(stomach_grow))
            stomach_flags[z0:z1][grow_mask] = 1
            seed_thr = float(stomach_seed) if (stomach_seed is not None and float(stomach_seed) > float(stomach_grow)) else float(stomach_grow)
            stomach_flags[z0:z1][grow_mask & (stomach_prob >= seed_thr)] = 2

        if lung_flags is not None and ct_hu_dhw is not None:
            lung_prob = exp_logits[2] / denom
            hu_chunk = ct_hu_dhw[z0:z1]
            seed_lung = (cls_chunk == 2) & (lung_prob >= lung_floor)
            grow_lung = (
                ((cls_chunk == 0) | (cls_chunk == 2))
                & (~seed_lung)
                & (lung_prob >= lung_grow)
                & (hu_chunk >= -450.0)
                & (hu_chunk <= 100.0)
            )
            lung_flags[z0:z1][grow_lung] = 1
            lung_flags[z0:z1][seed_lung] = 2

        organ_max_prob[z0:z1] = max_prob.astype(np.float16)

    # 3D connected hysteresis for stomach boundary recovery
    if stomach_flags is not None:
        candidates = (organ_mask == 8) | (stomach_flags >= 1)
        bbox = tight_bbox_slices(candidates)
        if bbox is not None:
            sub_cand = candidates[bbox]
            sub_seed = sub_cand & (stomach_flags[bbox] >= 2)
            if np.any(sub_seed):
                labeled, num_cc = ndimage.label(sub_cand, structure=CONN_6)
                seed_counts = np.bincount(labeled[sub_seed], minlength=num_cc + 1)[1:]
                total_counts = np.bincount(labeled.ravel())[1:]
                keep_ids = [
                    i + 1 for i in range(num_cc)
                    if int(seed_counts[i]) >= min(4, max(1, int(total_counts[i]) // 2))
                ]
                if keep_ids:
                    sub_cls = organ_mask[bbox]
                    promote = ((sub_cls == 0) | (sub_cls == 5)) & np.isin(labeled, keep_ids)
                    sub_cls[promote] = 8

    # 3D connected hysteresis for consolidated fibrotic lung lobes (interior core check rejects 1-voxel pleural halos)
    if lung_flags is not None:
        l_bbox = tight_bbox_slices(lung_flags >= 1)
        if l_bbox is not None:
            sub_lf = lung_flags[l_bbox]
            seed_m = sub_lf == 2
            grow_m = sub_lf == 1
            if np.any(seed_m) and np.any(grow_m):
                eroded_grow = ndimage.binary_erosion(grow_m, structure=CONN_6, border_value=0)
                if np.any(eroded_grow):
                    lbl_g, num_g = ndimage.label(grow_m, structure=CONN_6)
                    core_counts = np.bincount(lbl_g[eroded_grow], minlength=num_g + 1)[1:]
                    valid_g_ids = np.flatnonzero(core_counts >= 8) + 1
                    if valid_g_ids.size > 0:
                        fibrotic_lobes = np.isin(lbl_g, valid_g_ids)
                        labeled_l, num_l = ndimage.label(seed_m | fibrotic_lobes, structure=CONN_6)
                        l_seeds = np.bincount(labeled_l[seed_m], minlength=num_l + 1)[1:]
                        keep_l = np.flatnonzero(l_seeds >= 16) + 1
                        if keep_l.size > 0:
                            kept_fibrotic = fibrotic_lobes & np.isin(labeled_l, keep_l)
                            if np.any(kept_fibrotic):
                                organ_mask[l_bbox][kept_fibrotic] = 2
                                organ_max_prob[l_bbox][kept_fibrotic] = np.maximum(
                                    organ_max_prob[l_bbox][kept_fibrotic], np.float16(lung_floor)
                                )

    # Apply per-organ minimum probability floors (e.g. Lung, Intestine, Trachea)
    for organ_key, prob_floor in min_probs.items():
        organ_id, floor_val = int(organ_key), float(prob_floor)
        if 1 <= organ_id < num_classes and floor_val > 0.0:
            organ_mask[(organ_mask == organ_id) & (organ_max_prob < floor_val)] = 0

    organ_mask[organ_mask == 11] = 0
    return organ_mask, organ_max_prob


def clean_tumor_mask(tumor_logits, voxel_spacing_mm, ct_hu_dhw, config):
    """Apply Gaussian smoothing, 3D hysteresis thresholding, and component filters to tumor logits."""
    clipped = np.clip(np.asarray(tumor_logits, dtype=np.float32), -30.0, 30.0)
    tumor_probs = (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)
    smooth_sigma = float(config.get("tumor_smooth_sigma", 0.5))
    if smooth_sigma > 0.0:
        tumor_probs = ndimage.gaussian_filter(tumor_probs, sigma=smooth_sigma).astype(np.float32, copy=False)

    seed_thr = config.get("recommended_tumor_seed_threshold")
    grow_thr = config.get("recommended_tumor_grow_threshold")
    plain_thr = float(config.get("recommended_tumor_threshold", 0.5))
    use_hysteresis = seed_thr is not None and grow_thr is not None and float(seed_thr) > float(grow_thr)

    candidates = tumor_probs >= (float(grow_thr) if use_hysteresis else plain_thr)
    if not np.any(candidates):
        return np.zeros_like(candidates, dtype=bool), tumor_probs

    labeled, num_cc = ndimage.label(candidates, structure=CONN_26)
    if num_cc == 0:
        return np.zeros_like(candidates, dtype=bool), tumor_probs

    seed_counts = None
    if use_hysteresis:
        seeds = (tumor_probs >= float(seed_thr)) & (labeled > 0)
        if not np.any(seeds):
            return np.zeros_like(candidates, dtype=bool), tumor_probs
        seed_counts = np.bincount(labeled[seeds], minlength=num_cc + 1)[1:]

    vox_vol_mm3 = float(np.prod(voxel_spacing_mm))
    min_vol_mm3 = float(config.get("recommended_min_tumor_vol_mm3", 6.0))
    rel_ratio = float(config.get("recommended_rel_tumor_cc_ratio", 0.1))
    min_seed_vox = int(config.get("recommended_tumor_min_seed_voxels", 4))

    counts = np.bincount(labeled.ravel())[1:]
    min_vox = max(1, int(math.ceil(min_vol_mm3 / max(vox_vol_mm3, 1e-6)))) if min_vol_mm3 > 0 else 1
    bilateral_cap = max(min_vox, int(math.ceil(10.0 / max(vox_vol_mm3, 1e-6))))
    rel_min_vox = min(bilateral_cap, int(math.ceil(rel_ratio * int(counts.max())))) if rel_ratio > 0 else 0
    eff_min_vox = max(min_vox, rel_min_vox)

    slices = ndimage.find_objects(labeled)
    keep_ids = []
    for idx, vox_count in enumerate(counts.tolist()):
        comp_id = idx + 1
        if vox_count < eff_min_vox:
            continue
        if seed_counts is not None and int(seed_counts[idx]) < min(max(1, min_seed_vox), max(1, vox_count // 2)):
            continue
        comp_sl = slices[idx]
        if ct_hu_dhw is not None and comp_sl is not None:
            mean_hu = float(ct_hu_dhw[comp_sl][labeled[comp_sl] == comp_id].mean())
            if mean_hu < -280.0 or mean_hu > 700.0:
                continue
        keep_ids.append(comp_id)

    if not keep_ids:
        return np.zeros_like(candidates, dtype=bool), tumor_probs
    return np.isin(labeled, keep_ids), tumor_probs


def _clean_pelvic_intestine(organ_mask, ct_hu_dhw, spacing_dhw_mm):
    """Remove thin-necked or exterior-cleft pelvic Intestine leaks below the abdominal anchor."""
    sz, sx, sy = spacing_dhw_mm
    int_z = np.flatnonzero((organ_mask == 5).any(axis=(1, 2)))
    if int_z.size == 0:
        return
    iz0, iz1 = int(int_z[0]), int(int_z[-1]) + 1
    open_iters = max(1, int(round(0.40 / max(min(sz, sx, sy), 1e-4))))
    margin_slices = int(round(3.0 / max(sz, 1e-4)))

    bladder_z = np.flatnonzero((organ_mask == 1).any(axis=(1, 2)))
    if bladder_z.size > 0:
        z_anchor = int(bladder_z[-1]) + 1 + margin_slices
    elif iz1 - iz0 >= int(round(12.0 / max(sz, 1e-4))):
        z_anchor = iz0 + margin_slices
    else:
        return
    if iz0 >= z_anchor or iz1 <= z_anchor + open_iters + 2:
        return

    z_lo = max(0, iz0 - 1)
    z_hi = min(organ_mask.shape[0], z_anchor + open_iters + 2)
    bbox = tight_bbox_slices(organ_mask[z_lo:z_hi] == 5)
    if bbox is None:
        return
    x0 = max(0, bbox[1].start - 1)
    x1 = min(organ_mask.shape[1], bbox[1].stop + 1)
    y0 = max(0, bbox[2].start - 1)
    y1 = min(organ_mask.shape[2], bbox[2].stop + 1)

    sub_org = organ_mask[z_lo:z_hi, x0:x1, y0:y1]
    sub_int = sub_org == 5
    anchor_local = z_anchor - z_lo

    if ct_hu_dhw is not None and anchor_local > 0:
        sub_hu = ct_hu_dhw[z_lo : z_lo + anchor_local, x0:x1, y0:y1]
        pelvic_org = sub_org[:anchor_local]
        pelvic_int = sub_int[:anchor_local]
        air_seed = (pelvic_org == 0) & (sub_hu < -320.0)
        air_grow = pelvic_int & (sub_hu < -250.0)
        if air_seed.any() and air_grow.any():
            lbl_air, num_air = ndimage.label(air_seed | air_grow, structure=CONN_6)
            if num_air > 0:
                reached = np.isin(lbl_air, np.unique(lbl_air[air_seed])) & (lbl_air > 0)
                pelvic_org[pelvic_int & reached] = 0
                sub_int = sub_org == 5

    if not sub_int[:anchor_local].any():
        return
    eroded = ndimage.binary_erosion(sub_int, structure=CONN_6, iterations=open_iters, border_value=0)
    eroded[anchor_local:] = sub_int[anchor_local:]
    lbl_core, num_core = ndimage.label(eroded, structure=CONN_6)
    if num_core == 0:
        return
    anchor_ids = np.unique(lbl_core[anchor_local:][sub_int[anchor_local:]])
    anchor_ids = anchor_ids[anchor_ids > 0]
    if anchor_ids.size == 0:
        return
    kept_core = np.isin(lbl_core, anchor_ids)
    dilated = ndimage.binary_dilation(kept_core, structure=CONN_6, iterations=open_iters)
    dilated[anchor_local:] = True
    sub_org[sub_int & (~dilated)] = 0

    final_int = sub_org == 5
    if final_int[:anchor_local].any():
        lbl_fin, _ = ndimage.label(final_int, structure=CONN_6)
        fin_ids = np.unique(lbl_fin[anchor_local:][final_int[anchor_local:]])
        fin_ids = fin_ids[fin_ids > 0]
        sub_org[final_int & (~np.isin(lbl_fin, fin_ids))] = 0


def clean_organ_masks(organ_mask, spacing_dhw_mm, ct_hu_dhw=None):
    """Remove anatomically implausible organ islands on a (Z, X, Y) label map."""
    sz, sx, sy = (float(s) for s in spacing_dhw_mm)
    vox_mm3 = sz * sx * sy

    # 1. Trachea airway check: keep trachea voxels near air (< -300 HU)
    if ct_hu_dhw is not None:
        tr_bbox = tight_bbox_slices(organ_mask == 9)
        if tr_bbox is not None:
            tr_mask = organ_mask[tr_bbox] == 9
            tr_air = tr_mask & (ct_hu_dhw[tr_bbox] < -300.0)
            if float(np.count_nonzero(tr_air)) * vox_mm3 < 0.3:
                organ_mask[tr_bbox][tr_mask] = 0
            else:
                wall_iters = max(1, int(round(0.25 / max(min(sz, sx, sy), 1e-4))))
                near_air = ndimage.binary_dilation(tr_air, structure=CONN_6, iterations=wall_iters, mask=tr_mask)
                organ_mask[tr_bbox][tr_mask & (~near_air)] = 0

    # 2. Locate thorax Z range from Heart (3, requiring >= 35 mm^3) or fallback to Lung (2)
    heart_z_mm = None
    for ref_id in (3, 2):
        ref_bbox = tight_bbox_slices(organ_mask == ref_id)
        if ref_bbox is None:
            continue
        lbl_ref, num_ref = ndimage.label(organ_mask[ref_bbox] == ref_id, structure=CONN_6)
        if num_ref == 0:
            continue
        ref_counts = np.bincount(lbl_ref.ravel())[1:]
        big_idx = int(np.argmax(ref_counts))
        if ref_id == 3 and float(ref_counts[big_idx]) * vox_mm3 < 35.0:
            continue
        big_sl = ndimage.find_objects(lbl_ref)[big_idx][0]
        z_lo = (ref_bbox[0].start + big_sl.start) * sz
        z_hi = (ref_bbox[0].start + big_sl.stop - 1) * sz
        heart_z_mm = (z_lo, 0.5 * (z_lo + z_hi), z_hi)
        break

    # 3. Per-organ connected-component filtering and enclosed pinhole filling
    removed_pinholes = []
    for organ_id, (min_abs_mm3, min_rel, max_k) in ORGAN_COMPONENT_RULES.items():
        bbox = tight_bbox_slices(organ_mask == organ_id)
        if bbox is None:
            continue
        sub_bin = organ_mask[bbox] == organ_id
        labeled, num_cc = ndimage.label(sub_bin, structure=CONN_6)
        if num_cc == 0:
            continue

        counts = np.bincount(labeled.ravel())[1:].astype(np.float64)
        volumes = counts * vox_mm3
        comp_slices = ndimage.find_objects(labeled)
        allowed = np.ones(num_cc, dtype=bool)

        for idx in range(num_cc):
            c_sl = comp_slices[idx]
            z_center_mm = (bbox[0].start + 0.5 * (c_sl[0].start + c_sl[0].stop - 1)) * sz
            if organ_id == 2:  # Lung
                if heart_z_mm is not None and not (heart_z_mm[0] - 8.0 <= z_center_mm <= heart_z_mm[2] + 8.0):
                    allowed[idx] = False
                elif ct_hu_dhw is not None:
                    hu_vals = ct_hu_dhw[bbox][c_sl][labeled[c_sl] == (idx + 1)]
                    if float(hu_vals.mean()) > 80.0:
                        allowed[idx] = False
            elif organ_id == 9:  # Trachea
                if volumes[idx] > 80.0 or (heart_z_mm is not None and z_center_mm < heart_z_mm[1] - 2.0):
                    allowed[idx] = False
                elif ct_hu_dhw is not None:
                    hu_vals = ct_hu_dhw[bbox][c_sl][labeled[c_sl] == (idx + 1)]
                    if float(np.mean(hu_vals < -250.0)) < 0.50 or float(hu_vals.mean()) > -280.0:
                        allowed[idx] = False
            elif organ_id == 5 and heart_z_mm is not None:  # Intestine
                if z_center_mm > heart_z_mm[1] and volumes[idx] < 0.15 * float(volumes.max()):
                    allowed[idx] = False

        if not allowed.any() and organ_id not in (2, 9):
            allowed[:] = True

        keep = np.zeros(num_cc, dtype=bool)
        if allowed.any():
            vol_thresh = max(min_abs_mm3, min_rel * float(volumes[allowed].max()))
            kept_so_far = 0
            for idx in np.argsort(-volumes).tolist():
                if not allowed[idx]:
                    continue
                if kept_so_far == 0 or volumes[idx] >= vol_thresh:
                    keep[idx] = True
                    kept_so_far += 1
                    if max_k > 0 and kept_so_far >= max_k:
                        break

        if not keep.all():
            for idx in np.flatnonzero(~keep).tolist():
                c_sl = comp_slices[idx]
                local_m = labeled[c_sl] == (idx + 1)
                gz = slice(bbox[0].start + c_sl[0].start, bbox[0].start + c_sl[0].stop)
                gx = slice(bbox[1].start + c_sl[1].start, bbox[1].start + c_sl[1].stop)
                gy = slice(bbox[2].start + c_sl[2].start, bbox[2].start + c_sl[2].stop)
                organ_mask[gz, gx, gy][local_m] = 0
                if volumes[idx] < 3.0:
                    removed_pinholes.append((gz, gx, gy, local_m.copy()))

    # 4. Clean pelvic intestine boundary and fill tiny enclosed pinholes
    _clean_pelvic_intestine(organ_mask, ct_hu_dhw, spacing_dhw_mm)

    nz, nx, ny = organ_mask.shape
    for gz, gx, gy, hole_mask in removed_pinholes:
        z0, z1 = max(0, gz.start - 1), min(nz, gz.stop + 1)
        x0, x1 = max(0, gx.start - 1), min(nx, gx.stop + 1)
        y0, y1 = max(0, gy.start - 1), min(ny, gy.stop + 1)
        padded_hole = np.zeros((z1 - z0, x1 - x0, y1 - y0), dtype=bool)
        padded_hole[gz.start - z0 : gz.stop - z0, gx.start - x0 : gx.stop - x0, gy.start - y0 : gy.stop - y0] = hole_mask
        ring = ndimage.binary_dilation(padded_hole, structure=CONN_6) & (~padded_hole)
        neighbors = organ_mask[z0:z1, x0:x1, y0:y1][ring]
        if neighbors.size > 0 and not np.any(neighbors == 0):
            fill_label = int(np.argmax(np.bincount(neighbors)))
            if 1 <= fill_label <= 10:
                organ_mask[z0:z1, x0:x1, y0:y1][padded_hole] = fill_label

    return keep_largest_components(organ_mask)


def resolve_organ_tumor_overlap(organ_mask, organ_max_prob, tumor_mask, tumor_probs, tumor_id=12, protect_internal_organs=True):
    """Merge organ and tumor masks while protecting internal organs (1..10) from tumor overwrite."""
    combined = organ_mask.copy()
    combined[(organ_mask == 0) & tumor_mask] = tumor_id
    overlap = (organ_mask > 0) & tumor_mask
    if np.any(overlap) and not protect_internal_organs and organ_max_prob is not None:
        tumor_wins = tumor_probs[overlap] > organ_max_prob[overlap].astype(np.float32)
        ov_indices = np.flatnonzero(overlap)
        combined.ravel()[ov_indices[tumor_wins]] = tumor_id
    return combined
