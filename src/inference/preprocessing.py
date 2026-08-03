"""
preprocessing.py

Preprocessing and reconstruction utilities for the HECKTOR 2026
end-to-end inference pipeline.

This module performs exactly the same preprocessing used during
training:

    1. Load CT and PET scans
    2. Resample PET onto the CT reference geometry
    3. Resample to 1×1×1 mm
    4. Determine scan type
    5. Anatomical prior crop
    6. Pad to fixed cube size
    7. CT normalization
    8. PET normalization

The module also provides utilities for reconstructing the predicted
segmentation back into the original CT coordinate system and saving
the final .mha file required by the HECKTOR challenge.
"""

import gc

import numpy as np
import SimpleITK as sitk

from scipy.ndimage import zoom


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SPACING = np.array([1.0, 1.0, 1.0])

CUBE_SIZE = np.array([224, 224, 224])

CT_MIN = -1024
CT_MAX = 1024

PRIORS = {

    "HEAD_NECK": {

        "x": 0.50,
        "y": 0.42,
        "z": 0.48
    },

    "FULL_BODY": {

        "x": 0.50,
        "y": 0.42,
        "z": 0.82
    }
}


# ============================================================
# IMAGE LOADING
# ============================================================

def read_sitk(path):
    """Load a medical image (.mha or .nii.gz) as a SimpleITK image."""

    return sitk.ReadImage(str(path))


def sitk_to_xyz_array(image):
    """
    Convert a SimpleITK image to a NumPy array in (x, y, z) order.

    Returns
    -------
    array : ndarray (x, y, z)
    spacing : ndarray (x, y, z)
    """

    array = sitk.GetArrayFromImage(image)

    # SimpleITK: (z, y, x) -> NumPy pipeline: (x, y, z)
    array = np.transpose(array, (2, 1, 0)).astype(np.float32)

    spacing = np.asarray(
        image.GetSpacing(),
        dtype=np.float32,
    )

    return array, spacing


def load_image(path):
    """
    Load a medical image (.mha or .nii.gz) and return

        image : ndarray (x, y, z)
        spacing : ndarray (x, y, z)
    """

    return sitk_to_xyz_array(read_sitk(path))


# ============================================================
# PET / CT ALIGNMENT
# ============================================================

def align_pet_to_ct(ct_image, pt_image):
    """
    Resample PET onto the CT voxel grid in physical space.

    Grand Challenge .mha inputs arrive on separate native grids.
    Training .nii.gz pairs are already co-registered; this step is
    idempotent when inputs already share CT geometry.
    """

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)

    return resampler.Execute(pt_image)


# ============================================================
# SCAN TYPE
# ============================================================

def classify_scan(size_z_mm):
    """
    Determine whether the scan is a head-neck scan or a full-body scan.
    """

    if size_z_mm >= 700:
        return "FULL_BODY"

    return "HEAD_NECK"


# ============================================================
# FIXED CROP
# ============================================================

def get_cube_bounds(shape,
                    center_xyz,
                    cube_size):
    """
    Compute crop bounds while keeping the crop inside the image.
    """

    center_xyz = np.asarray(center_xyz).astype(int)

    start = center_xyz - cube_size // 2
    end = start + cube_size

    for d in range(3):

        if start[d] < 0:

            end[d] -= start[d]
            start[d] = 0

        if end[d] > shape[d]:

            shift = end[d] - shape[d]

            start[d] -= shift

            end[d] = shape[d]

        start[d] = max(start[d], 0)

    return start, end


# ============================================================
# PADDING
# ============================================================

def pad_to_shape(arr,
                 target_shape,
                 pad_value):
    """
    Pad array symmetrically to target size.

    Returns
    -------
    padded : ndarray
    pad_before : ndarray (3,)
        Voxel offset before the original content in each axis.
    """

    pads = []

    for current, target in zip(arr.shape,
                               target_shape):

        total_pad = max(0, target - current)

        before = total_pad // 2
        after = total_pad - before

        pads.append((before, after))

    padded = np.pad(
        arr,
        pads,
        mode="constant",
        constant_values=pad_value
    )

    pad_before = np.array(
        [p[0] for p in pads],
        dtype=np.int32,
    )

    return padded, pad_before


def preprocess_patient(ct_path,
                       pt_path):
    """
    Preprocess a single CT/PET pair.

    Geometry matches training (full 1 mm resample → priors → 224 crop).
    Peak RAM is reduced by sequential CT then PET resampling and releasing
    large arrays as soon as possible. Full-volume zoom results use float32
    (same precision as validation). Model inputs remain float32 224³ crops.
    """

    ct_sitk = read_sitk(ct_path)
    pt_sitk = read_sitk(pt_path)
    pt_sitk = align_pet_to_ct(ct_sitk, pt_sitk)

    ct, ct_spacing = sitk_to_xyz_array(ct_sitk)
    pt, _ = sitk_to_xyz_array(pt_sitk)

    del ct_sitk, pt_sitk
    gc.collect()

    original_shape = ct.shape

    size_z_mm = (
        ct.shape[2]
        * ct_spacing[2]
    )

    scan_type = classify_scan(
        size_z_mm
    )

    zoom_factor = (
        ct_spacing /
        TARGET_SPACING
    )

    # ------------------------------------------------------------------
    # CT: full-volume resample (float32) → crop/pad → float32 crop
    # ------------------------------------------------------------------
    ct_res = zoom(
        ct,
        zoom_factor,
        order=1
    ).astype(np.float32)

    del ct
    gc.collect()

    resampled_shape = ct_res.shape

    sx, sy, sz = resampled_shape

    center_x = int(
        PRIORS[scan_type]["x"] * sx
    )

    center_y = int(
        PRIORS[scan_type]["y"] * sy
    )

    center_z = int(
        PRIORS[scan_type]["z"] * sz
    )

    start, end = get_cube_bounds(
        resampled_shape,
        (
            center_x,
            center_y,
            center_z
        ),
        CUBE_SIZE
    )

    crop_shape = end - start

    ct_crop = np.asarray(
        ct_res[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ],
        dtype=np.float32,
    )

    del ct_res
    gc.collect()

    ct_crop, pad_before = pad_to_shape(
        ct_crop,
        CUBE_SIZE,
        pad_value=CT_MIN
    )

    # ------------------------------------------------------------------
    # PET: full-volume resample (float32) → same crop/pad → float32 crop
    # ------------------------------------------------------------------
    pt_res = zoom(
        pt,
        zoom_factor,
        order=1
    ).astype(np.float32)

    del pt
    gc.collect()

    pt_crop = np.asarray(
        pt_res[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ],
        dtype=np.float32,
    )

    del pt_res
    gc.collect()

    pt_crop, _ = pad_to_shape(
        pt_crop,
        CUBE_SIZE,
        pad_value=0
    )

    ct_crop = np.clip(
        ct_crop,
        CT_MIN,
        CT_MAX
    )

    ct_crop = (
        ct_crop.astype(np.float32)
        / float(CT_MAX)
    )

    pt_mean = np.mean(pt_crop)
    pt_std = np.std(pt_crop)

    pt_crop = (
        pt_crop - pt_mean
    ) / (pt_std + 1e-8)

    pt_crop = pt_crop.astype(np.float32)

    return {

        "ct": ct_crop,
        "pet": pt_crop,

        "crop_start": start.astype(np.int32),
        "crop_end": end.astype(np.int32),

        "crop_shape": crop_shape.astype(np.int32),
        "pad_before": pad_before.astype(np.int32),
        "resampled_shape": np.array(resampled_shape),
        "original_shape": np.array(original_shape),

        "reference_ct_path": ct_path
    }


# ============================================================
# RECONSTRUCT TO ORIGINAL SPACE
# ============================================================

def reconstruct_to_original_space(pred_mask,
                                  metadata):
    """
    Reconstruct a predicted segmentation from the cropped,
    1 mm isotropic space back to the original CT space.
    """

    crop_shape = metadata["crop_end"] - metadata["crop_start"]
    pad_before = metadata["pad_before"]

    pred_crop = pred_mask[
        pad_before[0]:pad_before[0] + crop_shape[0],
        pad_before[1]:pad_before[1] + crop_shape[1],
        pad_before[2]:pad_before[2] + crop_shape[2],
    ]

    full_mask = np.zeros(
        metadata["resampled_shape"],
        dtype=np.uint8
    )

    s = metadata["crop_start"]
    e = metadata["crop_end"]

    full_mask[
        s[0]:e[0],
        s[1]:e[1],
        s[2]:e[2]
    ] = pred_crop

    reference = sitk.ReadImage(
        metadata["reference_ct_path"]
    )

    mask_img = sitk.GetImageFromArray(
        np.transpose(full_mask, (2, 1, 0))
    )

    mask_img.SetSpacing(
        tuple(TARGET_SPACING.tolist())
    )

    mask_img.SetOrigin(reference.GetOrigin())
    mask_img.SetDirection(reference.GetDirection())

    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(reference)

    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor
    )

    resampler.SetTransform(
        sitk.Transform()
    )

    resampler.SetDefaultPixelValue(0)

    reconstructed = resampler.Execute(
        mask_img
    )

    reconstructed = sitk.GetArrayFromImage(
        reconstructed
    )

    reconstructed = np.transpose(
        reconstructed,
        (2, 1, 0)
    ).astype(np.uint8)

    return reconstructed


# ============================================================
# SAVE MHA
# ============================================================

def save_mha(mask,
             reference_ct_path,
             output_path):
    """
    Save segmentation as .mha using the original CT geometry.
    """

    reference = sitk.ReadImage(
        reference_ct_path
    )

    mask_img = sitk.GetImageFromArray(
        np.transpose(mask, (2, 1, 0))
    )

    mask_img.CopyInformation(
        reference
    )

    sitk.WriteImage(
        mask_img,
        output_path,
        useCompression=True
    )
