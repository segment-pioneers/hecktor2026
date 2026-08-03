import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from scipy.ndimage import zoom
from argparse import ArgumentParser

from .networks import AdaMSS_Seg


# ============================================================
# CONFIG
# ============================================================

TARGET_SPACING = np.array([1.0, 1.0, 1.0])

CUBE_SIZE = np.array([224, 224, 224])

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

THRESHOLD = 0.5


# ============================================================
# HELPERS
# ============================================================

def classify_scan(size_z_mm):

    if size_z_mm >= 700:
        return "FULL_BODY"

    return "HEAD_NECK"

def dice_binary(gt, pred):

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    denominator = (gt.sum() + pred.sum())

    if denominator == 0:
        return 1.0

    return (2.0 * intersection / denominator)


def compute_hecktor_dice(
    gt_mask,
    pred_mask
):

    dice_gtvp = dice_binary(
        gt_mask == 1,
        pred_mask == 1
    )

    dice_gtvn = dice_binary(
        gt_mask == 2,
        pred_mask == 2
    )

    dice_mean = (
        dice_gtvp
        + dice_gtvn
    ) / 2.0

    return {
        "GTVp": dice_gtvp,
        "GTVn": dice_gtvn,
        "Mean": dice_mean
    }

def reconstruction_dice(
    original_mask,
    reconstructed_mask
):

    return compute_hecktor_dice(
        original_mask,
        reconstructed_mask
    )

def patch_dice(
    gt_crop,
    pred_crop
):

    return compute_hecktor_dice(
        gt_crop,
        pred_crop
    )

def original_space_dice(
    gt_original,
    pred_original
):

    return compute_hecktor_dice(
        gt_original,
        pred_original
    )


def get_cube_bounds(shape, center_xyz, cube_size):

    center_xyz = np.array(center_xyz).astype(int)

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


def pad_to_shape(arr, target_shape, pad_value=0):

    pads = []

    for current, target in zip(arr.shape, target_shape):

        total_pad = max(0, target - current)

        before = total_pad // 2
        after = total_pad - before

        pads.append((before, after))

    return np.pad(
        arr,
        pads,
        mode="constant",
        constant_values=pad_value
    )


def resample_mask_to_reference(mask_resampled, reference_ct_path):

    reference = sitk.ReadImage(reference_ct_path)

    mask_img = sitk.GetImageFromArray(mask_resampled.transpose(2, 1, 0))

    mask_img.SetSpacing(tuple(TARGET_SPACING))

    mask_img.SetOrigin(reference.GetOrigin())

    mask_img.SetDirection(reference.GetDirection())

    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(reference)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampler.SetDefaultPixelValue(0)

    mask_orig = resampler.Execute(mask_img)

    mask_np = sitk.GetArrayFromImage(mask_orig)

    mask_np = mask_np.transpose(2, 1, 0)

    return mask_np.astype(np.uint8)


def verify_mha(ct_path, gt_path, mha_path):

    ct = sitk.ReadImage(ct_path)
    pred = sitk.ReadImage(mha_path)

    print("CT size:", ct.GetSize())
    print("Pred size:", pred.GetSize())

    print("CT spacing:", ct.GetSpacing())
    print("Pred spacing:", pred.GetSpacing())

    print("CT origin:", ct.GetOrigin())
    print("Pred origin:", pred.GetOrigin())

    print("CT direction:", ct.GetDirection())
    print("Pred direction:", pred.GetDirection())

    assert ct.GetSize() == pred.GetSize()

    assert np.allclose(
        ct.GetSpacing(),
        pred.GetSpacing()
    )

    assert np.allclose(
        ct.GetOrigin(),
        pred.GetOrigin()
    )

    assert np.allclose(
        ct.GetDirection(),
        pred.GetDirection()
    )

    labels = np.unique(
        sitk.GetArrayFromImage(pred)
    )

    assert set(labels).issubset(
        {0,1,2}
    )

    gt = nib.load(gt_path).get_fdata().astype(np.uint8)
    pred = sitk.GetArrayFromImage(
        sitk.ReadImage(mha_path)
    )

    pred = pred.transpose(2,1,0)
    scores = original_space_dice(
        gt,
        pred
    )

    print(scores)
    print("Verification passed.")

# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path, device):

    model = AdaMSS_Seg()

    state = torch.load(
        model_path,
        map_location=device
    )

    if (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model


# ============================================================
# INFERENCE
# ============================================================

def predict_patient(
    ct_path,
    pt_path,
    gt_path,
    model,
    device
):

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    ct_img = nib.load(ct_path)
    pt_img = nib.load(pt_path)
    gt_img = nib.load(gt_path)

    ct = ct_img.get_fdata().astype(np.float32)
    pt = pt_img.get_fdata().astype(np.float32)
    gt = gt_img.get_fdata().astype(np.float32)

    original_shape = ct.shape

    spacing = np.array(
        ct_img.header.get_zooms()[:3]
    )

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    zoom_factor = spacing / TARGET_SPACING

    ct_res = zoom(
        ct,
        zoom_factor,
        order=1
    ).astype(np.float32)

    pt_res = zoom(
        pt,
        zoom_factor,
        order=1
    ).astype(np.float32)

    gt_res = zoom(
        gt,
        zoom_factor,
        order=0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # ANATOMICAL CROP
    # --------------------------------------------------------

    size_z_mm = (
        ct.shape[2]
        * spacing[2]
    )

    scan_type = classify_scan(
        size_z_mm
    )

    sx, sy, sz = ct_res.shape

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
        ct_res.shape,
        (
            center_x,
            center_y,
            center_z
        ),
        CUBE_SIZE
    )

    ct_crop = ct_res[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ]

    pt_crop = pt_res[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ]

    gt_crop = gt_res[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ]

    # --------------------------------------------------------
    # PAD
    # --------------------------------------------------------

    ct_crop = pad_to_shape(
        ct_crop,
        CUBE_SIZE,
        pad_value=-1024
    )

    pt_crop = pad_to_shape(
        pt_crop,
        CUBE_SIZE,
        pad_value=0
    )

    gt_crop = pad_to_shape(
        gt_crop,
        CUBE_SIZE,
        pad_value=0
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    ct_crop = np.clip(
        ct_crop,
        -1024,
        1024
    )

    ct_crop = ct_crop / 1024.0

    pt_mean = np.mean(pt_crop)
    pt_std = np.std(pt_crop)

    pt_crop = (
        pt_crop - pt_mean
    ) / (pt_std + 1e-8)

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    ct_tensor = torch.from_numpy(
        ct_crop
    ).unsqueeze(0).unsqueeze(0).float()

    pt_tensor = torch.from_numpy(
        pt_crop
    ).unsqueeze(0).unsqueeze(0).float()

    ct_tensor = ct_tensor.to(device)
    pt_tensor = pt_tensor.to(device)

    with torch.no_grad():

        tumor_pred, node_pred = model(
            pt_tensor,
            ct_tensor
        )

    tumor_pred = (
        tumor_pred
        .cpu()
        .numpy()
        .squeeze()
    )

    node_pred = (
        node_pred
        .cpu()
        .numpy()
        .squeeze()
    )

    #tumor_bin = (tumor_pred > THRESHOLD)
    #node_bin = (node_pred > THRESHOLD)

    # --------------------------------------------------------
    # BUILD LABEL MAP
    # --------------------------------------------------------

    #crop_mask = np.zeros(CUBE_SIZE, dtype=np.uint8)

    #crop_mask[tumor_bin] = 1
    #crop_mask[node_bin] = 2
    
    crop_mask = np.zeros(tumor_pred.shape, dtype=np.uint8)

    tumor_only = (
        tumor_pred > THRESHOLD
    ) & (
        tumor_pred >= node_pred
    )

    node_only = (
        node_pred > THRESHOLD
    ) & (
        node_pred > tumor_pred
    )

    crop_mask[
        tumor_only
    ] = 1

    crop_mask[
        node_only
    ] = 2

    patch_scores = patch_dice(gt_crop, crop_mask)
    print("Patch Dice:")
    print(patch_scores)

    # --------------------------------------------------------
    # INSERT INTO RESAMPLED VOLUME
    # --------------------------------------------------------

    full_mask_res = np.zeros(
        ct_res.shape,
        dtype=np.uint8
    )

    crop_shape = (
        end - start
    )

    crop_mask = crop_mask[
        :crop_shape[0],
        :crop_shape[1],
        :crop_shape[2]
    ]

    full_mask_res[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ] = crop_mask

    # --------------------------------------------------------
    # BACK TO ORIGINAL SPACING
    # --------------------------------------------------------

    reverse_zoom = (
        np.array(original_shape)
        / np.array(full_mask_res.shape)
    )

    # --------------------------------------------------------
    # RECONSTRUCTION TEST
    # --------------------------------------------------------

    full_gt_res = np.zeros(
        ct_res.shape,
        dtype=np.uint8
    )

    gt_crop_trim = gt_crop[
        :crop_shape[0],
        :crop_shape[1],
        :crop_shape[2]
    ]

    full_gt_res[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ] = gt_crop_trim

    mask_reconstructed = zoom(
        full_gt_res,
        reverse_zoom,
        order=0
    ).astype(np.uint8)

    recon_scores = reconstruction_dice(
        gt,
        mask_reconstructed
    )
    print("Reconstruction Dice:")
    print(recon_scores)


    final_mask = zoom(
        full_mask_res,
        reverse_zoom,
        order=0
    ).astype(np.uint8)

    challenge_scores = original_space_dice(
        gt,
        final_mask
    )
    print("Original-Space Dice:")
    print(challenge_scores)


    #final_mask = resample_mask_to_reference(
    #    full_mask_res,
    #    ct_path
    #)

    #return final_mask
    return (    
        final_mask,
        patch_scores,
        recon_scores,
        challenge_scores
    )


# ============================================================
# SAVE MHA
# ============================================================

def save_mha(
    mask,
    reference_ct_path,
    output_path
):

    ref = sitk.ReadImage(
        reference_ct_path
    )

    out = sitk.GetImageFromArray(
        mask.transpose(2,1,0)
    )

    out.SetSpacing(
        ref.GetSpacing()
    )

    out.SetOrigin(
        ref.GetOrigin()
    )

    out.SetDirection(
        ref.GetDirection()
    )

    sitk.WriteImage(
        out,
        output_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--model_path",
        required=True
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    args = parser.parse_args()

    model = load_model(
        args.model_path,
        args.device
    )

    patients = sorted(
        os.listdir(args.input_dir)
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )
    for patient_id in patients:

        patient_dir = os.path.join(
            args.input_dir,
            patient_id
        )

        ct_path = os.path.join(
            patient_dir,
            f"{patient_id}__CT.nii.gz"
        )

        pt_path = os.path.join(
            patient_dir,
            f"{patient_id}__PT.nii.gz"
        )

        gt_path = os.path.join(
            patient_dir,
            f"{patient_id}.nii.gz"
        )

        try:
            (pred_mask, patch_scores, recon_scores, challenge_scores ) = predict_patient(ct_path, pt_path, gt_path, model, args.device)
        except Exception as e:

            print(f"[ERROR] {patient_id}: {e}")

            pred_mask = np.zeros(
                nib.load(ct_path).shape,
                dtype=np.uint8
            )

        output_path = os.path.join(
            args.output_dir,
            f"{patient_id}.mha"
        )

        save_mha(pred_mask, ct_path, output_path)

        #get_original_space_dice(output_path, gt_path)

        verify_mha(ct_path, gt_path, output_path)

        print("\n==============================")
        print(patient_id)
        print("==============================")

        print(
            f"Reconstruction Dice:"
            f" GTVp={recon_scores['GTVp']:.4f}"
            f" GTVn={recon_scores['GTVn']:.4f}"
            f" Mean={recon_scores['Mean']:.4f}"
        )

        print(
            f"Patch Dice:"
            f" GTVp={patch_scores['GTVp']:.4f}"
            f" GTVn={patch_scores['GTVn']:.4f}"
            f" Mean={patch_scores['Mean']:.4f}"
        )

        print(
            f"Original Dice:"
            f" GTVp={challenge_scores['GTVp']:.4f}"
            f" GTVn={challenge_scores['GTVn']:.4f}"
            f" Mean={challenge_scores['Mean']:.4f}"
        )

        print(
            f"Finished {patient_id}"
        )













