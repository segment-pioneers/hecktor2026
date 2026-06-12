import os
import pickle
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import label

import networks
import datagenerators


# ------------------------------------------------------------
# Clinical Features
# ------------------------------------------------------------

CLINICAL_COLUMNS = [
    "Age",
    "Gender",
    "HPV Status",
    "Tobacco Consumption",
    "Alcohol Consumption"
]


def build_clinical_features(df_row):
    """
    Returns:
        np.ndarray
    """

    age = float(df_row["Age"])
    gender = float(df_row["Gender"])

    hpv = df_row["HPV Status"]
    tobacco = df_row["Tobacco Consumption"]
    alcohol = df_row["Alcohol Consumption"]

    hpv_missing = 1.0 if pd.isna(hpv) else 0.0
    tobacco_missing = 1.0 if pd.isna(tobacco) else 0.0
    alcohol_missing = 1.0 if pd.isna(alcohol) else 0.0

    hpv = 0.0 if pd.isna(hpv) else float(hpv)
    tobacco = 0.0 if pd.isna(tobacco) else float(tobacco)
    alcohol = 0.0 if pd.isna(alcohol) else float(alcohol)

    return np.array([
        age,
        gender,

        hpv,
        hpv_missing,

        tobacco,
        tobacco_missing,

        alcohol,
        alcohol_missing
    ], dtype=np.float32)


# ------------------------------------------------------------
# Mask Utilities
# ------------------------------------------------------------

def voxel_volume_mm3():
    """
    Preprocessing resampled everything to 1x1x1 mm.

    Therefore:
        volume = voxel count
    """
    return 1.0


def mask_volume(mask):
    return float(mask.sum()) * voxel_volume_mm3()


def bbox_features(mask):
    """
    Returns:
        dx, dy, dz
    """

    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return [0.0, 0.0, 0.0]

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    dims = maxs - mins + 1

    return dims.astype(np.float32).tolist()


def suv_statistics(pet, mask):

    values = pet[mask > 0]

    if len(values) == 0:
        return [0.0, 0.0, 0.0]

    return [
        float(values.mean()),
        float(values.max()),
        float(values.std())
    ]


# ------------------------------------------------------------
# Tumor Features
# ------------------------------------------------------------

def tumor_features(pet, tumor_mask):

    volume = mask_volume(tumor_mask)

    suv_mean, suv_max, suv_std = suv_statistics(
        pet,
        tumor_mask
    )

    bbox_x, bbox_y, bbox_z = bbox_features(
        tumor_mask
    )

    return np.array([
        volume,
        suv_mean,
        suv_max,
        suv_std,
        bbox_x,
        bbox_y,
        bbox_z
    ], dtype=np.float32)


# ------------------------------------------------------------
# Node Features
# ------------------------------------------------------------

def node_features(pet, node_mask):

    total_volume = mask_volume(node_mask)

    suv_mean, suv_max, suv_std = suv_statistics(
        pet,
        node_mask
    )

    labeled, num_components = label(node_mask)

    largest_component = 0.0

    if num_components > 0:

        component_sizes = []

        for i in range(1, num_components + 1):

            size = np.sum(labeled == i)
            component_sizes.append(size)

        #largest_component = max(component_sizes)
        largest_component = (
            max(component_sizes)
            * voxel_volume_mm3()
        )

    return np.array([
        total_volume,
        largest_component,
        float(num_components),

        suv_mean,
        suv_max,
        suv_std
    ], dtype=np.float32)


# ------------------------------------------------------------
# Segmentation Model
# ------------------------------------------------------------

def load_segmentation_model(
    model_path,
    device
):

    model = networks.AdaMSS_Seg()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def predict_masks(
    model,
    pet,
    ct,
    device,
    threshold=0.5
):

    pet_t = torch.from_numpy(
        pet[np.newaxis, np.newaxis]
    ).float().to(device)

    ct_t = torch.from_numpy(
        ct[np.newaxis, np.newaxis]
    ).float().to(device)

    with torch.no_grad():

        pred_tumor, pred_node = model(
            pet_t,
            ct_t
        )

    pred_tumor = (
        pred_tumor
        .cpu()
        .numpy()
        .squeeze()
    )

    pred_node = (
        pred_node
        .cpu()
        .numpy()
        .squeeze()
    )

    pred_tumor = (pred_tumor > threshold).astype(
        np.uint8
    )

    pred_node = (pred_node > threshold).astype(
        np.uint8
    )

    return pred_tumor, pred_node


# ------------------------------------------------------------
# Single Case Feature Extraction
# ------------------------------------------------------------

def extract_case_features(
    npz_path,
    clinical_row,
    seg_model,
    device
):

    npz_data = np.load(npz_path)

    pet = np.asarray(
        npz_data["PET"],
        dtype=np.float32
    )

    ct = np.asarray(
        npz_data["CT"],
        dtype=np.float32
    )

    pred_tumor, pred_node = predict_masks(
        seg_model,
        pet,
        ct,
        device
    )

    tumor_feat = tumor_features(
        pet,
        pred_tumor
    )

    node_feat = node_features(
        pet,
        pred_node
    )

    clinical_feat = build_clinical_features(
        clinical_row
    )

    features = np.concatenate([
        tumor_feat,
        node_feat,
        clinical_feat
    ])

    return features


# ------------------------------------------------------------
# Dataset Feature Extraction
# ------------------------------------------------------------

def build_feature_table(
    data_dir,
    sample_names,
    clinical_csv,
    seg_model_path,
    device
):

    clinical_df = pd.read_csv(
        clinical_csv
    )

    clinical_df = clinical_df.set_index(
        "PatientID"
    )

    model = load_segmentation_model(
        seg_model_path,
        device
    )

    X = []

    patient_ids = []

    for sample in sample_names:

        patient_id = bytes.decode(
            sample
        ).replace(".npz", "")

        npz_path = os.path.join(
            data_dir,
            bytes.decode(sample)
        )

        clinical_row = clinical_df.loc[
            patient_id
        ]

        feat = extract_case_features(
            npz_path,
            clinical_row,
            model,
            device
        )

        X.append(feat)
        patient_ids.append(patient_id)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    return X, patient_ids


# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------

def save_feature_cache(
    cache_path,
    X,
    patient_ids
):

    with open(
        cache_path,
        "wb"
    ) as f:

        pickle.dump(
            {
                "X": X,
                "patient_ids": patient_ids
            },
            f
        )


def load_feature_cache(
    cache_path
):

    with open(
        cache_path,
        "rb"
    ) as f:

        data = pickle.load(f)

    return (
        data["X"],
        data["patient_ids"]
    )