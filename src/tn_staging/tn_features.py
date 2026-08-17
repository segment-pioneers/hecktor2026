import os
import pickle
import numpy as np
import pandas as pd

from scipy.ndimage import label


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


def masks_from_gt(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    return (mask == 1).astype(np.uint8), (mask == 2).astype(np.uint8)


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

def mask_centroid(mask):

    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return [0.0, 0.0, 0.0]

    centroid = coords.mean(axis=0)

    return centroid.astype(np.float32).tolist()


def bbox_ratios(dx, dy, dz):

    eps = 1e-6

    return [
        float(dx / (dy + eps)),
        float(dx / (dz + eps)),
        float(dy / (dz + eps))
    ]


def tlg_feature(volume, suv_mean):

    return float(volume * suv_mean)

def connected_component_centroids(node_mask):

    labeled, num_components = label(node_mask)

    centroids = []

    for i in range(1, num_components + 1):

        coords = np.argwhere(labeled == i)

        if len(coords) == 0:
            continue

        centroids.append(
            coords.mean(axis=0)
        )

    return centroids


def bilateral_involvement(node_mask):

    centroids = connected_component_centroids(
        node_mask
    )

    if len(centroids) == 0:
        return 0.0

    xs = np.array(
        [c[0] for c in centroids]
    )

    midline = node_mask.shape[0] / 2.0

    left = np.any(xs < midline)
    right = np.any(xs >= midline)

    return float(left and right)


def node_spread(node_mask):

    centroids = connected_component_centroids(
        node_mask
    )

    if len(centroids) < 2:
        return [0.0, 0.0, 0.0]

    centroids = np.asarray(
        centroids,
        dtype=np.float32
    )

    spread = (
        centroids.max(axis=0)
        -
        centroids.min(axis=0)
    )

    return spread.tolist()


def max_inter_node_distance(node_mask):

    centroids = connected_component_centroids(
        node_mask
    )

    if len(centroids) < 2:
        return 0.0

    centroids = np.asarray(
        centroids,
        dtype=np.float32
    )

    max_dist = 0.0

    for i in range(len(centroids)):

        for j in range(i + 1, len(centroids)):

            d = np.linalg.norm(
                centroids[i] - centroids[j]
            )

            max_dist = max(
                max_dist,
                d
            )

    return float(max_dist)


def tumor_node_distances(
    tumor_mask,
    node_mask
):

    tumor_coords = np.argwhere(
        tumor_mask > 0
    )

    if len(tumor_coords) == 0:
        return [0.0, 0.0]

    tumor_centroid = (
        tumor_coords.mean(axis=0)
    )

    node_centroids = (
        connected_component_centroids(
            node_mask
        )
    )

    if len(node_centroids) == 0:
        return [0.0, 0.0]

    distances = []

    for c in node_centroids:

        distances.append(
            np.linalg.norm(
                c - tumor_centroid
            )
        )

    return [
        float(np.min(distances)),
        float(np.max(distances))
    ]


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

'''
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
'''
def tumor_features(
    pet,
    tumor_mask
):

    volume = mask_volume(
        tumor_mask
    )

    suv_mean, suv_max, suv_std = (
        suv_statistics(
            pet,
            tumor_mask
        )
    )

    bbox_x, bbox_y, bbox_z = (
        bbox_features(
            tumor_mask
        )
    )

    centroid_x, centroid_y, centroid_z = (
        mask_centroid(
            tumor_mask
        )
    )

    tlg = tlg_feature(
        volume,
        suv_mean
    )

    ratio_xy, ratio_xz, ratio_yz = (
        bbox_ratios(
            bbox_x,
            bbox_y,
            bbox_z
        )
    )

    return np.array([
        volume,

        suv_mean,
        suv_max,
        suv_std,

        bbox_x,
        bbox_y,
        bbox_z,

        centroid_x,
        centroid_y,
        centroid_z,

        tlg,

        ratio_xy,
        ratio_xz,
        ratio_yz

    ], dtype=np.float32)

# ------------------------------------------------------------
# Node Features
# ------------------------------------------------------------
'''
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
'''

def node_features(
    pet,
    node_mask,
    tumor_mask
):

    total_volume = mask_volume(
        node_mask
    )

    suv_mean, suv_max, suv_std = (
        suv_statistics(
            pet,
            node_mask
        )
    )

    labeled, num_components = (
        label(node_mask)
    )

    largest_component = 0.0

    if num_components > 0:

        component_sizes = []

        for i in range(
            1,
            num_components + 1
        ):

            component_sizes.append(
                np.sum(
                    labeled == i
                )
            )

        largest_component = (
            max(component_sizes)
            * voxel_volume_mm3()
        )

    bilateral = (
        bilateral_involvement(
            node_mask
        )
    )

    spread_x, spread_y, spread_z = (
        node_spread(
            node_mask
        )
    )

    max_node_dist = (
        max_inter_node_distance(
            node_mask
        )
    )

    nearest_node_dist, farthest_node_dist = (
        tumor_node_distances(
            tumor_mask,
            node_mask
        )
    )

    return np.array([

        total_volume,
        largest_component,
        float(num_components),

        suv_mean,
        suv_max,
        suv_std,

        bilateral,

        spread_x,
        spread_y,
        spread_z,

        max_node_dist,

        nearest_node_dist,
        farthest_node_dist

    ], dtype=np.float32)

# ------------------------------------------------------------
# Single Case Feature Extraction
# ------------------------------------------------------------

def build_combined_mask(pred_tumor, pred_node):
    mask = np.zeros_like(pred_tumor, dtype=np.uint8)
    mask[pred_tumor > 0] = 1
    mask[pred_node > 0] = 2
    return mask


def extract_case_features(
    npz_path,
    clinical_row,
    mask=None,
):

    npz_data = np.load(npz_path)

    pet = np.asarray(
        npz_data["PET"],
        dtype=np.float32
    )

    if mask is None:
        mask = np.asarray(
            npz_data["MASK"],
            dtype=np.uint8
        )

    gt_tumor, gt_node = masks_from_gt(mask)

    tumor_feat = tumor_features(
        pet,
        gt_tumor
    )

    #node_feat = node_features(
    #    pet,
    #    gt_node
    #)
    node_feat = node_features(
        pet,
        gt_node,
        gt_tumor
    )

    ######################################
    #Add interaction features
    ######################################
    tumor_volume = tumor_feat[0]

    node_total_volume = node_feat[0]
    largest_node_volume = node_feat[1]

    eps = 1e-6

    interaction_feat = np.array([

        node_total_volume /
        (tumor_volume + eps),

        largest_node_volume /
        (tumor_volume + eps)

    ], dtype=np.float32)

    ######################################
    #clinical features
    ######################################
    clinical_feat = build_clinical_features(
        clinical_row
    )

    features = np.concatenate([
        tumor_feat,
        node_feat,
        interaction_feat,
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
    mask_source="ground_truth",
    seg_model=None,
    device="cpu",
):

    clinical_df = pd.read_csv(
        clinical_csv
    )

    clinical_df = clinical_df.set_index(
        "PatientID"
    )

    seg_model_obj = None

    if mask_source == "predicted":

        from segmentation.model_inference import (
            load_segmentation_model,
            predict_masks,
        )

        if seg_model is None:
            raise ValueError(
                "--seg_model is required when --mask_source predicted"
            )

        seg_model_obj = load_segmentation_model(
            seg_model,
            device,
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

        mask = None

        if mask_source == "predicted":

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
                seg_model_obj,
                pet,
                ct,
                device,
            )

            mask = build_combined_mask(
                pred_tumor,
                pred_node
            )

        feat = extract_case_features(
            npz_path,
            clinical_row,
            mask=mask,
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
    patient_ids,
    mask_source="ground_truth",
):

    with open(
        cache_path,
        "wb"
    ) as f:

        pickle.dump(
            {
                "X": X,
                "patient_ids": patient_ids,
                "mask_source": mask_source,
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