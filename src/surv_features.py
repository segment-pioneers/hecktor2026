"""
surv_features.py

Feature engineering for HECKTOR 2026 survival prediction.

Inputs
------
Clinical CSV
PET volume
Predicted segmentation mask (Stage-I): 0 background, 1 GTVp, 2 GTVn

Outputs
-------
Clinical features
+
Segmentation-derived features
+
TN-stage probabilities

Designed for:
    Random Survival Forest
    Gradient Boosting Survival
    CoxPH
"""

import numpy as np
import pandas as pd
from scipy import ndimage


# =============================================================================
# Clinical Features
# =============================================================================

def _binary_with_missing(value):
    """
    Convert binary variable:

        0 -> [0,0]
        1 -> [1,0]
      NaN -> [0,1]

    Returns:
        value_feature
        missing_indicator
    """
    if pd.isna(value):
        return [0.0, 1.0]

    return [float(value), 0.0]


def _performance_with_missing(value):
    """
    Performance Status:
        0..4

    Returns:
        value
        missing_indicator
    """
    if pd.isna(value):
        return [0.0, 1.0]

    return [float(value), 0.0]


def _treatment_onehot(value):
    """
    Treatment:
        0
        1
        2

    Returns:
        one-hot(3)
        missing_indicator
    """
    vec = [0.0, 0.0, 0.0]

    if pd.isna(value):
        return vec + [1.0]

    idx = int(value)

    if 0 <= idx <= 2:
        vec[idx] = 1.0

    return vec + [0.0]


def extract_clinical_features(row):
    """
    Parameters
    ----------
    row : pandas Series

    Returns
    -------
    list[float]
    """

    features = []

    # ---------------------------------------------------------
    # Age
    # ---------------------------------------------------------
    features.append(float(row["Age"]))

    # ---------------------------------------------------------
    # Gender
    # ---------------------------------------------------------
    features.append(float(row["Gender"]))

    # ---------------------------------------------------------
    # HPV
    # ---------------------------------------------------------
    features.extend(
        _binary_with_missing(row["HPV Status"])
    )

    # ---------------------------------------------------------
    # Tobacco
    # ---------------------------------------------------------
    features.extend(
        _binary_with_missing(row["Tobacco Consumption"])
    )

    # ---------------------------------------------------------
    # Alcohol
    # ---------------------------------------------------------
    features.extend(
        _binary_with_missing(row["Alcohol Consumption"])
    )

    # ---------------------------------------------------------
    # Performance Status
    # ---------------------------------------------------------
    features.extend(
        _performance_with_missing(row["Performance Status"])
    )

    # ---------------------------------------------------------
    # Treatment
    # ---------------------------------------------------------
    features.extend(
        _treatment_onehot(row["Treatment"])
    )

    return np.asarray(features, dtype=np.float32)


# =============================================================================
# Segmentation Features
# =============================================================================

def _safe_stats(values):
    """
    Robust SUV statistics.
    """

    if values.size == 0:
        return {
            "mean": 0.0,
            "max": 0.0,
            "std": 0.0
        }

    return {
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std())
    }


def _largest_component_volume(mask):
    """
    Largest connected component volume.
    """

    if mask.sum() == 0:
        return 0.0

    labeled, n_comp = ndimage.label(mask)

    if n_comp == 0:
        return 0.0

    sizes = ndimage.sum(
        mask,
        labeled,
        range(1, n_comp + 1)
    )

    return float(np.max(sizes))


def _component_count(mask):
    """
    Number of connected lesions.
    """

    _, n_comp = ndimage.label(mask)

    return float(n_comp)


def _bbox_features(mask):
    """Same logic as tn_features.bbox_features (dx, dy, dz in mm at 1 mm voxels)."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 0.0, 0.0, 0.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims = maxs - mins + 1
    return float(dims[0]), float(dims[1]), float(dims[2])


def _surface_area_estimate(mask):
    """
    Approximate surface area (mm^2) from boundary voxels at 1 mm isotropic spacing.
    """

    if mask.sum() == 0:
        return 0.0

    eroded = ndimage.binary_erosion(mask)
    boundary = mask & ~eroded

    return float(boundary.sum())


def _maximum_diameter(bbox_x, bbox_y, bbox_z):
    return float(max(bbox_x, bbox_y, bbox_z))


def _tumor_sphericity(volume, surface_area):
    if surface_area <= 0.0 or volume <= 0.0:
        return 0.0

    return float((36.0 * np.pi * volume ** 2) ** (1.0 / 3.0) / surface_area)


def _tumor_compactness(volume, surface_area):
    if surface_area <= 0.0 or volume <= 0.0:
        return 0.0

    return float(volume / np.sqrt(surface_area))


def extract_segmentation_features(
    pet,
    mask
):
    """
    Parameters
    ----------
    pet : ndarray
        SUV-normalized PET

    mask : ndarray
        Predicted combined mask: 0 background, 1 GTVp, 2 GTVn

    Returns
    -------
    np.ndarray
        18 segmentation features
    """

    tumor_mask = mask == 1
    node_mask = mask == 2

    # ---------------------------------------------------------
    # Volumes
    # ---------------------------------------------------------

    tumor_volume = float(tumor_mask.sum())

    node_volume = float(node_mask.sum())

    largest_node_volume = _largest_component_volume(
        node_mask
    )

    node_count = _component_count(
        node_mask
    )

    # ---------------------------------------------------------
    # PET statistics
    # ---------------------------------------------------------

    tumor_stats = _safe_stats(
        pet[tumor_mask]
    )

    node_stats = _safe_stats(
        pet[node_mask]
    )

    # ---------------------------------------------------------
    # TLG
    # ---------------------------------------------------------

    tumor_tlg = (
        tumor_volume *
        tumor_stats["mean"]
    )

    node_tlg = (
        node_volume *
        node_stats["mean"]
    )

    bbox_x, bbox_y, bbox_z = _bbox_features(tumor_mask)
    maximum_diameter = _maximum_diameter(bbox_x, bbox_y, bbox_z)
    surface_area = _surface_area_estimate(tumor_mask)
    sphericity = _tumor_sphericity(tumor_volume, surface_area)
    compactness = _tumor_compactness(tumor_volume, surface_area)

    features = [

        # Tumor burden
        tumor_volume,

        tumor_stats["mean"],
        tumor_stats["max"],
        tumor_stats["std"],

        bbox_x,
        bbox_y,
        bbox_z,
        maximum_diameter,

        # Node burden
        node_volume,
        largest_node_volume,
        node_count,

        node_stats["mean"],
        node_stats["max"],
        node_stats["std"],

        # Glycolysis
        tumor_tlg,
        node_tlg,

        # Shape
        sphericity,
        compactness,
    ]

    return np.asarray(
        features,
        dtype=np.float32
    )


# =============================================================================
# TN Probability Features
# =============================================================================

def extract_tn_probability_features(
    t_probs,
    n_probs
):
    """
    Parameters
    ----------
    t_probs
        length 5

    n_probs
        length 4

    Returns
    -------
    np.ndarray
    """

    return np.concatenate(
        [
            np.asarray(t_probs, dtype=np.float32),
            np.asarray(n_probs, dtype=np.float32),
        ]
    )


# =============================================================================
# Complete Survival Feature Vector
# =============================================================================

def build_survival_feature_vector(
    clinical_row,
    pet,
    mask,
    t_probs,
    n_probs,
):
    """
    Creates final feature vector.

    Returns
    -------
    np.ndarray
    """

    clinical = extract_clinical_features(
        clinical_row
    )

    seg = extract_segmentation_features(
        pet,
        mask
    )

    tn = extract_tn_probability_features(
        t_probs,
        n_probs
    )

    return np.concatenate(
        [
            clinical,
            seg,
            tn,
        ]
    )


# =============================================================================
# Feature Names
# =============================================================================

FEATURE_NAMES = [

    # Clinical
    "age",
    "gender",

    "hpv",
    "hpv_missing",

    "tobacco",
    "tobacco_missing",

    "alcohol",
    "alcohol_missing",

    "performance",
    "performance_missing",

    "treatment_0",
    "treatment_1",
    "treatment_2",
    "treatment_missing",

    # Segmentation
    "tumor_volume",

    "tumor_suv_mean",
    "tumor_suv_max",
    "tumor_suv_std",

    "bbox_x",
    "bbox_y",
    "bbox_z",
    "maximum_diameter",

    "node_volume",
    "largest_node_volume",
    "node_count",

    "node_suv_mean",
    "node_suv_max",
    "node_suv_std",

    "tumor_tlg",
    "node_tlg",

    "tumor_sphericity",
    "tumor_compactness",

    # T probabilities
    "T0_prob",
    "T1_prob",
    "T2_prob",
    "T3_prob",
    "T4_prob",

    # N probabilities
    "N0_prob",
    "N1_prob",
    "N2_prob",
    "N3_prob",
]