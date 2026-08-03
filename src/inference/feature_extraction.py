"""
feature_extraction.py

Feature extraction module for the HECKTOR 2026 inference
pipeline.

This module computes the handcrafted features required by

    1. TN staging model
    2. Recurrence-free survival model

Inputs
------
PET volume
Predicted segmentation mask
Clinical row
(Optional) TN probabilities

Outputs
-------
Feature vectors identical to those used during model
training.
"""

import numpy as np
import pandas as pd

from scipy import ndimage


# ============================================================
# CONSTANTS
# ============================================================

VOXEL_VOLUME = 1.0      # mm³

EPS = 1e-6


# ============================================================
# MASK UTILITIES
# ============================================================

def tumor_mask(mask):
    """
    Extract GTVp.
    """

    return (mask == 1)


def node_mask(mask):
    """
    Extract GTVn.
    """

    return (mask == 2)


def mask_volume(mask):
    """
    Volume in mm³.
    """

    return float(mask.sum()) * VOXEL_VOLUME


# ============================================================
# SUV STATISTICS
# ============================================================

def suv_statistics(
    pet,
    mask
):
    """
    Returns

        mean
        max
        std
    """

    values = pet[mask > 0]

    if values.size == 0:

        return (
            0.0,
            0.0,
            0.0
        )

    return (

        float(values.mean()),

        float(values.max()),

        float(values.std())
    )


# ============================================================
# BOUNDING BOX
# ============================================================

def bbox_features(mask):
    """
    Returns

        dx
        dy
        dz
    """

    coords = np.argwhere(mask)

    if len(coords) == 0:

        return (

            0.0,
            0.0,
            0.0
        )

    mins = coords.min(axis=0)

    maxs = coords.max(axis=0)

    dims = maxs - mins + 1

    return (

        float(dims[0]),

        float(dims[1]),

        float(dims[2])
    )


def centroid(mask):
    """
    Mask centroid.
    """

    coords = np.argwhere(mask)

    if len(coords) == 0:

        return (

            0.0,
            0.0,
            0.0
        )

    c = coords.mean(axis=0)

    return (

        float(c[0]),

        float(c[1]),

        float(c[2])
    )


def bbox_ratios(
    dx,
    dy,
    dz
):

    return (

        dx / (dy + EPS),

        dx / (dz + EPS),

        dy / (dz + EPS)
    )


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def connected_components(mask):
    """
    Connected component labeling.
    """

    labeled, n = ndimage.label(mask)

    return labeled, int(n)


def largest_component_volume(mask):
    """
    Largest connected lesion.
    """

    labeled, n = connected_components(mask)

    if n == 0:

        return 0.0

    sizes = ndimage.sum(
        mask,
        labeled,
        range(1, n + 1)
    )

    return float(np.max(sizes))

# ============================================================
# CONNECTED COMPONENT CENTROIDS
# ============================================================

def connected_component_centroids(mask):
    """
    Returns centroid of each connected component.
    """

    labeled, n = connected_components(mask)

    centroids = []

    for i in range(1, n + 1):

        coords = np.argwhere(
            labeled == i
        )

        if len(coords) == 0:
            continue

        centroids.append(
            coords.mean(axis=0)
        )

    return centroids


# ============================================================
# NODE FEATURES
# ============================================================

def bilateral_involvement(node_mask):
    """
    Returns 1 if lymph nodes are present on both
    sides of the neck.
    """

    centroids = connected_component_centroids(
        node_mask
    )

    if len(centroids) == 0:

        return 0.0

    xs = np.asarray(
        [c[0] for c in centroids]
    )

    midline = (
        node_mask.shape[0] / 2.0
    )

    left = np.any(xs < midline)

    right = np.any(xs >= midline)

    return float(left and right)


def node_spread(node_mask):
    """
    Spread of lymph node centroids.
    """

    centroids = connected_component_centroids(
        node_mask
    )

    if len(centroids) < 2:

        return (
            0.0,
            0.0,
            0.0
        )

    centroids = np.asarray(
        centroids,
        dtype=np.float32
    )

    spread = (

        centroids.max(axis=0)

        -

        centroids.min(axis=0)

    )

    return (

        float(spread[0]),

        float(spread[1]),

        float(spread[2])
    )


def max_inter_node_distance(node_mask):
    """
    Maximum centroid-to-centroid distance.
    """

    centroids = connected_component_centroids(
        node_mask
    )

    if len(centroids) < 2:

        return 0.0

    centroids = np.asarray(
        centroids,
        dtype=np.float32
    )

    max_distance = 0.0

    for i in range(len(centroids)):

        for j in range(i + 1,
                       len(centroids)):

            distance = np.linalg.norm(

                centroids[i]

                -

                centroids[j]

            )

            max_distance = max(
                max_distance,
                distance
            )

    return float(max_distance)


def tumor_node_distances(
    tumor_mask,
    node_mask
):
    """
    Returns

        nearest node

        farthest node
    """

    tumor_coords = np.argwhere(
        tumor_mask
    )

    if len(tumor_coords) == 0:

        return (

            0.0,
            0.0
        )

    tumor_center = (
        tumor_coords.mean(axis=0)
    )

    node_centers = (
        connected_component_centroids(
            node_mask
        )
    )

    if len(node_centers) == 0:

        return (

            0.0,
            0.0
        )

    distances = []

    for c in node_centers:

        distances.append(

            np.linalg.norm(

                c

                -

                tumor_center

            )

        )

    return (

        float(np.min(distances)),

        float(np.max(distances))
    )


# ============================================================
# SHAPE FEATURES
# ============================================================

def surface_area_estimate(mask):
    """
    Approximate surface area.

    Same implementation as used for
    survival feature extraction.
    """

    if mask.sum() == 0:

        return 0.0

    eroded = ndimage.binary_erosion(
        mask
    )

    boundary = (
        mask &
        ~eroded
    )

    return float(
        boundary.sum()
    )


def maximum_diameter(
    bbox_x,
    bbox_y,
    bbox_z
):
    """
    Maximum bounding-box dimension.
    """

    return float(

        max(

            bbox_x,
            bbox_y,
            bbox_z

        )

    )


def sphericity(
    volume,
    surface_area
):
    """
    Tumor sphericity.
    """

    if volume <= 0.0:

        return 0.0

    if surface_area <= 0.0:

        return 0.0

    return float(

        (

            36.0

            * np.pi

            * volume ** 2

        ) ** (1.0 / 3.0)

        /

        surface_area

    )


def compactness(
    volume,
    surface_area
):
    """
    Tumor compactness.
    """

    if volume <= 0.0:

        return 0.0

    if surface_area <= 0.0:

        return 0.0

    return float(

        volume

        /

        np.sqrt(surface_area)

    )


# ============================================================
# TLG
# ============================================================

def tlg(
    volume,
    suv_mean
):
    """
    Total Lesion Glycolysis.
    """

    return float(
        volume * suv_mean
    )

# ============================================================
# TN CLINICAL FEATURES
# ============================================================

def _clinical_value(clinical_row, key, required=False):
    if key in clinical_row.index:
        return clinical_row[key]
    if required:
        raise KeyError(
            f"Required clinical field {key!r} missing from ehr.json"
        )
    return np.nan


def extract_tn_clinical_features(clinical_row):
    """
    Clinical features used by the TN staging model.

    Returns
    -------
    np.ndarray
    """

    age = float(_clinical_value(clinical_row, "Age", required=True))

    gender = float(_clinical_value(clinical_row, "Gender", required=True))

    hpv = _clinical_value(clinical_row, "HPV Status")
    tobacco = _clinical_value(clinical_row, "Tobacco Consumption")
    alcohol = _clinical_value(clinical_row, "Alcohol Consumption")

    hpv_missing = 1.0 if pd.isna(hpv) else 0.0
    tobacco_missing = 1.0 if pd.isna(tobacco) else 0.0
    alcohol_missing = 1.0 if pd.isna(alcohol) else 0.0

    hpv = 0.0 if pd.isna(hpv) else float(hpv)
    tobacco = 0.0 if pd.isna(tobacco) else float(tobacco)
    alcohol = 0.0 if pd.isna(alcohol) else float(alcohol)

    return np.asarray([

        age,

        gender,

        hpv,
        hpv_missing,

        tobacco,
        tobacco_missing,

        alcohol,
        alcohol_missing

    ], dtype=np.float32)


# ============================================================
# TN FEATURE EXTRACTION
# ============================================================

def extract_tn_features(
    pet,
    mask,
    clinical_row
):
    """
    Feature vector for TN staging.

    Parameters
    ----------
    pet : ndarray

    mask : ndarray
        Combined segmentation

            0 background
            1 tumor
            2 node

    clinical_row : pandas.Series

    Returns
    -------
    np.ndarray
    """

    gtvp = tumor_mask(mask)

    gtvn = node_mask(mask)

    # --------------------------------------------------------
    # Tumor
    # --------------------------------------------------------

    tumor_volume = mask_volume(gtvp)

    (
        tumor_suv_mean,
        tumor_suv_max,
        tumor_suv_std

    ) = suv_statistics(
        pet,
        gtvp
    )

    (
        bbox_x,
        bbox_y,
        bbox_z

    ) = bbox_features(
        gtvp
    )

    (
        centroid_x,
        centroid_y,
        centroid_z

    ) = centroid(
        gtvp
    )

    tumor_tlg = tlg(
        tumor_volume,
        tumor_suv_mean
    )

    (
        ratio_xy,
        ratio_xz,
        ratio_yz

    ) = bbox_ratios(

        bbox_x,
        bbox_y,
        bbox_z
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    node_volume = mask_volume(
        gtvn
    )

    (
        node_suv_mean,
        node_suv_max,
        node_suv_std

    ) = suv_statistics(
        pet,
        gtvn
    )

    labeled, node_count = connected_components(
        gtvn
    )

    largest_node = largest_component_volume(
        gtvn
    )

    bilateral = bilateral_involvement(
        gtvn
    )

    (
        spread_x,
        spread_y,
        spread_z

    ) = node_spread(
        gtvn
    )

    max_node_distance = (
        max_inter_node_distance(
            gtvn
        )
    )

    (
        nearest_node,
        farthest_node

    ) = tumor_node_distances(

        gtvp,

        gtvn

    )

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    interaction_total = (

        node_volume

        /

        (tumor_volume + EPS)

    )

    interaction_largest = (

        largest_node

        /

        (tumor_volume + EPS)

    )

    # --------------------------------------------------------
    # Clinical
    # --------------------------------------------------------

    clinical = extract_tn_clinical_features(
        clinical_row
    )

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = np.concatenate([

        np.asarray([

            tumor_volume,

            tumor_suv_mean,
            tumor_suv_max,
            tumor_suv_std,

            bbox_x,
            bbox_y,
            bbox_z,

            centroid_x,
            centroid_y,
            centroid_z,

            tumor_tlg,

            ratio_xy,
            ratio_xz,
            ratio_yz,

        ], dtype=np.float32),

        np.asarray([

            node_volume,

            largest_node,

            float(node_count),

            node_suv_mean,
            node_suv_max,
            node_suv_std,

            bilateral,

            spread_x,
            spread_y,
            spread_z,

            max_node_distance,

            nearest_node,

            farthest_node,

        ], dtype=np.float32),

        np.asarray([

            interaction_total,

            interaction_largest

        ], dtype=np.float32),

        clinical

    ])

    return features

# ============================================================
# SURVIVAL CLINICAL FEATURES
# ============================================================

def _binary_with_missing(value):

    if pd.isna(value):

        return [0.0, 1.0]

    return [float(value), 0.0]


def _performance_with_missing(value):

    if pd.isna(value):

        return [0.0, 1.0]

    return [float(value), 0.0]


def _treatment_onehot(value):

    onehot = [0.0, 0.0, 0.0]

    if pd.isna(value):

        return onehot + [1.0]

    value = int(value)

    if 0 <= value <= 2:

        onehot[value] = 1.0

    return onehot + [0.0]


def extract_survival_clinical_features(
    clinical_row
):
    """
    Clinical features for survival prediction.
    """

    features = []

    features.append(
        float(_clinical_value(clinical_row, "Age", required=True))
    )

    features.append(
        float(_clinical_value(clinical_row, "Gender", required=True))
    )

    features.extend(
        _binary_with_missing(
            _clinical_value(clinical_row, "HPV Status")
        )
    )

    features.extend(
        _binary_with_missing(
            _clinical_value(clinical_row, "Tobacco Consumption")
        )
    )

    features.extend(
        _binary_with_missing(
            _clinical_value(clinical_row, "Alcohol Consumption")
        )
    )

    features.extend(
        _performance_with_missing(
            _clinical_value(clinical_row, "Performance Status")
        )
    )

    features.extend(
        _treatment_onehot(
            _clinical_value(clinical_row, "Treatment")
        )
    )

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# SURVIVAL FEATURE EXTRACTION
# ============================================================

def extract_survival_features(
    pet,
    mask,
    clinical_row,
    t_probs,
    n_probs
):
    """
    Feature vector for recurrence-free survival prediction.

    Parameters
    ----------
    pet : ndarray

    mask : ndarray
        Combined prediction

            0 background
            1 tumor
            2 node

    clinical_row : pandas.Series

    t_probs : ndarray
        Length 5

    n_probs : ndarray
        Length 4

    Returns
    -------
    np.ndarray
    """

    gtvp = tumor_mask(mask)

    gtvn = node_mask(mask)

    # --------------------------------------------------------
    # Volumes
    # --------------------------------------------------------

    tumor_volume = mask_volume(
        gtvp
    )

    node_volume = mask_volume(
        gtvn
    )

    largest_node = largest_component_volume(
        gtvn
    )

    _, node_count = connected_components(
        gtvn
    )

    # --------------------------------------------------------
    # PET statistics
    # --------------------------------------------------------

    (
        tumor_suv_mean,
        tumor_suv_max,
        tumor_suv_std

    ) = suv_statistics(
        pet,
        gtvp
    )

    (
        node_suv_mean,
        node_suv_max,
        node_suv_std

    ) = suv_statistics(
        pet,
        gtvn
    )

    # --------------------------------------------------------
    # Tumor geometry
    # --------------------------------------------------------

    (
        bbox_x,
        bbox_y,
        bbox_z

    ) = bbox_features(
        gtvp
    )

    max_diameter = maximum_diameter(

        bbox_x,

        bbox_y,

        bbox_z

    )

    surface = surface_area_estimate(
        gtvp
    )

    tumor_sphericity = sphericity(

        tumor_volume,

        surface

    )

    tumor_compactness = compactness(

        tumor_volume,

        surface

    )

    # --------------------------------------------------------
    # TLG
    # --------------------------------------------------------

    tumor_tlg = tlg(

        tumor_volume,

        tumor_suv_mean

    )

    node_tlg = tlg(

        node_volume,

        node_suv_mean

    )

    # --------------------------------------------------------
    # Segmentation features
    # --------------------------------------------------------

    segmentation_features = np.asarray([

        tumor_volume,

        tumor_suv_mean,
        tumor_suv_max,
        tumor_suv_std,

        bbox_x,
        bbox_y,
        bbox_z,

        max_diameter,

        node_volume,
        largest_node,
        float(node_count),

        node_suv_mean,
        node_suv_max,
        node_suv_std,

        tumor_tlg,
        node_tlg,

        tumor_sphericity,
        tumor_compactness

    ], dtype=np.float32)

    # --------------------------------------------------------
    # Clinical
    # --------------------------------------------------------

    clinical_features = (
        extract_survival_clinical_features(
            clinical_row
        )
    )

    # --------------------------------------------------------
    # TN probabilities
    # --------------------------------------------------------

    tn_features = np.concatenate([

        np.asarray(
            t_probs,
            dtype=np.float32
        ),

        np.asarray(
            n_probs,
            dtype=np.float32
        )

    ])

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    return np.concatenate([

        clinical_features,

        segmentation_features,

        tn_features

    ])