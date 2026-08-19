# Handcrafted Features and Missing-Value Handling

This document lists the exact feature vectors used for TN staging (37 features) and recurrence-free survival (RFS) prediction (41 features). Imaging features are computed on 1 mm isotropic PET/CT crops. Mask labels are `1` = GTVp and `2` = GTVn. Empty masks yield zeros for the corresponding imaging descriptors.

Implementation:

- TN: `tn_staging/tn_features.py` (no named array; names below follow `extract_case_features` order)
- RFS: `survival/surv_features.py` (`FEATURE_NAMES`)

---

## Missing-value handling

**TN staging (clinical).** Age and gender are used as numeric values. HPV status, tobacco consumption, and alcohol consumption each contribute a pair: the observed value (0 if missing) and a binary missing indicator (1 if missing, else 0).

**RFS (clinical).** HPV, tobacco, and alcohol use the same value + missing-indicator encoding. Performance status (0–4) uses the same pair (0 if missing). Treatment is a 3-way one-hot (`0`, `1`, `2`) plus a missing indicator; if treatment is missing, the one-hot vector is all zeros and the indicator is 1.

**Imaging.** There are no imputed radiomic values. If GTVp or GTVn is empty, volumes, PET intensity statistics, bounding boxes, centroids, and related descriptors are set to 0. Metabolic descriptors are mean / max / std of the **z-scored PET crop** inside the mask (not raw SUV).

---

## TN staging (37 features)

Order matches `extract_case_features` (tumor, then nodes, then interaction, then clinical). Feature names in this section are documentation labels only; `tn_features.py` does not define a `FEATURE_NAMES` list.

### Tumor / GTVp (14) — burden, metabolic, geometric, spatial

| # | Name | Category | Definition |
|---|---|---|---|
| 1 | `tumor_volume` | Anatomical | GTVp voxel count (mm³ at 1 mm spacing) |
| 2–4 | `tumor_suv_mean`, `tumor_suv_max`, `tumor_suv_std` | Metabolic | PET intensity (z-scored crop) mean / max / std inside GTVp |
| 5–7 | `tumor_bbox_x`, `tumor_bbox_y`, `tumor_bbox_z` | Geometric | Axis-aligned bounding-box extents (mm) |
| 8–10 | `tumor_centroid_x`, `tumor_centroid_y`, `tumor_centroid_z` | Spatial | Mean voxel coordinates of GTVp |
| 11 | `tumor_tlg` | Metabolic | Volume × PET intensity mean (z-scored crop) |
| 12–14 | `tumor_bbox_ratio_xy`, `tumor_bbox_ratio_xz`, `tumor_bbox_ratio_yz` | Geometric | Bounding-box aspect ratios |

### Nodes / GTVn (13) — burden, metabolic, distribution, tumor–node geometry

| # | Name | Category | Definition |
|---|---|---|---|
| 15 | `node_volume` | Anatomical | Total GTVn voxel count (mm³) |
| 16 | `largest_node_volume` | Anatomical | Largest connected-component volume (mm³) |
| 17 | `node_count` | Anatomical | Number of connected GTVn components |
| 18–20 | `node_suv_mean`, `node_suv_max`, `node_suv_std` | Metabolic | PET intensity (z-scored crop) mean / max / std inside GTVn |
| 21 | `bilateral_involvement` | Anatomical | 1 if node components exist on both sides of the crop midline, else 0 |
| 22–24 | `node_spread_x`, `node_spread_y`, `node_spread_z` | Spatial | Range of node-component centroids along each axis (0 if fewer than two components) |
| 25 | `max_inter_node_distance` | Spatial | Maximum Euclidean distance between node-component centroids |
| 26–27 | `nearest_node_dist`, `farthest_node_dist` | Interaction | Min / max distance from the GTVp centroid to node-component centroids |

### Tumor–node interaction (2)

| # | Name | Category | Definition |
|---|---|---|---|
| 28 | `node_to_tumor_volume_ratio` | Interaction | Total node volume / (tumor volume + ε) |
| 29 | `largest_node_to_tumor_volume_ratio` | Interaction | Largest node volume / (tumor volume + ε) |

### Clinical (8)

| # | Name | Category | Definition |
|---|---|---|---|
| 30 | `age` | Clinical | Age |
| 31 | `gender` | Clinical | Gender |
| 32–33 | `hpv`, `hpv_missing` | Clinical | HPV status and missing indicator |
| 34–35 | `tobacco`, `tobacco_missing` | Clinical | Tobacco consumption and missing indicator |
| 36–37 | `alcohol`, `alcohol_missing` | Clinical | Alcohol consumption and missing indicator |

---

## RFS prediction (41 features)

Order matches `FEATURE_NAMES` in `survival/surv_features.py`.

### Clinical (14)

| Name | Definition |
|---|---|
| `age`, `gender` | Age; gender |
| `hpv`, `hpv_missing` | HPV value and missing indicator |
| `tobacco`, `tobacco_missing` | Tobacco value and missing indicator |
| `alcohol`, `alcohol_missing` | Alcohol value and missing indicator |
| `performance`, `performance_missing` | Performance status (0–4) and missing indicator |
| `treatment_0`, `treatment_1`, `treatment_2`, `treatment_missing` | Treatment one-hot and missing indicator |

### Segmentation-derived (18) — burden, metabolic, geometric, morphology

| Name | Category | Definition |
|---|---|---|
| `tumor_volume` | Anatomical | GTVp voxel count (mm³) |
| `tumor_suv_mean`, `tumor_suv_max`, `tumor_suv_std` | Metabolic | PET intensity (z-scored crop) mean / max / std in GTVp |
| `bbox_x`, `bbox_y`, `bbox_z` | Geometric | GTVp bounding-box extents (mm) |
| `maximum_diameter` | Geometric | Max of the three GTVp bbox extents |
| `node_volume` | Anatomical | Total GTVn voxel count |
| `largest_node_volume` | Anatomical | Largest GTVn connected-component volume |
| `node_count` | Anatomical | Number of GTVn connected components |
| `node_suv_mean`, `node_suv_max`, `node_suv_std` | Metabolic | PET intensity (z-scored crop) mean / max / std in GTVn |
| `tumor_tlg`, `node_tlg` | Metabolic | Volume × PET intensity mean (z-scored crop) for GTVp and GTVn |
| `tumor_sphericity` | Morphology | \((36\pi V^2)^{1/3} / A\) using an eroded-boundary surface estimate \(A\) |
| `tumor_compactness` | Morphology | \(V / \sqrt{A}\) |

### TN-stage probabilities (9)

| Name | Definition |
|---|---|
| `T0_prob` … `T4_prob` | CatBoost T-stage class probabilities (5) |
| `N0_prob` … `N3_prob` | CatBoost N-stage class probabilities (4) |

These nine values are the full probability vectors, not the argmax stage labels.