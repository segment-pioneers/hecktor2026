"""
extract_surv_features.py

Build survival features for one fold.

Inputs
------
Clinical CSV
NPZ files (PET, CT, MASK)
Stage-I segmentation model (when --mask_source predicted)
TN probability predictions

Outputs
-------
survival_features.pkl
"""

import os
import pickle
import argparse

import numpy as np
import pandas as pd

from segmentation.model_inference import (
    load_segmentation_model,
    predict_masks,
)
from .surv_features import (
    build_survival_feature_vector,
    FEATURE_NAMES,
)


# ============================================================
# Utilities
# ============================================================

def load_pet_ct(npz_path):

    data = np.load(npz_path)

    return data["PET"], data["CT"]


def load_gt_mask(npz_path):

    data = np.load(npz_path)

    return np.asarray(data["MASK"], dtype=np.uint8)


def build_combined_mask(pred_tumor, pred_node):
    mask = np.zeros_like(pred_tumor, dtype=np.uint8)
    mask[pred_tumor > 0] = 1
    mask[pred_node > 0] = 2
    return mask


def patient_id_from_sample(sample):

    if isinstance(sample, bytes):
        sample = sample.decode()

    return os.path.splitext(sample)[0]


def load_tn_predictions(path):

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Dataset Builder
# ============================================================

def resolve_mask(
    npz_path,
    pet,
    ct,
    mask_source,
    seg_model,
    device,
):

    if mask_source == "ground_truth":
        return load_gt_mask(npz_path)

    pred_tumor, pred_node = predict_masks(
        seg_model,
        pet,
        ct,
        device,
    )

    return build_combined_mask(pred_tumor, pred_node)


def build_dataset(
    sample_names,
    data_dir,
    clinical_df,
    tn_predictions,
    seg_model,
    device,
    mask_source="ground_truth",
):
    X = []

    times = []

    events = []

    skipped = 0

    for sample in sample_names:

        patient_id = patient_id_from_sample(sample)

        row = clinical_df.loc[
            clinical_df["PatientID"] == patient_id
        ]

        if len(row) == 0:
            skipped += 1
            continue

        row = row.iloc[0]

        if pd.isna(row["RFS"]) or pd.isna(row["Relapse"]):
            skipped += 1
            continue

        if patient_id not in tn_predictions:
            skipped += 1
            continue

        npz_path = os.path.join(
            data_dir,
            patient_id + ".npz"
        )

        if not os.path.isfile(npz_path):
            skipped += 1
            continue

        pet, ct = load_pet_ct(npz_path)

        mask = resolve_mask(
            npz_path,
            pet,
            ct,
            mask_source,
            seg_model,
            device,
        )

        tn_entry = tn_predictions[patient_id]

        feature_vector = build_survival_feature_vector(
            clinical_row=row,
            pet=pet,
            mask=mask,
            t_probs=tn_entry["t_probs"],
            n_probs=tn_entry["n_probs"],
        )

        X.append(feature_vector)

        times.append(
            float(row["RFS"])
        )

        events.append(
            bool(row["Relapse"])
        )

    print(
        f"Built {len(X)} cases "
        f"(skipped {skipped})"
    )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(times, dtype=np.float32),
        np.asarray(events, dtype=bool),
    )


# ============================================================
# Main
# ============================================================

def main(
    data_dir,
    clinical_csv,
    tn_predictions,
    train_samples,
    valid_samples,
    output_file,
    seg_model=None,
    device="cpu",
    mask_source="ground_truth",
):

    print("Loading clinical CSV...")
    clinical_df = pd.read_csv(
        clinical_csv
    )

    print("Loading TN predictions...")
    tn_predictions = load_tn_predictions(
        tn_predictions
    )

    print(f"mask_source: {mask_source}")

    seg_model_obj = None

    if mask_source == "predicted":
        print(f"Loading segmentation model: {seg_model}")
        seg_model_obj = load_segmentation_model(
            seg_model,
            device,
        )
    else:
        print("Using ground-truth MASK from NPZ (segmentation model not loaded)")

    print("Loading fold splits...")

    train_samples = np.load(
        train_samples,
        allow_pickle=True,
    )

    valid_samples = np.load(
        valid_samples,
        allow_pickle=True,
    )

    print(
        f"Train: {len(train_samples)}"
    )

    print(
        f"Valid: {len(valid_samples)}"
    )

    print("\nBuilding train features...")

    (
        X_train,
        time_train,
        event_train,
    ) = build_dataset(
        train_samples,
        data_dir,
        clinical_df,
        tn_predictions,
        seg_model_obj,
        device,
        mask_source=mask_source,
    )

    print("\nBuilding valid features...")

    (
        X_valid,
        time_valid,
        event_valid,
    ) = build_dataset(
        valid_samples,
        data_dir,
        clinical_df,
        tn_predictions,
        seg_model_obj,
        device,
        mask_source=mask_source,
    )

    assert len(FEATURE_NAMES) == X_train.shape[1], (
        f"Feature name count {len(FEATURE_NAMES)} != "
        f"vector dim {X_train.shape[1]}"
    )

    payload = {

        "X_train":
            X_train,

        "time_train":
            time_train,

        "event_train":
            event_train,

        "X_valid":
            X_valid,

        "time_valid":
            time_valid,

        "event_valid":
            event_valid,

        "feature_names":
            FEATURE_NAMES,

        "mask_source":
            mask_source,
    }

    print(
        f"\nSaving: {output_file}"
    )

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nDone.")

    print(
        f"Train shape: {X_train.shape}"
    )

    print(
        f"Valid shape: {X_valid.shape}"
    )

    print(
        f"Features: {len(FEATURE_NAMES)}"
    )

    print(
        f"mask_source: {mask_source}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
    )

    parser.add_argument(
        "--clinical_csv",
        required=True,
    )

    parser.add_argument(
        "--tn_predictions",
        required=True,
    )

    parser.add_argument(
        "--mask_source",
        choices=["ground_truth", "predicted"],
        default="ground_truth",
        help="Use NPZ MASK (ground_truth) or fold segmentation predictions (predicted)",
    )

    parser.add_argument(
        "--seg_model",
        default=None,
        help="Stage-I best.pt for this fold; required when --mask_source predicted, ignored otherwise",
    )

    parser.add_argument(
        "--train_samples",
        required=True,
    )

    parser.add_argument(
        "--valid_samples",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda",
    )

    args = parser.parse_args()

    if args.mask_source == "predicted" and not args.seg_model:
        parser.error("--seg_model is required when --mask_source predicted")

    main(**vars(args))
