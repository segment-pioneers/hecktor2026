import os
import time
import logging
import pickle

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from . import tn_features


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

def setup_logging(output_dir):

    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("extract_tn_features")

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    log_path = os.path.join(output_dir, "extract_tn_features.log")

    fh = logging.FileHandler(log_path)

    fh.setFormatter(formatter)

    sh = logging.StreamHandler()

    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def load_all_npz_files(data_dir):

    files = []

    for fname in sorted(
        os.listdir(data_dir)
    ):

        if fname.endswith(".npz"):

            files.append(fname.encode())

    return np.asarray(files, dtype=object)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def extract_features(
    data_dir,
    clinical_csv,
    output_cache,
    mask_source="ground_truth",
    seg_model=None,
    device="cpu",
):

    output_dir = os.path.dirname(output_cache)

    if output_dir == "":
        output_dir = "."

    logger = setup_logging(output_dir)

    logger.info("mask_source=%s", mask_source)

    if mask_source == "predicted":
        logger.info("Segmentation model: %s", seg_model)
        logger.info("Device: %s", device)

    logger.info("Loading NPZ files...")

    sample_names = (load_all_npz_files(data_dir))

    logger.info("Found %d NPZ files", len(sample_names))

    if mask_source == "predicted":
        logger.info(
            "Building feature table "
            "(predicted masks from segmentation model)..."
        )
    else:
        logger.info(
            "Building feature table "
            "(ground-truth MASK from NPZ)..."
        )

    start_time = time.time()

    X, patient_ids = (
        tn_features.build_feature_table(
            data_dir=data_dir,
            sample_names=sample_names,
            clinical_csv=clinical_csv,
            mask_source=mask_source,
            seg_model=seg_model,
            device=device,
        )
    )

    elapsed = (time.time() - start_time)

    logger.info("Feature extraction completed in %.2f sec", elapsed)

    logger.info("Feature matrix shape: %s", str(X.shape))

    logger.info("Saving feature cache...")

    tn_features.save_feature_cache(
        output_cache,
        X,
        patient_ids,
        mask_source=mask_source,
    )

    logger.info("Saved: %s", output_cache)

    logger.info("Done.")

    print("\n=== Feature Extraction Complete ===")
    print(f"mask_source : {mask_source}")
    print(f"Patients : {len(patient_ids)}")
    print(f"Features : {X.shape[1]}")
    print(f"Cache    : {output_cache}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .npz files")
    parser.add_argument("--clinical_csv", type=str, required=True, help="HECKTOR clinical CSV")
    parser.add_argument("--output_cache", type=str, required=True, help="Output .pkl file")
    parser.add_argument(
        "--mask_source",
        type=str,
        choices=["ground_truth", "predicted"],
        default="ground_truth",
        help="Use NPZ MASK (ground_truth) or fold segmentation predictions (predicted)",
    )
    parser.add_argument(
        "--seg_model",
        type=str,
        default=None,
        help="Fold segmentation best.pt; required when --mask_source predicted, ignored otherwise",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda (used when --mask_source predicted)",
    )

    args = parser.parse_args()

    if args.mask_source == "predicted" and not args.seg_model:
        parser.error("--seg_model is required when --mask_source predicted")

    extract_features(
        data_dir=args.data_dir,
        clinical_csv=args.clinical_csv,
        output_cache=args.output_cache,
        mask_source=args.mask_source,
        seg_model=args.seg_model,
        device=args.device,
    )

