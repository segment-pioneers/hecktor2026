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

def extract_features(data_dir, clinical_csv, output_cache):

    output_dir = os.path.dirname(output_cache)

    if output_dir == "":
        output_dir = "."

    logger = setup_logging(output_dir)

    logger.info("Loading NPZ files...")

    sample_names = (load_all_npz_files(data_dir))

    logger.info("Found %d NPZ files", len(sample_names))

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
        )
    )

    elapsed = (time.time() - start_time)

    logger.info("Feature extraction completed in %.2f sec", elapsed)

    logger.info("Feature matrix shape: %s", str(X.shape))

    logger.info("Saving feature cache...")

    tn_features.save_feature_cache(output_cache, X, patient_ids)

    logger.info("Saved: %s", output_cache)

    logger.info("Done.")

    print("\n=== Feature Extraction Complete ===")
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

    args = parser.parse_args()

    extract_features(
        data_dir=args.data_dir,
        clinical_csv=args.clinical_csv,
        output_cache=args.output_cache,
    )

