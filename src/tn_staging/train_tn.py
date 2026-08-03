import os
import csv
import json
import time
import random
import logging

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from . import tn_features, tn_models


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

def setup_logging(log_dir):

    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train_tn.log")

    logger = logging.getLogger("train_tn")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# ------------------------------------------------------------
# Metrics CSV
# ------------------------------------------------------------

def init_metrics_csv(metrics_path):

    if os.path.isfile(metrics_path):
        return

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_bal_acc", "n_bal_acc", "mean_bal_acc"])


def append_metrics_csv(metrics_path, t_bal_acc, n_bal_acc, mean_bal_acc):

    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([t_bal_acc, n_bal_acc, mean_bal_acc])


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_split_patient_ids(split_file):

    split = np.load(split_file, allow_pickle=True)

    patient_ids = []

    for sample in split:

        patient_id = bytes.decode(sample).replace(".npz", "")

        patient_ids.append(patient_id)

    return patient_ids


def build_index_map(patient_ids):

    return {
        pid: idx
        for idx, pid in enumerate(patient_ids)
    }


def build_targets(clinical_df, patient_ids):

    y_t = []
    y_n = []

    kept_ids = []

    for pid in patient_ids:

        if pid not in clinical_df.index:
            continue

        row = clinical_df.loc[pid]

        t_stage = row["T-stage"]

        if pd.isna(t_stage):
            continue

        n_stage = row["N-stage"]

        y_t.append(tn_models.encode_t_stage(t_stage))

        y_n.append(tn_models.encode_n_stage(n_stage))

        kept_ids.append(pid)

    return (np.asarray(y_t), np.asarray(y_n), kept_ids)


def subset_features(X, cache_patient_ids, desired_patient_ids):

    idx_map = build_index_map(cache_patient_ids)

    rows = []

    for pid in desired_patient_ids:

        rows.append(idx_map[pid])

    return X[rows]


# ------------------------------------------------------------
# Main Training
# ------------------------------------------------------------

def train(feature_cache, clinical_csv, train_samples, valid_samples, model_dir, seed=42):

    start_time = time.time()

    set_seed(seed)

    os.makedirs(model_dir, exist_ok=True)
    logger = setup_logging(model_dir)

    metrics_path = os.path.join(model_dir, "metrics.csv")
    init_metrics_csv(metrics_path)

    logger.info("Loading feature cache...")
    X_all, cache_patient_ids = (tn_features.load_feature_cache(feature_cache))
    logger.info("Feature matrix shape: %s", str(X_all.shape))

    clinical_df = pd.read_csv(clinical_csv)
    clinical_df = clinical_df.set_index("PatientID")

    train_patient_ids = (load_split_patient_ids(train_samples))
    valid_patient_ids = (load_split_patient_ids(valid_samples))

    y_t_train, y_n_train, train_patient_ids = (build_targets(clinical_df, train_patient_ids))
    y_t_valid, y_n_valid, valid_patient_ids = (build_targets(clinical_df, valid_patient_ids))

    X_train = subset_features(X_all, cache_patient_ids, train_patient_ids)
    X_valid = subset_features(X_all, cache_patient_ids, valid_patient_ids)
    logger.info("Train samples: %d", len(X_train))
    logger.info("Valid samples: %d", len(X_valid))

    # --------------------------------------------------------
    # Train T-stage
    # --------------------------------------------------------

    logger.info("Training T-stage model...")

    t_model = tn_models.train_t_model(X_train, y_t_train, X_valid, y_t_valid, random_seed=seed)

    best_iteration_t = t_model.get_best_iteration()
    logger.info("T-stage best iteration (val BA): %d", best_iteration_t)

    t_bal_acc, _, t_report = (tn_models.evaluate_t_model(t_model, X_valid, y_t_valid))

    logger.info("\nT-stage report:\n%s", t_report)

    # --------------------------------------------------------
    # Train N-stage
    # --------------------------------------------------------

    logger.info("Training N-stage model...")

    n_model = tn_models.train_n_model(X_train, y_n_train, X_valid, y_n_valid, random_seed=seed)

    best_iteration_n = n_model.get_best_iteration()
    logger.info("N-stage best iteration (val BA): %d", best_iteration_n)

    n_bal_acc, _, n_report = (tn_models.evaluate_n_model(n_model, X_valid, y_n_valid))

    logger.info("\nN-stage report:\n%s", n_report)

    mean_bal_acc = (t_bal_acc + n_bal_acc) / 2.0

    logger.info("T-stage BA: %.4f", t_bal_acc)
    logger.info("N-stage BA: %.4f", n_bal_acc)
    logger.info("Mean BA: %.4f", mean_bal_acc)
    append_metrics_csv(metrics_path, t_bal_acc, n_bal_acc, mean_bal_acc)

    tn_models.save_t_model(t_model, os.path.join(model_dir, "best_t_model.cbm"))
    tn_models.save_n_model(n_model, os.path.join(model_dir, "best_n_model.cbm"))

    metadata = {
        "selection_metric": "balanced_accuracy",
        "best_iteration_t": int(best_iteration_t),
        "best_iteration_n": int(best_iteration_n),
        "t_bal_acc": float(t_bal_acc),
        "n_bal_acc": float(n_bal_acc),
        "mean_bal_acc": float(mean_bal_acc),
        "train_size": int(len(X_train)),
        "valid_size": int(len(X_valid)),
        "seed": int(seed),
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:

        json.dump(metadata, f, indent=2)

    elapsed = (time.time() - start_time)

    logger.info("Finished in %.2f sec", elapsed)

    print("\n=== TN Training Complete ===")
    print(f"T-stage BA : {t_bal_acc:.4f}")
    print(f"N-stage BA : {n_bal_acc:.4f}")
    print(f"Mean BA    : {mean_bal_acc:.4f}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--feature_cache", type=str, required=True)
    parser.add_argument("--clinical_csv", type=str, required=True)
    parser.add_argument("--train_samples", type=str, required=True)
    parser.add_argument("--valid_samples", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(**vars(args))

