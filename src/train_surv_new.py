"""
train_surv_new.py

Train survival model for one fold.

Pipeline
--------
survival_features.pkl
        ↓
Random Survival Forest
        ↓
best.pkl
latest.pkl

Validation metric:
    C-index
"""

import os
import csv
import json
import pickle
import logging
import argparse

from surv_models import (
    train_rsf,
    evaluate_cindex,
    feature_importance,
)


# ============================================================
# Logging
# ============================================================

def setup_logging(log_dir):

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("train_surv")

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(
        os.path.join(log_dir, "train.log")
    )

    fh.setFormatter(formatter)

    sh = logging.StreamHandler()

    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# ============================================================
# CSV
# ============================================================

def save_metrics_csv(
    path,
    train_cindex,
    valid_cindex,
):

    with open(path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "train_cindex",
                "valid_cindex",
            ]
        )

        writer.writerow(
            [
                train_cindex,
                valid_cindex,
            ]
        )


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(
    path,
    model,
    train_cindex,
    valid_cindex,
    feature_names,
):

    payload = {

        "model":
            model,

        "train_cindex":
            train_cindex,

        "valid_cindex":
            valid_cindex,

        "feature_names":
            feature_names,

    }

    with open(path, "wb") as f:
        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# ============================================================
# Feature Importance
# ============================================================

def save_feature_importance(
    path,
    model,
    feature_names,
):

    try:

        importance = feature_importance(
            model,
            feature_names,
        )

        with open(path, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "feature",
                    "importance",
                ]
            )

            for name, value in importance:

                writer.writerow(
                    [
                        name,
                        value,
                    ]
                )

    except Exception as e:

        print(
            "Warning: failed to save feature importance:",
            e
        )


# ============================================================
# Main
# ============================================================

def train(
    features_file,
    model_dir,
    random_state=42,
):

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    logger = setup_logging(
        model_dir
    )

    logger.info(
        "Loading features: %s",
        features_file,
    )

    with open(features_file, "rb") as f:

        data = pickle.load(f)

    X_train = data["X_train"]
    X_valid = data["X_valid"]

    time_train = data["time_train"]
    event_train = data["event_train"]

    time_valid = data["time_valid"]
    event_valid = data["event_valid"]

    feature_names = data["feature_names"]

    logger.info(
        "Train shape: %s",
        str(X_train.shape),
    )

    logger.info(
        "Valid shape: %s",
        str(X_valid.shape),
    )

    logger.info(
        "Training Random Survival Forest..."
    )

    model = train_rsf(
        X_train,
        time_train,
        event_train,
        random_state=random_state,
    )

    logger.info(
        "Computing C-index..."
    )

    train_cindex = evaluate_cindex(
        model,
        X_train,
        time_train,
        event_train,
    )

    valid_cindex = evaluate_cindex(
        model,
        X_valid,
        time_valid,
        event_valid,
    )

    logger.info(
        "Train C-index: %.4f",
        train_cindex,
    )

    logger.info(
        "Valid C-index: %.4f",
        valid_cindex,
    )

    #
    # Save checkpoints
    #

    best_path = os.path.join(
        model_dir,
        "best.pkl",
    )

    latest_path = os.path.join(
        model_dir,
        "latest.pkl",
    )

    save_checkpoint(
        best_path,
        model,
        train_cindex,
        valid_cindex,
        feature_names,
    )

    save_checkpoint(
        latest_path,
        model,
        train_cindex,
        valid_cindex,
        feature_names,
    )

    logger.info(
        "Saved: %s",
        best_path,
    )

    logger.info(
        "Saved: %s",
        latest_path,
    )

    #
    # Metrics CSV
    #

    metrics_path = os.path.join(
        model_dir,
        "metrics.csv",
    )

    save_metrics_csv(
        metrics_path,
        train_cindex,
        valid_cindex,
    )

    #
    # Feature importance
    #

    importance_path = os.path.join(
        model_dir,
        "feature_importance.csv",
    )

    save_feature_importance(
        importance_path,
        model,
        feature_names,
    )

    #
    # Config
    #

    config_path = os.path.join(
        model_dir,
        "config.json",
    )

    with open(
        config_path,
        "w",
    ) as f:

        json.dump(
            {
                "random_state":
                    random_state,

                "n_train":
                    int(X_train.shape[0]),

                "n_valid":
                    int(X_valid.shape[0]),

                "n_features":
                    int(X_train.shape[1]),

                "train_cindex":
                    float(train_cindex),

                "valid_cindex":
                    float(valid_cindex),
            },
            f,
            indent=4,
        )

    logger.info(
        "=" * 60
    )

    logger.info(
        "Training finished"
    )

    logger.info(
        "Train C-index: %.4f",
        train_cindex,
    )

    logger.info(
        "Valid C-index: %.4f",
        valid_cindex,
    )

    logger.info(
        "Best model: %s",
        best_path,
    )

    logger.info(
        "Latest model: %s",
        latest_path,
    )

    logger.info(
        "=" * 60
    )

    print("\n=== Training Complete ===")
    print(
        f"Train C-index: {train_cindex:.4f}"
    )
    print(
        f"Valid C-index: {valid_cindex:.4f}"
    )
    print(
        f"Best model: {best_path}"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features_file",
        required=True,
        help="survival_features.pkl",
    )

    parser.add_argument(
        "--model_dir",
        required=True,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    train(
        **vars(args)
    )