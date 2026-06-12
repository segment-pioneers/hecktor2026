"""
predict_tn.py

Generate TN probabilities for survival prediction.

Input
-----
tn_feature_cache.pkl

Models
------
best_t_model.cbm
best_n_model.cbm

Output
------
tn_predictions.pkl

Format
------
{
    patient_id:
    {
        "t_probs": np.ndarray(5,),
        "n_probs": np.ndarray(4,)
    }
}
"""

import os
import pickle
import argparse

import numpy as np

import tn_features
import tn_models


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def predict(
    feature_cache,
    t_model_path,
    n_model_path,
    output_file,
):

    print("Loading feature cache...")

    X_all, patient_ids = (
        tn_features.load_feature_cache(
            feature_cache
        )
    )

    print(
        f"Feature matrix: {X_all.shape}"
    )

    print(
        f"Patients: {len(patient_ids)}"
    )

    print("Loading models...")

    t_model = tn_models.load_t_model(
        t_model_path
    )

    n_model = tn_models.load_n_model(
        n_model_path
    )

    print("Computing probabilities...")

    #
    # CatBoost returns:
    #
    # T-stage:
    #   [N, 5]
    #
    # N-stage:
    #   [N, 4]
    #

    t_probs = t_model.predict_proba(
        X_all
    )

    n_probs = n_model.predict_proba(
        X_all
    )

    predictions = {}

    for idx, patient_id in enumerate(patient_ids):

        predictions[patient_id] = {

            "t_probs":
                t_probs[idx].astype(
                    np.float32
                ),

            "n_probs":
                n_probs[idx].astype(
                    np.float32
                ),
        }

    print(
        f"Saving: {output_file}"
    )

    with open(
        output_file,
        "wb"
    ) as f:

        pickle.dump(
            predictions,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        f"Saved probabilities for "
        f"{len(predictions)} patients"
    )

    #
    # Small sanity check
    #

    first_patient = next(
        iter(predictions.keys())
    )

    print("\nExample:")

    print(
        first_patient
    )

    print(
        "T:",
        predictions[first_patient][
            "t_probs"
        ]
    )

    print(
        "N:",
        predictions[first_patient][
            "n_probs"
        ]
    )

    print("\nDone.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_cache",
        required=True,
        type=str,
        help="tn_feature_cache.pkl",
    )

    parser.add_argument(
        "--t_model_path",
        required=True,
        type=str,
        help="best_t_model.cbm",
    )

    parser.add_argument(
        "--n_model_path",
        required=True,
        type=str,
        help="best_n_model.cbm",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="tn_predictions.pkl",
    )

    args = parser.parse_args()

    predict(
        **vars(args)
    )