"""
tn_staging.py

Five-fold ensemble TN staging module for HECKTOR 2026.

Responsibilities
----------------
1. Load the five T-stage models.
2. Load the five N-stage models.
3. Predict TN probabilities using soft voting.
4. Save the predicted TN stage in JSON format.

This module does NOT:

- perform segmentation
- extract handcrafted features
- perform survival prediction
"""

import json
import numpy as np

from catboost import CatBoostClassifier

from feature_extraction import (
    extract_tn_features
)


# ============================================================
# LABEL MAPS
# ============================================================

INT_TO_T_STAGE = {

    0: "T0",
    1: "T1",
    2: "T2",
    3: "T3",
    4: "T4"

}

INT_TO_N_STAGE = {

    0: "N0",
    1: "N1",
    2: "N2",
    3: "N3"

}


# ============================================================
# LOAD MODELS
# ============================================================

def load_t_model(model_path):
    """
    Load one trained T-stage model.
    """

    model = CatBoostClassifier()

    model.load_model(model_path)

    return model


def load_n_model(model_path):
    """
    Load one trained N-stage model.
    """

    model = CatBoostClassifier()

    model.load_model(model_path)

    return model


# ============================================================
# TN STAGING ENSEMBLE
# ============================================================

class TNStager:

    def __init__(
        self,
        t_model_paths,
        n_model_paths
    ):

        self.t_models = []

        self.n_models = []

        for path in t_model_paths:

            self.t_models.append(
                load_t_model(path)
            )

        for path in n_model_paths:

            self.n_models.append(
                load_n_model(path)
            )

    # ============================================================
    # PREDICT PROBABILITIES
    # ============================================================

    def predict_proba(
        self,
        pet,
        mask,
        clinical_row
    ):
        """
        Predict TN probabilities using a
        five-fold soft-voting ensemble.

        Returns
        -------
        dict
        {

            "t_stage"
            "n_stage"

            "t_probs"
            "n_probs"

        }
        """

        # --------------------------------------------------------
        # FEATURE EXTRACTION
        # --------------------------------------------------------

        features = extract_tn_features(
            pet,
            mask,
            clinical_row
        )

        features = features.reshape(1, -1)

        # --------------------------------------------------------
        # T-STAGE ENSEMBLE
        # --------------------------------------------------------

        t_sum = None

        for model in self.t_models:

            probs = model.predict_proba(
                features
            )[0]

            if t_sum is None:

                t_sum = probs

            else:

                t_sum += probs

        t_probs = (
            t_sum
            / len(self.t_models)
        )

        # --------------------------------------------------------
        # N-STAGE ENSEMBLE
        # --------------------------------------------------------

        n_sum = None

        for model in self.n_models:

            probs = model.predict_proba(
                features
            )[0]

            if n_sum is None:

                n_sum = probs

            else:

                n_sum += probs

        n_probs = (
            n_sum
            / len(self.n_models)
        )

        # --------------------------------------------------------
        # FINAL LABELS
        # --------------------------------------------------------

        t_index = int(
            np.argmax(
                t_probs
            )
        )

        n_index = int(
            np.argmax(
                n_probs
            )
        )

        return {

            "t_stage":
                INT_TO_T_STAGE[t_index],

            "n_stage":
                INT_TO_N_STAGE[n_index],

            "t_probs":
                t_probs.astype(
                    np.float32
                ),

            "n_probs":
                n_probs.astype(
                    np.float32
                )

        }


    # ============================================================
    # PREDICT
    # ============================================================

    def predict(
        self,
        pet,
        mask,
        clinical_row
    ):
        """
        Convenience wrapper.

        Identical to predict_proba().
        """

        return self.predict_proba(
            pet,
            mask,
            clinical_row
        )
    
        # ============================================================
    # SAVE JSON
    # ============================================================

    def save_json(
        self,
        prediction,
        t_json,
        n_json
    ):

        with open(t_json, "w") as f:

            json.dump(
                prediction["t_stage"],
                f
            )

        with open(n_json, "w") as f:

            json.dump(
                prediction["n_stage"],
                f
            )


    # ============================================================
    # PREDICT AND SAVE
    # ============================================================

    def predict_and_save(
        self,
        pet,
        mask,
        clinical_row,
        t_json,
        n_json
        ):
        """
        Complete TN staging pipeline.

        1. Extract TN features.
        2. Predict TN stage.
        3. Save JSON.

        Returns
        -------
        dict

            {
                "t_stage"
                "n_stage"
                "t_probs"
                "n_probs"
            }
        """

        prediction = self.predict(

            pet,

            mask,

            clinical_row

        )

        self.save_json(

            prediction,

            t_json,
            n_json

        )

        return prediction