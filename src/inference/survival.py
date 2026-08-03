"""
survival.py

Five-fold ensemble recurrence-free survival prediction module
for HECKTOR 2026.

Responsibilities
----------------
1. Load the five trained Random Survival Forest models.
2. Predict recurrence-free survival using soft ensemble.
3. Export an RFS-time estimate in days for the challenge output.

This module assumes that

- segmentation has already been completed,
- TN probabilities have already been computed.
"""

import json
import pickle
import numpy as np

from feature_extraction import (
    extract_survival_features
)


# ============================================================
# SURVIVAL CURVE -> RFS TIME
# ============================================================

def _expected_survival_time(surv_fn):
    """
    Restricted mean survival time from the area under S(t).
    """

    times = np.asarray(surv_fn.x, dtype=np.float64)
    probs = np.asarray(surv_fn.y, dtype=np.float64)

    if times.size == 0:
        return np.nan

    if times.size == 1:
        return float(times[0] * probs[0])

    return float(np.trapz(probs, times))


def _rfs_from_survival_function(surv_fn):
    """
    Convert one survival curve into an RMST-based RFS estimate in days.
    """

    expected = _expected_survival_time(surv_fn)

    if np.isfinite(expected) and expected > 0.0:
        return expected

    return np.nan


def _risk_to_rfs_fallback(risk):
    """
    Fallback export when RMST extraction is unavailable.
    """

    return float(-1.0 * risk)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):
    """
    Load one trained Random Survival Forest model.
    """

    with open(model_path, "rb") as f:

        payload = pickle.load(f)

    return payload["model"]


# ============================================================
# SURVIVAL ENSEMBLE
# ============================================================

class SurvivalPredictor:

    def __init__(
        self,
        model_paths
    ):
        """
        Parameters
        ----------
        model_paths : list[str]

            Five trained RSF models.
        """

        self.models = []

        for path in model_paths:

            self.models.append(
                load_model(path)
            )

    # ============================================================
    # PREDICT RFS
    # ============================================================

    def predict(
        self,
        pet,
        mask,
        clinical_row,
        t_probs,
        n_probs
    ):
        """
        Predict recurrence-free survival using a five-fold ensemble.

        Parameters
        ----------
        pet : ndarray

        mask : ndarray

        clinical_row : pandas.Series

        t_probs : ndarray
            Ensemble T-stage probabilities.

        n_probs : ndarray
            Ensemble N-stage probabilities.

        Returns
        -------
        dict

            {
                "risk": float,
                "rfs_days": float,
            }
        """

        # --------------------------------------------------------
        # FEATURE EXTRACTION
        # --------------------------------------------------------

        features = extract_survival_features(
            pet,
            mask,
            clinical_row,
            t_probs,
            n_probs
        )

        features = features.reshape(1, -1)

        # --------------------------------------------------------
        # ENSEMBLE PREDICTION
        # --------------------------------------------------------

        risk_sum = 0.0
        rfs_values = []

        for model in self.models:

            risk = float(model.predict(features)[0])
            risk_sum += risk

            surv_fns = model.predict_survival_function(features)
            rfs_estimate = _rfs_from_survival_function(surv_fns[0])

            if np.isfinite(rfs_estimate) and rfs_estimate > 0.0:
                rfs_values.append(rfs_estimate)

        risk = risk_sum / len(self.models)

        if rfs_values:
            rfs_days = float(np.mean(rfs_values))
        else:
            rfs_days = _risk_to_rfs_fallback(risk)

        if not np.isfinite(rfs_days) or rfs_days <= 0.0:
            rfs_days = _risk_to_rfs_fallback(risk)

        return {

            "risk": float(risk),
            "rfs_days": float(rfs_days),

        }
    
    # ============================================================
    # SAVE JSON
    # ============================================================

    def save_json(
        self,
        prediction,
        output_json
    ):
        """
        Save predicted RFS time in days for the HECKTOR 2026 challenge.

        Example
        -------
        842.0
        """

        with open(output_json, "w") as f:

            json.dump(
                float(prediction["rfs_days"]),
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
        t_probs,
        n_probs,
        output_json
    ):
        """
        Complete survival prediction pipeline.

        Parameters
        ----------
        pet : ndarray

        mask : ndarray

        clinical_row : pandas.Series

        t_probs : ndarray

        n_probs : ndarray

        output_json : str

        Returns
        -------
        dict

            {
                "risk": float,
                "rfs_days": float,
            }
        """

        prediction = self.predict(

            pet,

            mask,

            clinical_row,

            t_probs,

            n_probs

        )

        self.save_json(

            prediction,

            output_json

        )

        return prediction