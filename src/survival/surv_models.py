"""
surv_models.py

Survival models for HECKTOR 2026.

Uses:
    scikit-survival

Primary model:
    Random Survival Forest (RSF)
"""

import pickle
import numpy as np

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

from sksurv.ensemble import RandomSurvivalForest


# ============================================================
# Utilities
# ============================================================

def make_survival_target(
    times,
    events,
):
    """
    Convert arrays into scikit-survival format.

    Parameters
    ----------
    times : ndarray
        RFS

    events : ndarray
        Relapse (0/1)

    Returns
    -------
    structured array
    """

    return Surv.from_arrays(
        event=events.astype(bool),
        time=times.astype(float),
    )


# ============================================================
# RSF Training
# ============================================================

def train_rsf(
    X_train,
    time_train,
    event_train,
    random_state=42,
):
    """
    Train Random Survival Forest.
    """

    y_train = make_survival_target(
        time_train,
        event_train,
    )

    model = RandomSurvivalForest(
        n_estimators=1000,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )


    '''
    model = RandomSurvivalForest(
        n_estimators=300,        # 1000 is overkill and increases variance
        min_samples_split=40,    # Increase to prevent hyper-specific splits
        min_samples_leaf=20,     # Ensure leaves represent a solid patient sub-population
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    
    
    
    '''

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# Prediction
# ============================================================

def predict_risk(
    model,
    X,
):
    """
    Produce risk scores.

    Higher risk
    =>
    shorter survival.
    """

    return model.predict(X)


# ============================================================
# Evaluation
# ============================================================

def evaluate_cindex(
    model,
    X,
    times,
    events,
):
    """
    Compute concordance index.
    """

    risk_scores = predict_risk(
        model,
        X,
    )

    cindex = concordance_index_censored(
        events.astype(bool),
        times.astype(float),
        risk_scores,
    )[0]

    return float(cindex)


# ============================================================
# Feature Importance
# ============================================================

def feature_importance(
    model,
    feature_names,
):
    """
    Returns sorted feature importances.

    NOTE:
    RSF permutation importance is
    usually more reliable, but this
    provides a quick overview.
    """

    if not hasattr(model, "feature_importances_"):
        return []

    importance = model.feature_importances_

    pairs = list(
        zip(feature_names, importance)
    )

    pairs.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return pairs


# ============================================================
# Save / Load
# ============================================================

def save_model(
    path,
    model,
    metadata=None,
):
    payload = {
        "model": model,
        "metadata": metadata,
    }

    with open(path, "wb") as f:
        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_model(path):

    with open(path, "rb") as f:
        payload = pickle.load(f)

    return payload["model"]


# ============================================================
# Validation Summary
# ============================================================

def train_and_validate(
    X_train,
    time_train,
    event_train,
    X_valid,
    time_valid,
    event_valid,
    random_state=42,
):
    """
    Convenience wrapper.

    Returns
    -------
    model
    train_cindex
    valid_cindex
    """

    model = train_rsf(
        X_train,
        time_train,
        event_train,
        random_state=random_state,
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

    return (
        model,
        train_cindex,
        valid_cindex,
    )