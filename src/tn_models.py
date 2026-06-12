import os
import json
import pickle
import numpy as np

from catboost import CatBoostClassifier

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
)


# ------------------------------------------------------------
# Label Encoders
# ------------------------------------------------------------

T_STAGE_TO_INT = {
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
}

INT_TO_T_STAGE = {
    v: k
    for k, v in T_STAGE_TO_INT.items()
}

N_STAGE_TO_INT = {
    "N0": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
}

INT_TO_N_STAGE = {
    v: k
    for k, v in N_STAGE_TO_INT.items()
}


def encode_t_stage(stage):

    if stage not in T_STAGE_TO_INT:
        raise ValueError(
            f"Unknown T-stage: {stage}"
        )

    return T_STAGE_TO_INT[stage]


def encode_n_stage(stage):

    if stage not in N_STAGE_TO_INT:
        raise ValueError(
            f"Unknown N-stage: {stage}"
        )

    return N_STAGE_TO_INT[stage]


def decode_t_stage(idx):
    return INT_TO_T_STAGE[int(idx)]


def decode_n_stage(idx):
    return INT_TO_N_STAGE[int(idx)]


# ------------------------------------------------------------
# Target Extraction — disabled: use train_tn.build_targets instead
# (returns kept_ids; avoids X/y length mismatch when rows are skipped)
# ------------------------------------------------------------

# def build_targets(
#     clinical_df,
#     patient_ids
# ):
#
#     y_t = []
#     y_n = []
#
#     for pid in patient_ids:
#
#         row = clinical_df.loc[pid]
#
#         t_stage = row["T-stage"]
#         n_stage = row["N-stage"]
#
#         if str(t_stage) == "nan":
#             continue
#
#         y_t.append(
#             encode_t_stage(t_stage)
#         )
#
#         y_n.append(
#             encode_n_stage(n_stage)
#         )
#
#     return (
#         np.asarray(y_t, dtype=np.int64),
#         np.asarray(y_n, dtype=np.int64),
#     )


# ------------------------------------------------------------
# CatBoost Eval Metric (matches sklearn balanced_accuracy_score)
# ------------------------------------------------------------

class BalancedAccuracyMetric(object):

    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        scores = np.vstack(approxes).T
        pred = np.argmax(scores, axis=1)
        y_true = np.asarray(target, dtype=np.int64).flatten()
        score = balanced_accuracy_score(y_true, pred)
        return score, 1.0


# ------------------------------------------------------------
# CatBoost Factory
# ------------------------------------------------------------

def create_t_model(
    random_seed=42
):

    return CatBoostClassifier(
        loss_function="MultiClass",

        auto_class_weights="Balanced",

        iterations=1000,
        learning_rate=0.03,
        depth=6,

        eval_metric=BalancedAccuracyMetric(),

        random_seed=random_seed,

        verbose=False,
    )


def create_n_model(
    random_seed=42
):

    return CatBoostClassifier(
        loss_function="MultiClass",

        auto_class_weights="Balanced",

        iterations=1000,
        learning_rate=0.03,
        depth=6,

        eval_metric=BalancedAccuracyMetric(),

        random_seed=random_seed,

        verbose=False,
    )


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_t_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    random_seed=42,
):

    model = create_t_model(
        random_seed=random_seed
    )

    model.fit(
        X_train,
        y_train,

        eval_set=(
            X_valid,
            y_valid
        ),

        use_best_model=True,
    )

    return model


def train_n_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    random_seed=42,
):

    model = create_n_model(
        random_seed=random_seed
    )

    model.fit(
        X_train,
        y_train,

        eval_set=(
            X_valid,
            y_valid
        ),

        use_best_model=True,
    )

    return model


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_t_model(
    model,
    X,
    y
):

    pred = model.predict(X)

    pred = pred.astype(
        np.int64
    ).flatten()

    score = balanced_accuracy_score(
        y,
        pred
    )

    report = classification_report(
        y,
        pred,
        digits=4,
        zero_division=0,
    )

    return (
        score,
        pred,
        report
    )


def evaluate_n_model(
    model,
    X,
    y
):

    pred = model.predict(X)

    pred = pred.astype(
        np.int64
    ).flatten()

    score = balanced_accuracy_score(
        y,
        pred
    )

    report = classification_report(
        y,
        pred,
        digits=4,
        zero_division=0,
    )

    return (
        score,
        pred,
        report
    )


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

def predict_t_stage(
    model,
    features
):

    pred = model.predict(
        features
    )

    pred = int(pred[0])

    return decode_t_stage(pred)


def predict_n_stage(
    model,
    features
):

    pred = model.predict(
        features
    )

    pred = int(pred[0])

    return decode_n_stage(pred)


def predict_t_proba(
    model,
    features
):

    probs = model.predict_proba(
        features
    )[0]

    return {
        decode_t_stage(i): float(p)
        for i, p in enumerate(probs)
    }


def predict_n_proba(
    model,
    features
):

    probs = model.predict_proba(
        features
    )[0]

    return {
        decode_n_stage(i): float(p)
        for i, p in enumerate(probs)
    }


# ------------------------------------------------------------
# Save / Load
# ------------------------------------------------------------

def save_t_model(
    model,
    model_path
):

    model.save_model(
        model_path
    )


def save_n_model(
    model,
    model_path
):

    model.save_model(
        model_path
    )


def load_t_model(
    model_path
):

    model = create_t_model()

    model.load_model(
        model_path
    )

    return model


def load_n_model(
    model_path
):

    model = create_n_model()

    model.load_model(
        model_path
    )

    return model


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

def save_training_metadata(
    output_json,
    metadata
):

    with open(
        output_json,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )


def load_training_metadata(
    input_json
):

    with open(
        input_json,
        "r"
    ) as f:

        metadata = json.load(f)

    return metadata