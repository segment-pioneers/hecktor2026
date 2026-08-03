"""

HECKTOR 2026 Challenge — Inference Entry Point

This script is executed at container runtime on Grand Challenge.
It reads a paired CT + PET scan and clinical data (EHR) from /input,
runs the three-stage pipeline, and writes all outputs to /output:

  Subtask 1 — Segmentation:
      /output/images/head-neck-tumor-segmentation/output.mha
        (label 0=background, 1=GTVp, 2=GTVn)

  Subtask 2 — TN Staging:
      /output/t-stage.json   → "T2"
      /output/n-stage.json   → "N1"

  Subtask 3 — Prognosis (RFS):
      /output/rfs.json   → <float> RFS time in days

Model weights are loaded from /opt/ml/model at runtime (uploaded separately
as a tarball to Grand Challenge via Algorithm > Models).

To test locally:
    ./do_test_run.sh

To save and upload:
    ./do_save.sh
"""

#Build: OOM-safe float32 2026-07-23

import json
from glob import glob
from pathlib import Path
import gc

import pandas as pd

import SimpleITK
import numpy as np

from preprocessing import preprocess_patient
from segmentation import Segmentor
from tn_staging import TNStager
from survival import SurvivalPredictor

# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
NUM_FOLDS = 5

# ============================================================
# LOAD MODELS
# ============================================================

def load_models():
    """
    Load all trained models.

    Returns
    -------
    segmentor : Segmentor

    tn_stager : TNStager

    survival_predictor : SurvivalPredictor
    """

    print("Loading segmentation models...", flush=True)

    segmentor = Segmentor([
        str(MODEL_PATH / f"seg_fold{i}.pt")
        for i in range(1, NUM_FOLDS + 1)
    ])

    print("Loading TN staging models...", flush=True)

    tn_stager = TNStager(

        t_model_paths=[
            str(MODEL_PATH / f"best_t_model_fold{i}.cbm")
            for i in range(1, NUM_FOLDS + 1)
        ],
        n_model_paths=[
            str(MODEL_PATH / f"best_n_model_fold{i}.cbm")
            for i in range(1, NUM_FOLDS + 1)
        ],

    )

    print("Loading survival models...", flush=True)

    survival_predictor = SurvivalPredictor([
        str(MODEL_PATH / f"survival_fold{i}.pkl")
        for i in range(1, NUM_FOLDS + 1)
    ])

    print("All models loaded successfully.\n", flush=True)

    return (segmentor, tn_stager, survival_predictor)

def run():

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    ct_path = get_image_file(INPUT_PATH / "images/ct")
    pet_path = get_image_file(INPUT_PATH / "images/pet")
    ehr = load_json(INPUT_PATH / "ehr.json")
    clinical_row = pd.Series(ehr)

    # ------------------------------------------------------------------
    # 2. Preprocess before loading models (peak RAM)
    # ------------------------------------------------------------------
    print("Preprocessing CT/PET...", flush=True)
    data = preprocess_patient(ct_path, pet_path)
    gc.collect()
    print("Preprocessing complete.\n", flush=True)

    # ------------------------------------------------------------------
    # 3. Load models
    # ------------------------------------------------------------------
    segmentor, tn_stager, survival_predictor = load_models()

    # ------------------------------------------------------------------
    # 4. Subtask 1 — Segmentation
    # ------------------------------------------------------------------
    seg_prediction = run_segmentation(
        segmentor=segmentor,
        data=data,
        ehr=ehr,
    )

    output_location = OUTPUT_PATH / "images/head-neck-tumor-segmentation"

    write_segmentation(
        location=output_location,
        array=seg_prediction["mask"],
        reference_path=ct_path,
    )

    # ------------------------------------------------------------------
    # 5. Subtask 2 — TN Staging
    # ------------------------------------------------------------------
    tn_prediction = run_tn_staging(tn_stager=tn_stager, pet=seg_prediction["pet"], mask=seg_prediction["prediction"], clinical_row=clinical_row,)

    write_json(location=OUTPUT_PATH / "t-stage.json", data=tn_prediction["t_stage"])
    write_json(location=OUTPUT_PATH / "n-stage.json", data=tn_prediction["n_stage"])

    # ------------------------------------------------------------------
    # 6. Subtask 3 — Prognosis
    # ------------------------------------------------------------------
    survival_prediction = run_prognosis(
        survival_predictor=survival_predictor,
        pet=seg_prediction["pet"],
        mask=seg_prediction["prediction"],
        clinical_row=clinical_row,
        t_probs=tn_prediction["t_probs"],
        n_probs=tn_prediction["n_probs"],
    )

    write_json(location=OUTPUT_PATH / "rfs.json", data=float(survival_prediction["rfs_days"]))

    return 0


# =============================================================================
# Subtask 1 — Segmentation
# =============================================================================

def run_segmentation(
    segmentor,
    data,
    ehr,
):
    """
    Run segmentation.

    Parameters
    ----------
    segmentor : Segmentor

    data : dict
        Output of preprocess_patient.

    ehr : dict
        Not used by the segmentation model.
        Included for consistency with the official template.

    Returns
    -------
    dict

        {
            "mask":
                ndarray
                Original-space segmentation.
                Used for saving output.mha.

            "prediction":
                ndarray
                224×224×224 segmentation used for
                handcrafted feature extraction.

            "pet":
                ndarray
                Preprocessed PET.

            "ct":
                ndarray
                Preprocessed CT.

            "tumor_probability":
                ndarray

            "node_probability":
                ndarray
        }
    """

    #
    # Segmentor.predict() performs
    #
    #   - ensemble inference
    #   - reconstruction
    #   - saving-ready mask generation
    #
    # Preprocessing is done once in run() before models are loaded.
    #

    seg_prediction = segmentor.predict(

        data=data,

        output_mha=None,

    )

    return seg_prediction


# =============================================================================
# Subtask 2 — TN Staging
# =============================================================================

def run_tn_staging(
    tn_stager,
    pet,
    mask,
    clinical_row,
):
    """
    Run TN staging.

    Parameters
    ----------
    tn_stager : TNStager

    pet : ndarray
        Preprocessed PET volume.

    mask : ndarray
        Predicted segmentation mask in the 224×224×224
        preprocessed space.

    clinical_row : pandas.Series
        Clinical information for the current patient.

    Returns
    -------
    dict

        {
            "t_stage": str,
            "n_stage": str,
            "t_probs": ndarray,
            "n_probs": ndarray,
        }
    """

    prediction = tn_stager.predict(

        pet=pet,

        mask=mask,

        clinical_row=clinical_row,

    )

    return prediction


# =============================================================================
# Subtask 3 — Prognosis
# =============================================================================

def run_prognosis(
    survival_predictor,
    pet,
    mask,
    clinical_row,
    t_probs,
    n_probs,
):
    """
    Run recurrence-free survival prediction.

    Parameters
    ----------
    survival_predictor : SurvivalPredictor

    pet : ndarray
        Preprocessed PET volume.

    mask : ndarray
        Predicted segmentation mask in the
        224×224×224 preprocessed space.

    clinical_row : pandas.Series
        Clinical information for the current patient.

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

    prediction = survival_predictor.predict(

        pet=pet,

        mask=mask,

        clinical_row=clinical_row,

        t_probs=t_probs,

        n_probs=n_probs,

    )

    return prediction


# =============================================================================
# I/O utilities
# =============================================================================

def get_image_file(location):
    files = (
        glob(str(location / "*.mha"))
        + glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.nii.gz"))
    )
    if not files:
        raise FileNotFoundError(f"No image file found in {location}")
    return files[0]


def load_json(location):
    with open(location, "r") as f:
        return json.load(f)


def write_json(location, data):
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        json.dump(data, f, indent=2)


def write_segmentation(location, array, reference_path):
    location.mkdir(parents=True, exist_ok=True)
    reference = SimpleITK.ReadImage(reference_path)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    img = SimpleITK.GetImageFromArray(
        np.transpose(array, (2, 1, 0)).astype(np.uint8)
    )
    img.CopyInformation(reference)

    SimpleITK.WriteImage(
        img,
        str(location / "output.mha"),
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run())