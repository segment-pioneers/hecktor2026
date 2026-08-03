"""
segmentation.py

5-fold ensemble segmentation module for HECKTOR 2026.
"""

import torch
import numpy as np

import networks

from preprocessing import (
    reconstruct_to_original_space,
    save_mha,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD ONE MODEL
# ============================================================

def load_model(model_path):
    """
    Load one trained segmentation model.
    """

    model = networks.AdaMSS_Seg()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# SEGMENTATION ENSEMBLE
# ============================================================

class Segmentor:

    def __init__(self,
                 model_paths):

        self.models = []

        for path in model_paths:

            self.models.append(
                load_model(path)
            )

    @torch.no_grad()
    def predict(self,
                data,
                output_mha=None):
        """
        Predict segmentation from precomputed preprocess_patient output.
        """

        ct = torch.from_numpy(
            data["ct"]
        ).unsqueeze(0).unsqueeze(0)

        pet = torch.from_numpy(
            data["pet"]
        ).unsqueeze(0).unsqueeze(0)

        ct = ct.to(
            DEVICE,
            non_blocking=True
        )

        pet = pet.to(
            DEVICE,
            non_blocking=True
        )

        tumor_sum = None
        node_sum = None

        for model in self.models:

            tumor_pred, node_pred = model(
                pet,
                ct
            )

            if tumor_sum is None:

                tumor_sum = tumor_pred

                node_sum = node_pred

            else:

                tumor_sum += tumor_pred

                node_sum += node_pred

        tumor_prob = tumor_sum / len(self.models)
        node_prob = node_sum / len(self.models)

        tumor_prob = (
            tumor_prob
            .squeeze()
            .cpu()
            .numpy()
        )

        node_prob = (
            node_prob
            .squeeze()
            .cpu()
            .numpy()
        )

        prediction = np.zeros(
            tumor_prob.shape,
            dtype=np.uint8
        )

        foreground = np.maximum(
            tumor_prob,
            node_prob
        ) >= 0.5

        tumor_voxels = (
            foreground &
            (tumor_prob >= node_prob)
        )

        node_voxels = (
            foreground &
            (node_prob > tumor_prob)
        )

        prediction[tumor_voxels] = 1
        prediction[node_voxels] = 2

        final_mask = reconstruct_to_original_space(
            prediction,
            data
        )

        if output_mha is not None:
            save_mha(
                final_mask,
                data["reference_ct_path"],
                output_mha
            )

        return {

            "mask": final_mask,

            "prediction": prediction,

            "pet": data["pet"],

            "ct": data["ct"],

            "tumor_probability": tumor_prob,

            "node_probability": node_prob
        }

    @torch.no_grad()
    def predict_preprocessed(self,
                             ct,
                             pet):
        """
        Predict directly from preprocessed CT/PET arrays.
        """

        ct = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0)
        pet = torch.from_numpy(pet).unsqueeze(0).unsqueeze(0)

        ct = ct.to(DEVICE)
        pet = pet.to(DEVICE)

        tumor_sum = None
        node_sum = None

        for model in self.models:

            tumor_pred, node_pred = model(
                pet,
                ct
            )

            if tumor_sum is None:

                tumor_sum = tumor_pred

                node_sum = node_pred

            else:

                tumor_sum += tumor_pred

                node_sum += node_pred

        tumor_prob = (
            tumor_sum / len(self.models)
        )

        node_prob = (
            node_sum / len(self.models)
        )

        return (
            tumor_prob.squeeze().cpu().numpy(),
            node_prob.squeeze().cpu().numpy()
        )
