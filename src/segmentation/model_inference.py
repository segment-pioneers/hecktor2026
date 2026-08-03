import numpy as np
import torch

from .networks import AdaMSS_Seg


def load_segmentation_model(
    model_path,
    device
):

    model = AdaMSS_Seg()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def predict_masks(
    model,
    pet,
    ct,
    device,
    threshold=0.5
):

    pet_t = torch.from_numpy(
        pet[np.newaxis, np.newaxis]
    ).float().to(device)

    ct_t = torch.from_numpy(
        ct[np.newaxis, np.newaxis]
    ).float().to(device)

    with torch.no_grad():

        pred_tumor, pred_node = model(
            pet_t,
            ct_t
        )

    pred_tumor = (
        pred_tumor
        .cpu()
        .numpy()
        .squeeze()
    )

    pred_node = (
        pred_node
        .cpu()
        .numpy()
        .squeeze()
    )

    pred_tumor = (pred_tumor > threshold).astype(
        np.uint8
    )

    pred_node = (pred_node > threshold).astype(
        np.uint8
    )

    return pred_tumor, pred_node
