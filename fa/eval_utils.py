import numpy as np


def compute_iou(pred_label: np.ndarray, actual_label: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(pred_label == 1, actual_label == 1).sum(axis=-1)
    union = np.logical_or(pred_label == 1, actual_label == 1).sum(axis=-1)
    iou = np.mean(intersection / union)

    return iou
