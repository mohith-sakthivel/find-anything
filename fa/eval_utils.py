import numpy as np


def compute_iou(pred_label: np.ndarray, actual_label: np.ndarray) -> np.ndarray:
    num_classes = int(max(pred_label.max(), actual_label.max()))
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)

    for i in range(num_classes):
        I_all[i] = np.logical_and(pred_label == i, actual_label == i).sum()
        U_all[i] = np.logical_or(pred_label == 1, actual_label == 1).sum()

    return np.mean(I_all / U_all)
