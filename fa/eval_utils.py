import numpy as np
import wandb
import torch

def compute_iou(pred_label: np.ndarray, actual_label: np.ndarray) -> np.ndarray:
    num_classes = int(max(pred_label.max(), actual_label.max()))
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)

    for i in range(num_classes):
        I_all[i] = np.logical_and(pred_label == i, actual_label == i).sum()
        U_all[i] = np.logical_or(pred_label == 1, actual_label == 1).sum()

    return np.mean(I_all / U_all)

def get_pc_viz(data: dict, pred_max: np.ndarray) -> tuple:
    query_viz = []
    gt_viz = []
    
    for i in range(4):
        query = data['query'][i,:,:3].cpu().numpy()
        pred = np.expand_dims(pred_max[i], 1)
        gt = data['class_labels'][i].unsqueeze(1).cpu().numpy()

        query_viz.append(wandb.Object3D(np.concatenate([query, pred], axis=1)))
        gt_viz.append(wandb.Object3D(np.concatenate([query, gt], axis=1)))
    
    return query_viz, gt_viz