import numpy as np
import torch
import wandb

def compute_iou(pred_label: np.ndarray, actual_label: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(pred_label == 1, actual_label == 1).sum(axis=-1)
    union = np.logical_or(pred_label == 1, actual_label == 1).sum(axis=-1)
    iou = np.mean(intersection / union)

    return iou

def get_pc_viz(data: dict, pred_thresh: torch.Tensor) -> tuple:
    query_viz = []
    gt_viz = []
    support_viz = []

    negative_color = [0, 255, 255]
    positive_color = [255, 255, 0]
    
    for i in range(4):
        query = data['query'][i,:,:3].cpu().numpy()

        pred = pred_thresh[i].cpu().numpy()
        pred_arr = np.zeros((query.shape[0], 3))
        pred_arr[pred == 0] = negative_color
        pred_arr[pred == 1] = positive_color

        gt = data['class_labels'][i].cpu().numpy()
        gt_arr = np.zeros((query.shape[0], 3))
        gt_arr[gt == 0] = negative_color
        gt_arr[gt == 1] = positive_color

        support = data['support'][i,:,:3].cpu().numpy()
        support_arr = np.ones((support.shape[0], 3)) * 255

        query_viz.append(wandb.Object3D(np.concatenate([query, pred_arr], axis=1)))
        gt_viz.append(wandb.Object3D(np.concatenate([query, gt_arr], axis=1)))
        support_viz.append(wandb.Object3D(np.concatenate([support, support_arr], axis=1)))
    
    return query_viz, gt_viz, support_viz