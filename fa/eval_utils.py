import numpy as np
import torch
import wandb

from typing import Dict, Optional, Tuple


def compute_iou(pred_label: np.ndarray, actual_label: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(pred_label == 1, actual_label == 1).sum(axis=-1)
    union = np.logical_or(pred_label == 1, actual_label == 1).sum(axis=-1)
    iou = np.mean(intersection / union)

    return iou


def get_pc_viz(data: Dict, pred_thresh: torch.Tensor, n_samples: Optional[int] = None) -> Tuple:
    if n_samples is None:
        n_samples = min(len(data['scene']), 4)
    scene_viz = []
    gt_viz = []
    template_viz = []

    negative_color = [0, 255, 255]
    positive_color = [255, 255, 0]
    
    for i in range(n_samples):
        scene = data['scene'][i,:,:3].cpu().numpy()

        pred = pred_thresh[i].cpu().numpy()
        pred_arr = np.zeros((scene.shape[0], 3))
        pred_arr[pred == 0] = negative_color
        pred_arr[pred == 1] = positive_color

        gt = data['class_labels'][i].cpu().numpy()
        gt_arr = np.zeros((scene.shape[0], 3))
        gt_arr[gt == 0] = negative_color
        gt_arr[gt == 1] = positive_color

        template = data['template'][i,:,:3].cpu().numpy()
        template_arr = np.ones((template.shape[0], 3)) * 255

        scene_viz.append(wandb.Object3D(np.concatenate([scene, pred_arr], axis=1)))
        gt_viz.append(wandb.Object3D(np.concatenate([scene, gt_arr], axis=1)))
        template_viz.append(wandb.Object3D(np.concatenate([template, template_arr], axis=1)))
    
    return scene_viz, gt_viz, template_viz