import tqdm
import datetime
import argparse
import numpy as np
from typing import Dict
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fa.utils import AttrDict, save_checkpoint, seed_everything
from fa.dgcnn import DGCNNSeg
from fa.fusion import DynamicConvolution
from fa.predictor import DGCNNPredHead
from fa.model import FindAnything
from fa.dataset import FindAnythingDataset
from fa.eval_utils import compute_iou, get_pc_viz


config = AttrDict()

# Setup
config.seed = 0
config.device = "cuda"
config.num_workers = 12

# Model
# config.aggr_feat_size = 128

# Train
config.epochs = 250
config.batch_size = 12
config.lr = 5e-4
config.train_dataset_size = 2e3
config.gamma = 0.5
config.step_size = 50
config.pos_sample_weight = 2

# Test
config.eval_freq = 1  # Epochs after which to eval model
config.test_dataset_size = 200

# Problem Framework
config.num_query_points = 4096
config.num_support_points = 1024
config.pred_threshold = 0.5

# Utils
config.log_freq = 20  # Iters to log after
config.save_freq = 2  # Epochs to save after. Set None to not save.
config.save_path = "checkpoints"
config.checkpoint = None
config.wandb = 'online'


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, config: Dict) -> Dict:
    model.eval()

    true_labels = []
    pred_labels = []
    loss = []
    first_pass = True

    for data in tqdm.tqdm(iterable=data_loader, desc=f"Test", total=len(data_loader)):

        data = {k: v.to(config.device) for k, v in data.items()}

        pred = model(
            scene_pointcloud=data['query'],
            template_pointcloud=data['support'],
        )
        loss.append(F.binary_cross_entropy_with_logits(pred, data['class_labels'], 
                                                       reduction='mean', 
                                                       pos_weight=torch.Tensor([config.pos_sample_weight]).to(config.device)).cpu())

        pred_thresh = (pred > config.pred_threshold).to(torch.float32)
        pred_labels.append(pred_thresh.cpu().numpy())
        true_labels.append(data['class_labels'].cpu().numpy())

        if first_pass:
            query_viz, gt_viz, support_viz = get_pc_viz(data, pred_thresh)
            first_pass = False
            
    pred_labels = np.concatenate(pred_labels, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    metrics = {
        "loss": np.mean(loss),
        "accuracy": accuracy_score(true_labels.reshape(-1), pred_labels.reshape(-1)),
        "balanced_accuracy": balanced_accuracy_score(true_labels.reshape(-1), pred_labels.reshape(-1)),
        "iou": compute_iou(pred_labels, true_labels),
        "query point cloud": query_viz,
        "gt_point_cloud": gt_viz,
        "support point cloud": support_viz
    }

    return metrics


def train_model(config: Dict) -> None:

    seed_everything(config.seed)
    if config.checkpoint is None:
        run_id = config.exp_name + '-' + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    else:
        run_id = config.checkpoint

    wandb_run = wandb.init(
        project="find-anything",
        config=config,
        id=run_id,
        resume=None if config.checkpoint is None else "must",
        mode=config.wandb
    )

    train_dataset = FindAnythingDataset(
        split="train",
        num_query_points=config.num_query_points,
        num_support_points=config.num_support_points,
        dataset_size=config.train_dataset_size,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_dataset = FindAnythingDataset(
        split="test",
        num_query_points=config.num_query_points,
        num_support_points=config.num_support_points,
        dataset_size=config.test_dataset_size
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    feat_extractor = DGCNNSeg()
    template_feat_extractor = DGCNNSeg()

    feat_agg = DynamicConvolution(
        scene_feat_dim=feat_extractor.feat_dim,
        template_feat_dim=template_feat_extractor.feat_dim
    )

    model = FindAnything(
        scene_feat_extractor=feat_extractor,
        template_feat_extractor=template_feat_extractor,
        fusion_module=feat_agg,
        pred_head=nn.Identity()
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config.step_size,
        gamma=config.gamma
    )
    start_epoch = 0
    best_iou = -1

    if config.checkpoint is not None:
        checkpoint = torch.load(Path(config.save_path) / config.checkpoint / "last_epoch")

        model = checkpoint["model"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_sample_weight]).to(config.device))
    model.to(config.device)

    for epoch in range(start_epoch, config.epochs):
        model.train()

        for iter, data in tqdm.tqdm(
            iterable=enumerate(train_dataloader),
            desc=f"Epoch {epoch:04d}/{config.epochs}",
            total=len(train_dataloader)
        ):
            optimizer.zero_grad()

            data = {k: v.to(config.device) for k, v in data.items()}

            pred = model(
                scene_pointcloud=data['query'],
                template_pointcloud=data['support'],
            )

            loss = loss_fn(pred, data['class_labels'])

            loss.backward()
            optimizer.step()

            if ((iter + 1) % config.log_freq == 0):
                tqdm.tqdm.write(f"Loss: {loss.detach().cpu().item():.6f}")
                wandb.log({"train": {
                    "loss": loss.detach().cpu().item(),
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0]
                }})
        
        scheduler.step() 

        best_epoch = False
        if ((epoch + 1) % config.eval_freq == 0):
            metrics = evaluate_model(model, test_dataloader, config)

            if (metrics["iou"] > best_iou):
                best_iou = metrics["iou"]
                best_epoch = True

            wandb.log({"test": metrics}, commit=False) 
            tqdm.tqdm.write(f"Epoch: {epoch + 1:04d} \t iou: {metrics['iou']:.4f} \t loss: {metrics['loss']:0.6f}")   

        if ((epoch + 1) % config.save_freq == 0 or best_epoch):
            save_path = Path(config.save_path) / run_id / ("best_epoch" if best_epoch else "last_epoch")
            save_checkpoint(
                model=model,
                path=save_path,
                save_only_state_dict=False,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1
            )
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--exp_name", default="default", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    args = parser.parse_args()

    for k, v in vars(args).items():
        if k in config and v is None:
            continue
        config[k] = v

    train_model(config)
