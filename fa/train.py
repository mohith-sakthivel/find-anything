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
from fa.fusion import SimpleAggregator
from fa.predictor import DGCNNPredHead
from fa.model import FindAnything
from eval_utils import compute_iou


config = AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.epochs = 45
config.save_dir = ""
config.batch_size = 16
config.eval_freq = 1  # Epochs after which to eval model
config.lr = 1e-3
config.eval_samples = 2000
config.log_freq = 20  # Iters to log after
config.save_freq = 5  # Epochs to save after. Set None to not save.
config.save_path = "checkpoints"
config.use_pretrained_dgcnn = 'scannet'
config.use_checkpoint = None


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader) -> Dict:
    model.eval()

    true_labels = []
    pred_labels = []
    loss = []

    for data in tqdm.tqdm(iterable=data_loader, desc=f"Test", total=len(data_loader)):

        data = data.to(config.device)

        pred = model(
            scene_pointcloud=data['query'],
            template_pointcloud=data['support'],
        )
        loss.append(F.binary_cross_entropy_with_logits(pred, data['support_mask'], reduction='mean').cpu().item())

        pred_labels.append(pred.argmax(dim=-1).cpu().numpy())
        true_labels.append(data['support_mask'].cpu().numpy())
    
    pred_labels = np.concatenate(pred_labels, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    metrics = {
        "loss": np.mean(loss),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "balanced_accuracy": balanced_accuracy_score(true_labels, pred_labels),
        "iou": compute_iou(pred_labels, true_labels)
    }

    return metrics


def train_model(config: Dict) -> None:

    seed_everything(config.seed)
    if config.use_checkpoint is None:
        run_id = config.expname + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    else:
        run_id = config.use_checkpoint

    wandb_run = wandb.init(
        project="find-anything",
        config=config,
        id=run_id,
        resume=None if config.use_checkpoint is None else "must"
    )

    train_dataloader = None
    test_dataloader = None

    feat_extractor = DGCNNSeg()
    if config.use_pretrained_dgcnn == "scannet":
        feat_extractor.load_state_dict(torch.load("pretrained_weights/scannet.pth"))
    elif config.use_pretrained_dgcnn == "s3dis":
        feat_extractor.load_state_dict(torch.load("pretrained_weights/s3dis_model_6.pth"))

    feat_agg = SimpleAggregator(
        scene_feat_dim=feat_extractor.feat_dim,
        template_feat_dim=feat_extractor.feat_dim
    )
    pred_head = DGCNNPredHead(in_dim=feat_agg.out_dim)

    model = FindAnything(
        scene_feat_extractor=feat_extractor,
        fusion_module=feat_agg,
        pred_head=pred_head,
        template_feat_extractor=None,
        use_common_feat_extractor=True
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=20,
        gamma=0.5
    )
    start_epoch = 0
    best_score = -1

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)

        model = checkpoint["model"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]

    loss_fn = nn.BCEWithLogitsLoss()
    model.to(config.device)

    for epoch in range(start_epoch, config.epochs):
        model.train()

        wandb.log({"train": {"epoch": epoch, "lr": scheduler.get_lr()}}, commit=False)

        for iter, data in tqdm.tqdm(
            iterable=enumerate(train_dataloader),
            desc=f"Epoch {epoch:04d}/{config.epochs} train",
            total=len(train_dataloader)
        ):
            optimizer.zero_grad()

            data = data.to(config.device)

            pred = model(
                scene_pointcloud=data['query'],
                template_pointcloud=data['support'],
            )

            loss = loss_fn(pred, data['support_mask'])

            loss.backward()
            optimizer.step()

            if ((iter + 1) % config.log_freq == 0):
                wandb.log({
                    "train": {
                        "loss": loss.detach().cpu().item()
                    },
                })
        
        scheduler.step() 

        best_epoch = False
        if ((epoch + 1) % config.eval_freq == 0):
            metrics = evaluate_model(model, test_dataloader)

            wandb.log({"test": metrics}, commit=False)    

        if ((epoch + 1) % config.save_freq == 0 or best_epoch):
            save_path = Path(config.save_path) / run_id / "best_epoch" if best_epoch else f"epoch_{epoch + 1}" 
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
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--exp_name", default="default", type=int)
    args = parser.parse_args()

    for k, v in args.items():
        config[k] = v

    train_model(config)
