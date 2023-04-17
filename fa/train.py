import tqdm
import random
import argparse
import numpy as np
from typing import Dict

import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from fa.utils import AttrDict, save_checkpoint, seed_everything
from fa.dgcnn import DGCNNSeg
from fa.fusion import SimpleAggregator
from fa.predictor import DGCNNPredHead
from fa.model import FindAnything


config = AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.epochs = 45
config.save_dir = ""
config.batch_size = 16
config.eval_freq = 2000
config.lr = 1e-3
config.eval_samples = 2000
config.log_freq = 20  # Iters to log after
config.save_freq = 5  # Epochs to save after. Set None to not save.


def train_model(config: Dict) -> None:

    train_dataloader = None
    test_dataloader = None

    feat_extractor = DGCNNSeg()
    feat_agg = SimpleAggregator(in_dim=feat_extractor.feat_dim)
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

        if ((epoch + 1) % config.save_freq == 0):
            save_checkpoint(
                model=model,
                path=config.save_path,
                save_only_state_dict=False,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--exp_name", default="default", type=int)
    args = parser.parse_args()

    for k, v in args.items():
        config[k] = v

    wandb_run = wandb.init(
        project="find-anything",
        config=config
    )


    seed_everything(config.seed)
    train_model(config)
