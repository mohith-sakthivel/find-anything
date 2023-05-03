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

from fa.model import FindAnything
from fa.dataset import FindAnythingDataset
from fa import FEATURE_EXTRACTORS, AGGREGATORS, PREDICTORS
from fa.utils import AttrDict, save_checkpoint, seed_everything
from fa.eval_utils import compute_iou, get_pc_viz


config = AttrDict()

# Setup
config.seed = 0
config.device = "cuda"
config.num_workers = 12
config.use_normals_for_scene = False
config.use_normals_for_template = True

# Model
config.feat_extractor = 'dgcnn'
config.aggregator = 'simple'
config.predictor = 'dgcnn'
config.aggr_feat_dim = 128

config.feat_extractor_args = AttrDict()

config.aggregator_args = AttrDict()

config.predictor_args = AttrDict()

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
config.num_scene_points = 4096
config.num_template_points = 1024
config.pred_threshold = 0.5

# Utils
config.log_freq = 20  # Iters to log after
config.save_freq = 2  # Epochs to save after. Set None to not save.
config.save_path = "checkpoints"
config.checkpoint = None
config.wandb = 'online'


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, config: Dict, generate_visuals: bool = True) -> Dict:
    model.eval()

    loss = []
    first_pass = True
    pred_list = []

    for data in tqdm.tqdm(iterable=data_loader, desc=f"Test", total=len(data_loader)):

        data = {k: v.to(config.device) for k, v in data.items()}

        cosine_sim = nn.CosineSimilarity()

        scene_feature = model.scene_feat_extractor(data['scene'])
        template_feature = model.template_feat_extractor(data['template'])
        
        max_template = torch.amax(template_feature, -1, keepdim=True)
        pred = cosine_sim(scene_feature, max_template)
        pred_list.append(pred)

    
    pred_labels = []
    true_labels = []
    for i in np.linspace(0, 1, 101):
        pred_labels.clear()
        true_labels.clear()
        for pred in pred_list:
            pred_thresh = (pred > i).to(torch.float32)
            pred_labels.append(pred_thresh.cpu().numpy())
            true_labels.append(data['class_labels'].cpu().numpy())
        pred_labels_out = np.concatenate(pred_labels, axis=0)
        true_labels_out = np.concatenate(true_labels, axis=0)
        print(f"Balanced accuracy at {i}:", balanced_accuracy_score(true_labels_out.reshape(-1), pred_labels_out.reshape(-1)))
        print(f"IoU at {i}:", compute_iou(pred_labels_out, true_labels_out))


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
        num_scene_points=config.num_scene_points,
        num_template_points=config.num_template_points,
        dataset_size=config.train_dataset_size,
        use_normals_for_scene=config.use_normals_for_scene,
        use_normals_for_template=config.use_normals_for_template
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_dataset = FindAnythingDataset(
        split="test",
        num_scene_points=config.num_scene_points,
        num_template_points=config.num_template_points,
        dataset_size=config.test_dataset_size,
        use_normals_for_scene=config.use_normals_for_scene,
        use_normals_for_template=config.use_normals_for_template
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    scene_feat_extractor = FEATURE_EXTRACTORS[config.feat_extractor](
        pc_dim=train_dataset.scene_pc_dim,
        **config.feat_extractor_args)
    template_feat_extractor = FEATURE_EXTRACTORS[config.feat_extractor](
        pc_dim=train_dataset.template_pc_dim,
        **config.feat_extractor_args
    )

    feat_agg = AGGREGATORS[config.aggregator](
        scene_feat_dim=scene_feat_extractor.feat_dim,
        template_feat_dim=scene_feat_extractor.feat_dim,
        out_dim=config.aggr_feat_dim,
        **config.aggregator_args
    )

    pred_head = PREDICTORS[config.predictor](in_dim=feat_agg.out_dim, **config.predictor_args)

    model = FindAnything(
        scene_feat_extractor=scene_feat_extractor,
        template_feat_extractor=template_feat_extractor,
        fusion_module=feat_agg,
        pred_head=pred_head,
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

        wandb.log({"train/epoch": epoch, "train/lr": scheduler.get_last_lr()[0]}, commit=False)

        for iter, data in tqdm.tqdm(
            iterable=enumerate(train_dataloader),
            desc=f"Epoch {epoch:04d}/{config.epochs}",
            total=len(train_dataloader)
        ):
            optimizer.zero_grad()

            data = {k: v.to(config.device) for k, v in data.items()}

            pred = model(
                scene_pointcloud=data['scene'],
                template_pointcloud=data['template'],
            )

            loss = loss_fn(pred, data['class_labels'])

            loss.backward()
            optimizer.step()

            if ((iter + 1) % config.log_freq == 0):
                tqdm.tqdm.write(f"Loss: {loss.detach().cpu().item():.6f}")
                wandb.log({"train/loss": loss.detach().cpu().item()})
        
        scheduler.step() 

        best_epoch = False
        if ((epoch + 1) % config.eval_freq == 0):
            metrics = evaluate_model(model, test_dataloader, config)

            if (metrics["iou"] > best_iou):
                best_iou = metrics["iou"]
                best_epoch = True

            tqdm.tqdm.write(f"Epoch: {epoch + 1:04d} \t iou: {metrics['iou']:.4f} \t loss: {metrics['loss']:0.6f}")

            metrics = {f"test/{k}": v for k, v in metrics.items()}
            wandb.log(metrics, commit=False)  

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
