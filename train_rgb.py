#!/usr/bin/env python

import argparse
from typing import override
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocess.preprocess_module import PreprocessModule, BayerPacking
from ultralytics import YOLO, RTDETR
from utils.dataloaders import create_yolo_dataset, detection_collate_fn

import argparse
import math
import random
from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ultralytics import YOLO, RTDETR
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
# YOLO detection loss (v8DetectionLoss). If your version doesn't export it, you'll need to replicate from source:
from ultralytics.models.yolo.loss import v8DetectionLoss

# RT-DETR trainer for reference (to replicate bipartite matching logic):
from ultralytics.models.rtdetr.train import RTDETRTrainer

# 2) Your pipeline code
from preprocess.preprocess_module import PreprocessModule, BayerPacking

# 3) Example data utilities. 
# You can replicate mosaic/cutmix from YOLO or use simpler transformations
# We'll do a minimal approach here.
from ultralytics.data import build_yolo_dataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import LOGGER, RANK

# =============================================================================================

class ParallelTrainer(BaseTrainer):
    def __init__(self, opt):
        super(ParallelTrainer, self).__init__(opt)
        def get_model(self, cfg=None, weights=None, verbose=None):
            self.yolo_model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
            if weights and 'yolo' in weights:
                self.yolo_model.load(weights['yolo'])
            for name, param in self.yolo_model.named_parameters():
                if "backbone" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            self.rtdetr_model = RTDETRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
            if weights and 'rtdetr' in weights:
                self.rtdetr_model.load(weights['rtdetr'])

            




def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default='', help='Path to YOLO weights (e.g. yolo.pt)')
    parser.add_argument('--rtdetr-weights', type=str, default='', help='Path to RT-DETR weights (e.g. rtdetr.pth)')
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    args = parser.parse_args()

    trainer = ParallelTrainer(overrides=vars(args))
    trainer.train()

if __name__ == "__main__":  
    main(opt)






















class RawPipeline(nn.Module):
    def __init__(self, packing, preprocess, yolo, rtdetr):
        super(RawPipeline, self).__init__()
        # self.packing = packing
        self.preprocess = preprocess
        self.yolo = yolo
        self.rtdetr = rtdetr

    def forward(self, x):
        # x = self.packing(x)
        x = self.preprocess(x)
        yolo_out = self.yolo(x)
        rtdetr_out = self.rtdetr(x)
        return yolo_out, rtdetr_out
    

def load_and_freeze_yolo(weights):
    yolo_model = YOLO(weights)
    for name, param in yolo_model.model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return yolo_model.model


def load_and_freeze_rtdetr(weights):
    rtdetr_model = RTDETR(weights)
    for name, param in rtdetr_model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return rtdetr_model


# ================== Loss functions ================== #
def yolo_loss(yolo_model, yolo_out, yolo_target):
    pass

def rtdetr_loss(rtdetr_model, rtdetr_out, rtdetr_target):
    pass  


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading YOLO model...")
    yolo_model = load_and_freeze_yolo(opt.yolo_weights).to(device)

    print("Loading RT-DETR model...")
    rtdetr_model = load_and_freeze_rtdetr(opt.rtdetr_weights).to(device)

    # print("Creating pipeline...")
    pipeline = RawPipeline(BayerPacking(), PreprocessModule(), yolo_model, rtdetr_model).to(device)

    # print("Loading data...")
    train_dataset = create_yolo_dataset(
        yaml_path=opt.yolo_data,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        mode="train",
        transform=None,
        device=device,
        img_shape=opt.img_shape,
        norm_value=opt.norm_value
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=opt.num_workers,
        pin_memory=True
    )


    optimiser = optim.Adam(filter(lambda p: p.requires_grad, pipeline.parameters()), lr=opt.lr)

    # print(f"[INFO] Starting traing for {opt.epochs} epochs...")

    for epoch in range(opt.epochs):
        pipeline.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            yolo_out, rtdetr_out = pipeline(img)

            yolo_loss(yolo_model, yolo_out, target)
            rtdetr_loss(rtdetr_model, rtdetr_out, target)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if i % opt.print_freq == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")



