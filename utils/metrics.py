#!/usr/bin/env python

import torch
import numpy as np
from torchvision.ops import box_iou

class DetectionMetrics:
    """
    Tracks precision, recall, and a naive mAP@IoU=0.5 across an entire dataset.
    For a more complete solution, integrate with COCO metrics or Ultralytics' ap_per_class.
    """
    def __init__(self, iou_threshold=0.5, num_classes=5):
        """
        Args:
            iou_threshold (float): IoU threshold to consider a detection correct.
            num_classes (int): number of classes in your dataset.
        """
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        # We store stats per class: tp, fp, fn
        self.stats = {
            c: {"tp": 0, "fp": 0, "fn": 0} for c in range(num_classes)
        }

    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """
        Update stats with a single batch item (one image).
        Args:
            pred_boxes (Tensor): shape [N,4], in (x1,y1,x2,y2)
            pred_scores (Tensor): shape [N]
            pred_labels (Tensor): shape [N]
            gt_boxes (Tensor): shape [M,4]
            gt_labels (Tensor): shape [M]
        """
        if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
            return  # no predictions, no ground truth → nothing to update

        if pred_boxes.numel() == 0 and gt_boxes.numel() != 0:
            # all GT are missed
            for lbl in gt_labels:
                self.stats[lbl.item()]["fn"] += 1
            return

        if gt_boxes.numel() == 0 and pred_boxes.numel() != 0:
            # all predictions are false positives
            for lbl in pred_labels:
                self.stats[lbl.item()]["fp"] += 1
            return

        # Compute IoU between all predictions & GT
        ious = box_iou(pred_boxes, gt_boxes)  # shape [N, M]
        pred_matched = [False]*len(pred_boxes)
        gt_matched = [False]*len(gt_boxes)

        # For each prediction, find best IoU
        for i in range(len(pred_boxes)):
            max_iou, max_j = torch.max(ious[i], dim=0)
            if max_iou >= self.iou_threshold and not gt_matched[max_j]:
                # check if label matches
                if pred_labels[i] == gt_labels[max_j]:
                    self.stats[pred_labels[i].item()]["tp"] += 1
                else:
                    self.stats[pred_labels[i].item()]["fp"] += 1
                pred_matched[i] = True
                gt_matched[max_j] = True

        # Unmatched predictions → false positives
        for i in range(len(pred_boxes)):
            if not pred_matched[i]:
                self.stats[pred_labels[i].item()]["fp"] += 1

        # Unmatched GT → false negatives
        for j in range(len(gt_boxes)):
            if not gt_matched[j]:
                self.stats[gt_labels[j].item()]["fn"] += 1

    def compute(self):
        """
        Compute precision, recall, naive F1, and a naive 'mAP' across all classes.
        Returns a dict with { 'precision': float, 'recall': float, 'f1': float, 'mAP': float }.
        """
        precisions = []
        recalls = []
        for c in range(self.num_classes):
            tp = self.stats[c]["tp"]
            fp = self.stats[c]["fp"]
            fn = self.stats[c]["fn"]

            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            precisions.append(prec)
            recalls.append(rec)

        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-6)

        # A naive "mAP" at IoU=0.5 (not a true area-under-curve integration).
        # For a real mAP, you'd integrate over multiple confidence thresholds or use pycocotools.
        naive_map = mean_precision * mean_recall

        return {
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": f1,
            "mAP": naive_map
        }
