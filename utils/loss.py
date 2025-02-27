import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
    
    def forward(self, predictions, targets):
        pred_boxes = predictions.boxes.data[:, :4]
        pred_scores = predictions.scores.data[:, 4]
        pred_labels = predictions.boxes.data[:, 5]
        
        target_boxes = targets['boxes']
        target_scores = targets['scores']
        target_labels = targets['labels']
        
        # No prediction scenario
        if len(pred_boxes) == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # Bounding box loss (Smooth L1)
        box_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
        
        # Confidence Score loss (Binary Cross Entropy)
        score_loss = F.binary_cross_entropy(pred_scores, target_scores)
        
        # Classification loss (Cross Entropy)
        class_loss = F.cross_entropy(pred_labels.long(), target_labels.long())
        
        total_loss = box_loss + score_loss + class_loss
        
        return total_loss