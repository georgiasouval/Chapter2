import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from torch.utils.data import DataLoader
from preprocess.preprocess_module import PreprocessModule
from ultralytics import YOLO
from ultralytics.models import RTDETR
from utils.loss import DetectionLoss
from dataloader import CustomDataset  # Import your dataset class

# ======================= ARGUMENT PARSER =======================
def get_args():
    parser = argparse.ArgumentParser(description="Train the preprocessing module for object detection.")
    
    # Dataset & Model Paths
    parser.add_argument("--train_images", type=str, required=True, help="Path to training images (folder).")
    parser.add_argument("--train_labels", type=str, required=True, help="Path to training labels (folder).")
    parser.add_argument("--val_images", type=str, required=True, help="Path to validation images (folder).")
    parser.add_argument("--val_labels", type=str, required=True, help="Path to validation labels (folder).")
    parser.add_argument("--yolo_weights", type=str, required=True, help="Path to YOLO pretrained weights.")
    parser.add_argument("--detr_weights", type=str, required=True, help="Path to RT-DETR pretrained weights.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'.")

    return parser.parse_args()

# ======================= OBJECT DETECTION PIPELINE =======================
class ObjectDetectionPipeline(nn.Module):
    def __init__(self, yolo_model, detr_model):
        super(ObjectDetectionPipeline, self).__init__()

        self.preprocess_module = PreprocessModule()  # Trainable

        # Frozen Object Detectors
        self.yolo = yolo_model.eval()
        self.detr = detr_model.eval()

        for param in self.yolo.parameters():
            param.requires_grad = False
        for param in self.detr.parameters():
            param.requires_grad = False

    def forward(self, raw_img):
        processed_img = self.preprocess_module(raw_img)
        yolo_output = self.yolo(processed_img)
        detr_output = self.detr(processed_img)
        return yolo_output, detr_output

# ======================= LOAD MODELS =======================
def load_yolo(weights_path, device):
    model_wrapper = YOLO(weights_path)
    model = model_wrapper.model.eval()
    return model.to(device)

def load_detr(weights_path, device):
    model_wrapper = RTDETR(weights_path)
    model = model_wrapper.model.eval()
    return model.to(device)

# ======================= TRAIN & VALIDATION LOOPS =======================
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (raw_img, targets) in enumerate(dataloader):
        raw_img = raw_img.to(device)
        yolo_target, detr_target = targets, targets  # Assume same targets for both

        optimizer.zero_grad()

        # Forward pass
        yolo_output, detr_output = model(raw_img)

        # Compute losses
        yolo_loss = criterion['yolo'](yolo_output, yolo_target)
        detr_loss = criterion['rtdetr'](detr_output, detr_target)
        loss = yolo_loss + detr_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:  # Log every 10 batches
            print(f"Train Step [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (raw_img, targets) in enumerate(dataloader):
            raw_img = raw_img.to(device)
            yolo_target, detr_target = targets, targets

            # Forward pass
            yolo_output, detr_output = model(raw_img)

            # Compute losses
            yolo_loss = criterion['yolo'](yolo_output, yolo_target)
            detr_loss = criterion['rtdetr'](detr_output, detr_target)
            loss = yolo_loss + detr_loss

            total_loss += loss.item()

    return total_loss / len(dataloader)

# ======================= MAIN FUNCTION =======================
if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("\n===== Loading Models =====")
    yolo_model = load_yolo(args.yolo_weights, device)
    detr_model = load_detr(args.detr_weights, device)
    model = ObjectDetectionPipeline(yolo_model, detr_model).to(device)

    print("\n===== Preparing Dataset & DataLoader =====")
    train_dataset = CustomDataset(img_paths=args.train_images, label_paths=args.train_labels, device=device, norm_value=65535.0)
    val_dataset = CustomDataset(img_paths=args.val_images, label_paths=args.val_labels, device=device, norm_value=65535.0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = {
        'yolo': DetectionLoss(),
        'rtdetr': DetectionLoss()
    }

    optimizer = optim.Adam(model.preprocess_module.parameters(), lr=args.learning_rate)

    print("\n===== Starting Training =====")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")

    print("\n===== Training Complete! =====")
