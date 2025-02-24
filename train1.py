import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocess.preprocess_module import *
from ultralytics import YOLO
from ultralytics.models import RTDETR


# ======================= MAIN PIPELINE =======================
class ObjectDetectionPipeline(nn.Module):
    def __init__(self, yolo_model, detr_model):
        super(ObjectDetectionPipeline, self).__init__()
        # self.packing_layer = BayerPacking()
        self.preprocess_module = PreprocessModule()

        # Object detectors (frozen)
        self.yolo = yolo_model.eval()
        self.detr = detr_model.eval()

        for param in self.yolo.parameters():
            param.requires_grad = False
        for param in self.detr.parameters():
            param.requires_grad = False

    def forward(self, raw_img):
        # packed_img = self.packing_layer(raw_img)
        processed_img = self.preprocess_module(raw_img)

        yolo_output = self.yolo(processed_img)
        detr_output = self.detr(processed_img)

        return yolo_output, detr_output


# ======================= LOAD MODELS =======================
def load_yolo(weights_path='/networkhome/WMGDS/souval_g/Multi-model-RAW-Network/yolo11x.pt', data_path='/networkhome/WMGDS/souval_g/datasets/dataset48/day/dataset.yaml', device="cuda" if torch.cuda.is_available() else "cpu"):
    model = YOLO(weights_path)
    model.overrides['data'] = data_path
    return model.to(device)


def load_detr(weights_path='/networkhome/WMGDS/souval_g/Multi-model-RAW-Network/rtdetr-l.pt', data_path='/networkhome/WMGDS/souval_g/datasets/dataset48/day/dataset.yaml', device="cuda" if torch.cuda.is_available() else "cpu"):
    model = RTDETR(weights_path)
    model.overrides['data'] = data_path  #
    return model.to(device)


# ======================= TRAINING STEP =======================
def training_step(model, batch, criterion, optimizer):
    raw_img, targets = batch
    optimizer.zero_grad()

    yolo_output, detr_output = model(raw_img)

    yolo_loss = criterion['yolo'](yolo_output, targets)
    detr_loss = criterion['detr'](detr_output, targets)

    total_loss = yolo_loss + detr_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# ======================= MAIN TRAINING LOOP =======================
def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            loss = training_step(model, batch, criterion, optimizer)
            epoch_loss += loss

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    yolo_model = load_yolo().to(device)
    detr_model = load_detr().to(device)

    # Initialize pipeline
    model = ObjectDetectionPipeline(yolo_model, detr_model).to(device)

    # Run in evaluation mode to ensure models are properly called
    model.eval()
    sample_input = torch.randn(1, 3, 300, 300).to(device)  # Example input image
    yolo_output, detr_output = model(sample_input)

    print("YOLO Output:", yolo_output)
    print("RT-DETR Output:", detr_output)
