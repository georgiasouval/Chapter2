import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocess.preprocess_module import PreprocessModule  # Adjust if needed
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
def load_yolo(weights_path='/networkhome/WMGDS/souval_g/Multi-model-RAW-Network/yolo11x.pt', 
              data_path='/networkhome/WMGDS/souval_g/datasets/dataset48/day/dataset.yaml', 
              device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the YOLO wrapper and update the dataset path
    model_wrapper = YOLO(weights_path)
    model_wrapper.overrides['data'] = data_path
    # Extract the underlying PyTorch model and set it to evaluation mode
    model = model_wrapper.model
    model.eval()
    return model.to(device)


def load_detr(weights_path='/networkhome/WMGDS/souval_g/Multi-model-RAW-Network/rtdetr-l.pt', 
              data_path='/networkhome/WMGDS/souval_g/datasets/dataset48/day/dataset.yaml', 
              device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the RTDETR wrapper and update the dataset path
    model_wrapper = RTDETR(weights_path)
    model_wrapper.overrides['data'] = data_path
    # Extract the underlying PyTorch model and set it to evaluation mode
    model = model_wrapper.model
    model.eval()
    return model.to(device)


# ======================= MAIN =======================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load underlying torch models for evaluation
    yolo_model = load_yolo().to(device)
    detr_model = load_detr().to(device)

    # Initialize pipeline with the loaded models
    model = ObjectDetectionPipeline(yolo_model, detr_model).to(device)

    model.eval()
    sample_input = torch.randn(1, 3, 640, 640).to(device)

    yolo_output, detr_output = model(sample_input)

    print(yolo_output, "YOLO Output")
    print(detr_output, "RT-DETR Output")