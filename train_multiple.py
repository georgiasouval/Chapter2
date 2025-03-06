
import torch
from Raw_Yolo.models import yolo
from ultralytics.nn.tasks import RTDETRDetectionModel, DetectionModel
device = 'cuda'


rtdetr_cfg = "/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/Chapter2/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml"
rtdetr_weights = "/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/Chapter2/rtdetr-l.pt"

yolo_cfg = "/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/Chapter2/ultralytics/cfg/models/11/yolo11.yaml"
yolo_weights = "/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/Chapter2/yolo11x.pt"


rtdetr = RTDETRDetectionModel(rtdetr_cfg, nc=5)

model = DetectionModel(yolo_cfg, nc=5)

rtdetr.eval()
model.eval()

sample_input = torch.randn(4, 3, 300, 300).to(device)  # Example input image
output1 = rtdetr(sample_input)
output2 = model(sample_input)
print("RT-DETR Output shape:", output1.shape)
print("yolo Output shape:", output2.shape)