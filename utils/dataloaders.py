import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
BIT8 = 2**8
BIT16 = 2**16
BIT24 = 2**24

def load_yolo_targets(label_path, img_width, img_height, device='cuda'):
    boxes = []
    labels = []

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id, centre_x, centre_y, width, height = map(float, line.split())
            x1 = (centre_x - width / 2) * img_width
            y1 = (centre_y - height / 2) * img_height
            x2 = (centre_x + width / 2) * img_width
            y2 = (centre_y + height / 2) * img_height
            boxes.append([x1, y1, x2, y2])
            labels.append(int(class_id))

    # Handle case where no objects exist
    if len(boxes) == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
            'scores': torch.zeros((0,), dtype=torch.float32, device=device),
            'labels': torch.zeros((0,), dtype=torch.long, device=device)
        }

    targets = {
        'boxes': torch.tensor(boxes, dtype=torch.float32, device=device),
        'scores': torch.ones(len(boxes), dtype=torch.float32, device=device),  # Default confidence = 1.0
        'labels': torch.tensor(labels, dtype=torch.long, device=device)
    }

    return targets

def read_raw_24b(file_path, img_shape, read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data.astype(np.float32)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    return raw_data 


def load_image(img_path, device='cuda', norm_value=255.0, img_shape=None):
    ext = os.path.splitext(img_path)[-1].lower()

    if ext in ['.jpg', '.jpeg', '.png']:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / norm_value

    elif ext in ['.npy']:
        img = np.load(img_path).astype(np.float32) / norm_value

    elif ext in ['.raw']:
        if img_shape is None:
            raise ValueError("Image shape must be provided for RAW images.")
        img = read_raw_24b(img_path, img_shape)

    else:
        raise ValueError(f'Unsupported image format: {ext}')    

    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(device)  # (H, W, C) → (C, H, W)
    return img


class CustomDataset(Dataset):
    def __init__(self, img_paths, label_paths, transform=None, device='cuda', img_shape=(1, 1856, 2880), norm_value=255.0):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform
        self.device = device
        self.img_shape = img_shape
        self.norm_value = norm_value

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx], self.device, norm_value=self.norm_value, img_shape=self.img_shape)
        img_height, img_width = img.shape[1:]

        targets = load_yolo_targets(self.label_paths[idx], img_width, img_height, self.device)

        if self.transform:
            img = self.transform(img)

        return img, targets

transform = transforms.Compose([
    transforms.Resize((640, 640)),   # Resize to model input size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

dataset = CustomDataset(
    img_paths=["image1.jpg", "image2.npy", "image3.raw"],
    label_paths=["image1.txt", "image2.txt", "image3.txt"],
    transform=transform,
    norm_value=BIT16
)

img, targets = dataset[0]  # Load first image and its labels
print(img.shape, targets)
