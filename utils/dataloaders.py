import os
import glob
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BIT8 = 2**8
BIT16 = 2**16
BIT24 = 2**24

# ==============================
# 1. Parsing YOLO labels
# ==============================
def load_yolo_targets(label_path, img_width, img_height, device='cuda'):
    """Loads YOLO-format .txt annotations and converts them to absolute coords."""
    boxes = []
    labels = []
    try:
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
    except FileNotFoundError:
        # If no label file, treat as zero objects
        pass

    if len(boxes) == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
            'scores': torch.zeros((0,), dtype=torch.float32, device=device),
            'labels': torch.zeros((0,), dtype=torch.long, device=device)
        }

    targets = {
        'boxes': torch.tensor(boxes, dtype=torch.float32, device=device),
        'scores': torch.ones(len(boxes), dtype=torch.float32, device=device),
        'labels': torch.tensor(labels, dtype=torch.long, device=device)
    }
    return targets

# ==============================
# 2. RAW reading helper
# ==============================
def read_raw_24b(file_path, img_shape, read_type=np.uint8):
    """
    Example for reading a custom 24-bit RAW file, 
    combining 3 bytes per pixel into a single value.
    Adjust as needed for your RAW format.
    """
    raw_data = np.fromfile(file_path, dtype=read_type).astype(np.float32)
    # Combine 3 bytes into one 24-bit value
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    return raw_data

# ==============================
# 3. Loading an image from path
# ==============================
def load_image(img_path, device='cuda', norm_value=255.0, img_shape=None):
    """
    Loads an image from various formats:
      - .jpg/.png: via OpenCV
      - .npy: via NumPy
      - .raw: via a custom read function (24-bit example)
    Then converts to a PyTorch tensor on the specified device.
    """
    ext = os.path.splitext(img_path)[-1].lower()

    if ext in ['.jpg', '.jpeg', '.png']:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / norm_value

    elif ext in ['.npy']:
        img = np.load(img_path).astype(np.float32) / norm_value

    elif ext in ['.raw']:
        if img_shape is None:
            raise ValueError("Image shape must be provided for RAW images.")
        img = read_raw_24b(img_path, img_shape)
        img /= norm_value

    else:
        raise ValueError(f'Unsupported image format: {ext}')    

    # Convert (H, W, C) -> (C, H, W) for PyTorch
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(device)
    return img

# ==============================
# 4. Custom Dataset
# ==============================
class CustomDataset(Dataset):
    """
    Expects parallel lists of img_paths and label_paths (both same length).
    Each image can be .jpg/.png/.npy/.raw, each label is YOLO-format .txt.
    """
    def __init__(self, 
                 img_paths, 
                 label_paths, 
                 transform=None, 
                 device='cuda', 
                 img_shape=(1, 1856, 2880), 
                 norm_value=255.0):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform
        self.device = device
        self.img_shape = img_shape
        self.norm_value = norm_value

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        # Load image
        img = load_image(img_path, 
                         device=self.device, 
                         norm_value=self.norm_value, 
                         img_shape=self.img_shape)

        # Gather YOLO-format targets
        _, h, w = img.shape
        targets = load_yolo_targets(label_path, w, h, self.device)

        # Optionally apply transforms (resizing, color jitter, etc.)
        # Note: If using torchvision transforms, ensure they work with [C,H,W] Tensors
        if self.transform:
            img = self.transform(img)

        return img, targets

# ==============================
# 5. Utility to parse dataset.yaml
# ==============================
def parse_dataset_yaml(yaml_path):
    """
    Reads dataset.yaml and returns the paths for train, val, test.
    Expects keys: 'train', 'val', 'test'.
    """
    with open(yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)

    train_dir = data_dict['train']
    val_dir   = data_dict['val']
    test_dir  = data_dict.get('test', None)  # might not always be present

    return train_dir, val_dir, test_dir

# ==============================
# 6. Gather image & label paths
# ==============================
def gather_paths(images_dir, labels_dir, exts=('.jpg','.jpeg','.png','.npy','.raw')):
    """
    Collects all images from `images_dir` that match `exts`, 
    and pairs each with a YOLO .txt in `labels_dir`.
    """
    img_paths, label_paths = [], []

    for ext in exts:
        pattern = os.path.join(images_dir, f'*{ext}')
        for img_path in sorted(glob.glob(pattern)):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(labels_dir, base_name + '.txt')
            if os.path.exists(txt_path):
                img_paths.append(img_path)
                label_paths.append(txt_path)
            else:
                # If there's no matching .txt, skip or handle as needed
                pass

    return img_paths, label_paths

# ==============================
# 7. Create Dataset from dataset.yaml
# ==============================
def create_yolo_dataset(yaml_path, mode='train', transform=None, 
                        device='cuda', img_shape=(1,1856,2880), norm_value=255.0):
    """
    Reads dataset.yaml, picks the correct images folder for `mode`,
    infers the corresponding labels folder by replacing 'images' -> 'labels',
    and returns a CustomDataset.
    """
    train_dir, val_dir, test_dir = parse_dataset_yaml(yaml_path)

    if mode == 'train':
        images_dir = train_dir
    elif mode == 'val':
        images_dir = val_dir
    elif mode == 'test':
        if test_dir is None:
            raise ValueError("No 'test' key found in dataset.yaml.")
        images_dir = test_dir
    else:
        raise ValueError("mode must be one of ['train', 'val', 'test']")

    # Infer labels directory by replacing 'images' with 'labels' in the path
    labels_dir = images_dir.replace('images', 'labels')

    img_paths, label_paths = gather_paths(images_dir, labels_dir)

    dataset = CustomDataset(
        img_paths=img_paths,
        label_paths=label_paths,
        transform=transform,
        device=device,
        img_shape=img_shape,
        norm_value=norm_value
    )
    return dataset

# ==============================
# 8. Custom collate function
# ==============================
def detection_collate_fn(batch):
    """
    Expects a list of (img, target) pairs, where:
      - img: Tensor [C, H, W]
      - target: dict with keys 'boxes', 'labels', etc.
    Returns:
      - images: stacked tensor [batch_size, C, H, W]
      - targets: list of dicts (one per image)
    """
    images, targets = list(zip(*batch))  # unzip the batch into two lists
    images = torch.stack(images, dim=0)  # stack all images
    return images, list(targets)

# ==============================
# Example usage
# ==============================
if __name__ == "__main__":
    # Example: adjusting for your environment
    yaml_path = "/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/Multi-model-RAW-Network/data/dataset.yaml"  # ensure this is correct

    # Simple transform that resizes to 640x640 and normalizes.
    # For RAW data, you might remove color-based augmentations.
    example_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Normalize(mean=[0.5], std=[0.5])  # If single-channel, adapt as needed
    ])

    # Create train/val sets
    train_dataset = create_yolo_dataset(yaml_path, mode='train', transform=example_transform)
    val_dataset   = create_yolo_dataset(yaml_path, mode='val',   transform=example_transform)

    # Create DataLoaders with the custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=2,
        collate_fn=detection_collate_fn
    )

    # Quick test to confirm it loads
    for imgs, targets in train_loader:
        print("Batch images shape:", imgs.shape)  # e.g. [4, 3, 640, 640]
        print("Number of target dicts:", len(targets))  # should be 4
        for i, t in enumerate(targets):
            print(f"  Image {i} has {t['boxes'].shape[0]} boxes")
        break
