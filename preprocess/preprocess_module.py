import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessModule(nn.Module):
    def __init__(self):
        super(PreprocessModule, self).__init__()
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.tensor(1.5))  # Learnable gamma initialization
        self.denoise_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.denoise_bn = nn.BatchNorm2d(32)
        self.edge_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.constant_(self.edge_conv.weight, 0.0)  # Initialize as high-pass filter
        self.fusion_conv = nn.Conv2d(64, 32, kernel_size=1, bias=False)  # Fuse denoised and edge-enhanced features
        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = torch.pow(torch.clamp(x, 0, 1), self.gamma)  # Learnable gamma correction
        denoised = self.relu(self.denoise_bn(self.denoise_conv(x)))
        edges = self.edge_conv(x)
        fused = torch.cat([denoised, edges], dim=1)
        fused = self.relu(self.fusion_conv(fused))
        output = torch.sigmoid(self.output_conv(fused))  # Output in [0, 1] range
        return output


class BayerPacking(nn.Module):
    def __init__(self):
        super(BayerPacking, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=2, stride=2, bias=False)
        r_filter = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32)  # R
        g_filter = torch.tensor([[[[0, 0.5], [0.5, 0]]]], dtype=torch.float32)  # G (averaged)
        b_filter = torch.tensor([[[[0, 0], [0, 1]]]], dtype=torch.float32)  # B
        filters = torch.cat([r_filter, g_filter, b_filter], dim=0)  # Shape: (3,1,2,2)
        self.conv.weight = nn.Parameter(filters, requires_grad=False)

    def forward(self, raw_img):
        return self.conv(raw_img) # raw_img: (batch, 1, H, W) -> (batch, 3, H/2, W/2)   
        
