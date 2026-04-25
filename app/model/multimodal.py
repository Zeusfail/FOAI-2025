import torch
import torch.nn as nn
from torchvision import models


class MultimodalToxicityModel(nn.Module):
    def __init__(self, tabular_input_size):
        super().__init__()

        self.cnn_branch = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        num_ftrs = self.cnn_branch.classifier[1].in_features
        self.cnn_branch.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
        )

        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.attention = nn.Sequential(
            nn.Linear(256 + 128, 384),
            nn.Tanh(),
            nn.Linear(384, 384),
            nn.Softmax(dim=1),
        )

        self.fusion_layers = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.5),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, image, tabular):
        img_features = self.cnn_branch(image)
        tab_features = self.tabular_branch(tabular)
        combined = torch.cat([img_features, tab_features], dim=1)
        attention_weights = self.attention(combined)
        weighted_features = combined * attention_weights
        return self.fusion_layers(weighted_features)
