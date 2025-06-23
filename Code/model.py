# In Code/model.py

from torch import nn

class OurCNN(nn.Module):
    """
    CNN architecture inspired by the "Pernosphere" project and course materials.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Feature Extractor part
        self.cnn_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier part (MLP)
        self.mlp_layers = nn.Sequential(
            nn.Flatten(),
            # Note: The input features '64 * 16 * 16' depend on the output of cnn_layers
            # and the input image size (128x128). This needs to be calculated.
            # 128 -> 64 (Pool1) -> 32 (Pool2) -> 16 (Pool3). So 64x16x16.
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout for regularization
            nn.Linear(512, num_classes) # num_classes is the number of food types
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        logits = self.mlp_layers(x)
        return logits