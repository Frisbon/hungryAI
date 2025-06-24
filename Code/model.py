from torch import nn

class OurCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Layer del CNN
        self.cnn_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Image size: 128 -> 64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Image size: 64 -> 32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Image size: 32 -> 16
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Image size: 16 -> 8
        )
        
        # Er classifaier
        self.mlp_layers = nn.Sequential(
            nn.Flatten(),

            # nn.LazyLinear automatically detects the input features (16384 in this case) so I don't have to calculate it manually.
            # im also lazy btw
            nn.LazyLinear(1024), 
            
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        logits = self.mlp_layers(x)
        return logits