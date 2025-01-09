import torch
import torch.nn as nn
import torch.nn.functional as F

class QRDQNModel(nn.Module):
    def __init__(self, num_actions, num_quantiles=51):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.num_actions = num_actions

        # Now we expect 3 input channels, for (210, 160, 3) observations
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # self.fc_input_dim = self._get_conv_output_dim()
        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_input_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_actions * num_quantiles)
        # )

        # We no longer hardcode 7×7×64 because the output size changes
        # Let's figure it out dynamically by using a placeholder
        
        #self.fc_input_dim = None  # We'll set this after a forward pass if needed
        
        # For a known shape, you can hardcode once you confirm the size:
        # e.g., for (210,160,3), it might be 10×8 after three conv layers => 10*8*64 = 5120

        # We'll guess a dimension, then we'll adapt if needed
        
        self.fc = nn.Sequential(
            nn.Linear(22528, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions * num_quantiles)
        )
        
    # def _get_conv_output_dim(self):
    #     # Dummy batch: (1, 3, 210, 160) for color Breakout
    #     dummy = torch.zeros(1, 3, 210, 160)
    #     out = self.conv(dummy)
    #     return out.numel()  # total elements for a single sample

    def forward(self, x):
        # x shape: (B, 210, 160, 3)
        # Convert to float, permute to (B, 3, H, W)
        x = x.float().permute(0, 3, 1, 2)  
        x = self.conv(x)
        #print("After conv shape:", x.shape)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        #print("Flattened shape:", x.shape)

        # If we don’t know the shape, we can do a quick print once or dynamically set fc_input_dim
        # print(x.shape)  # e.g., [Batch, Channels*Height*Width]
        # Then update the first Linear layer dimension accordingly

        x = self.fc(x)
        # Reshape to (B, num_actions, num_quantiles)
        x = x.view(-1, self.num_actions, self.num_quantiles)
        return x
