import torch.nn as nn
import torch
from src.SE import SEBlock

class GRUModel(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        dropout_rate=0.2,
        height=48,
        width=72,
        gru_hidden=512,
        gru_layers=2
    ):
        super().__init__()
        self.spatial_encoder1 = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(init_dim, init_dim * 2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(init_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(init_dim * 2, init_dim * 2, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(init_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(init_dim * 2, init_dim * 4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(init_dim * 4),
            nn.ReLU(inplace=True)
        )

        self.spatial_encoder2 = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(init_dim, init_dim * 2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(init_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(init_dim * 2, init_dim * 2, kernel_size=kernel_size, stride=2, padding=1),  # ↓ 2x spatial resolution
            nn.BatchNorm2d(init_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(init_dim * 2, init_dim * 4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(init_dim * 4),
            nn.ReLU(inplace=True)
        )

        self.se1 = SEBlock(channels=init_dim*4)
        self.se2 = SEBlock(channels=init_dim*4)

        flattened_size = init_dim*4 * height//2 * width//2

        self.temporal_model1 = nn.GRU(
            input_size=flattened_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        self.temporal_model2 = nn.GRU(
            input_size=flattened_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        self.ln1 = nn.LayerNorm(gru_hidden)

        self.ln2 = nn.LayerNorm(gru_hidden)

        self.output_proj1 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden * 2),
            nn.LayerNorm(gru_hidden * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(gru_hidden * 2, n_output_channels // 2 * height * width),
            nn.Unflatten(1, (n_output_channels // 2, height, width))
        )

        self.output_proj2 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden * 2),
            nn.LayerNorm(gru_hidden * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(gru_hidden * 2, n_output_channels // 2 * height * width),
            nn.Unflatten(1, (n_output_channels // 2, height, width))
        )

    def forward(self, x): 
        B, T, C, H, W = x.shape                 # x shape [B, T, C, H, W]
        x = x.view(B * T, C, H, W)
        tas_x = self.spatial_encoder1(x)        # [B*T, init_dim, H, W]
        pr_x = self.spatial_encoder2(x)
        tas_se = self.se1(tas_x)
        pr_se = self.se2(pr_x)
        tas_se = tas_se.view(B,T,-1)            # [B, T, init_dim*H*W]
        pr_se = pr_se.view(B,T,-1)              # [B, T, init_dim*H*W]
        tas_gru_out,_ = self.temporal_model1(tas_se)
        tas_gru_out = self.ln1(tas_gru_out)     
        tas_hidden = tas_gru_out[:,-1]          # [B, gru_hidden]
        pr_gru_out,_ = self.temporal_model2(pr_se)
        pr_gru_out = self.ln2(pr_gru_out)
        pr_hidden = pr_gru_out[:,-1]            # [B, gru_hidden]
        tas_out = self.output_proj1(tas_hidden)
        pr_out = self.output_proj2(pr_hidden)
        out = torch.cat([tas_out,pr_out], dim=1) # [B, n_output_channels, H, W]
        return out