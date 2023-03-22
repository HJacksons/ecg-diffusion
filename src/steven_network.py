import torch.nn as nn


class KanResWide_X(nn.Module):
    def __init__(self, input, output_size):
        super().__init__()
        self.input = input
        self.outpu_size = output_size
        
        self.initial_block = nn.Sequential(
            nn.Conv1d(in_channels=input, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.kanres_module = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=51, stride=1, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=51, stride=1, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([
            self.kanres_module for _ in range(8)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(32, output_size)


    def forward(self, x):
        x = self.initial_block(x)
        x = self.pool(x)
        for layer in self.layers:
            x = layer(x) + x
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
