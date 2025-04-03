import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.bottleneck = nn.Linear(vocab_size, 512)
        self.output = nn.Linear(512, vocab_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.bottleneck.weight, std=0.02)
        nn.init.zeros_(self.bottleneck.bias)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.bottleneck(x))
