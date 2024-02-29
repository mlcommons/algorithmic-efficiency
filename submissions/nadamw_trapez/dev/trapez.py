import torch
import torch.nn as nn
import wandb

from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import SequentialLR

# Define a simple model
class SimpleModel(nn.Module):
  def __init__(self):
    super(SimpleModel, self).__init__()
    self.fc = nn.Linear(10, 2)  # A simple linear layer

  def forward(self, x):
    return self.fc(x)

# Initialize the model
model = SimpleModel()

# Define an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

# Scheduler
warmup_steps = 10
warmup = LinearLR(
    optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)

constant_steps = 40
constant = ConstantLR(optimizer, factor=1.0, total_iters=constant_steps)

decay_steps = 50
linear_decay = LinearLR(
    optimizer, start_factor=1., end_factor=0., total_iters=decay_steps)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, constant, linear_decay],
    milestones=[warmup_steps, constant_steps+warmup_steps])

wandb.init(project='exp')

for step in range(100):
  optimizer.step()
  scheduler.step()
  wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
