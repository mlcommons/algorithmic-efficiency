import torch 
import numpy as np
from spectrum_augmenter import SpecAug

inputs = torch.ones(256, 2048, 80)
input_paddings = torch.zeros(256, 2048)

specaug = SpecAug()
outputs, output_paddings = specaug(inputs, input_paddings)

print(outputs.shape)