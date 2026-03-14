import torch
import torch.nn as nn
import torchvision
from model import ShortChunkCNN

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    model = ShortChunkCNN()
    info = get_parameter_number(model)
    print(f"Total: {info['Total']}")
    print(f"Trainable: {info['Trainable']}")