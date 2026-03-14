import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        #################################################
        # create your model here
        #################################################
        self.model = GPT2Model(cfg)

    def forward(self, x):
        #################################################
        # create your model here
        #################################################
        y = self.model(x)
        return y
    
if __name__ == "__main__":
    cfg = GPT2Config()
    x = torch.randint(0, cfg.vocab_size, (1, 512))
    model = Model(cfg)
    y = model(x)
    output = y['last_hidden_state'] # outputshape = (batch, x_len, d_embed)