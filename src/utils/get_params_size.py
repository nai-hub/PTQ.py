import torch.nn as nn 

def get_params_size(model):
    
    assert isinstance(model, nn.Module),f'Input type is {type(model)} !'    
    total_params = sum(p.numel() for p in model.parameters())
    total_params = total_params / 1e6 # M
    
    return total_params