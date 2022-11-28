#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 05:39:32 2021

@author: W.H.
"""

import torch
import torch.nn as nn
import os

# from memory_profiler import profile 
# @profile(precision=4,stream=open('memory_profiler.log','w+'))
def get_model_size(input):
    """
    output: input size (MB)
    """
    if isinstance(input,nn.Module):
        torch.save(input.state_dict(),'temp.p')
    else:
        torch.save(input,'temp.p')
    
    out_size = os.path.getsize('temp.p') * 1.0 / 1e6
    os.remove('temp.p')

    return out_size

