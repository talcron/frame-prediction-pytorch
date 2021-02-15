import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv2d(input_, input_dim, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, name='conv2d', padding='SAME'):
    def uniform(std_dev, size):
        return np.random.uniform(
            low=-std_dev * np.sqrt(3),
            high=std_dev * np.sqrt(3),
            size=size
        ).astype('float32')
    """ 
    init weights like in    
    "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    """
    results = nn.Conv2d()