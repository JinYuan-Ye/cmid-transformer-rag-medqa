import torch
import torch.nn as nn
import math

class DynamicWeightsLoss(nn.Module):
    def __init__(self):
        super(DynamicWeightsLoss, self).__init__()
        # Initialize learnable parameters (log_vars) for numerical stability
        # We use log_var = log(sigma^2) -> sigma^2 = exp(log_var)
        # 1 / (2 * sigma^2) = 1 / (2 * exp(log_var)) = 0.5 * exp(-log_var)
        # log(sigma) = 0.5 * log_var
        # Loss = sum( 0.5 * exp(-log_var) * loss + 0.5 * log_var )
        self.log_vars = nn.Parameter(torch.zeros(3)) 

    def forward(self, loss_4, loss_36, loss_gen):
        # Task 1: 4-class
        loss_total = 0.5 * torch.exp(-self.log_vars[0]) * loss_4 + 0.5 * self.log_vars[0]
        
        # Task 2: 36-class
        loss_total += 0.5 * torch.exp(-self.log_vars[1]) * loss_36 + 0.5 * self.log_vars[1]
        
        # Task 3: Generation
        loss_total += 0.5 * torch.exp(-self.log_vars[2]) * loss_gen + 0.5 * self.log_vars[2]
        
        return loss_total

def get_loss_functions(pad_id):
    # Standard CE for 4-class
    loss_fct_4 = nn.CrossEntropyLoss()
    
    # Label Smoothing for 36-class
    loss_fct_36 = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Label Smoothing for Generation (ignore padding)
    loss_fct_gen = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    
    return loss_fct_4, loss_fct_36, loss_fct_gen
