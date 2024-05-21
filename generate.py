# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:41:22 2021

@author: Kiminjo

Modified on Tue May 21 2024
Cuda is available now.
Feat.KwakJewoo
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataset import one_hot_encoding

def generate(model, seed_characters, temperature, network_type, char2int_dict, int2char_dict):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    samples = [string for string in seed_characters]
    hidden = model.init_hidden(1)
    
    for string in seed_characters :
        character, hidden = predict(model, string, hidden, temperature, char2int_dict, int2char_dict, device, network_type)

    samples.append(character)
    
    for i in range(2000) :
        
        char, hidden = predict(model, samples[-1], hidden, temperature, char2int_dict, int2char_dict, device, network_type)
        samples.append(char)
        
    return ''.join(samples)

def predict(model, string, hidden, temperature, char2int_dict, int2char_dict, device, network_type):
    x = torch.tensor([[char2int_dict[string]]], dtype=torch.float)
    x = one_hot_encoding(x)
    x = x.to(device)  # Ensure input tensor is on the correct device
    
    # Move each tensor in the hidden state tuple to the specified device
    if isinstance(hidden, tuple):
        hidden = tuple(h.to(device) for h in hidden)
    else:
        hidden = hidden.to(device)
    
    output, hidden = model.forward(x, hidden)
    
    # Apply temperature to the output probabilities
    prob = F.softmax(output / temperature, dim=1).data
    
    # Sample a character from the probability distribution
    top_char = torch.multinomial(prob, 1).cpu().numpy().squeeze()
    character = int2char_dict[int(top_char)]
    
    return character, hidden
