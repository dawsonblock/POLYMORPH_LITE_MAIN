import pytest
import torch
import numpy as np
from ai.model import RamanCNN

def test_model_architecture():
    model = RamanCNN()
    assert isinstance(model, RamanCNN)

def test_forward_pass():
    model = RamanCNN()
    # Create dummy input (batch_size=1, input_size=1024)
    input_tensor = torch.randn(1, 1024)
    output = model(input_tensor)
    
    # Output should be (1, 1) - concentration
    assert output.shape == (1, 1)
    
    # Output should be between 0 and 1 (Sigmoid)
    val = output.item()
    assert 0.0 <= val <= 1.0

def test_batch_processing():
    model = RamanCNN()
    batch_size = 5
    input_tensor = torch.randn(batch_size, 1024)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 1)
