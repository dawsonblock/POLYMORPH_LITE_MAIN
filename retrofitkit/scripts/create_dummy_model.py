import torch
import torch.nn as nn
import torch.onnx
import os

class SimpleRamanCNN(nn.Module):
    def __init__(self):
        super(SimpleRamanCNN, self).__init__()
        # Input: 1 channel (intensity), 1000 points
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 498, 3) # 3 classes: Form I, Form II, Amorphous
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.softmax(x)

def export_model():
    model = SimpleRamanCNN()
    model.eval()
    
    # Dummy input: Batch size 1, 1 Channel, 1000 Wavelength points
    dummy_input = torch.randn(1, 1, 1000)
    
    output_path = "retrofitkit/core/ai/models/raman_model.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    export_model()
