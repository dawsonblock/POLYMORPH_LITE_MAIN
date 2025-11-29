import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np
import torch

# Try to load the trained model, else fallback (for first run)
try:
    raman_runner = bentoml.pytorch.get("raman_predictor:latest").to_runner()
except:
    # Fallback if model not trained yet
    print("Warning: No trained model found. Using dummy runner.")
    raman_runner = None

svc = bentoml.Service("raman_service", runners=[raman_runner] if raman_runner else [])

@svc.api(input=NumpyNdarray(), output=JSON())
def predict(input_series: np.ndarray):
    if raman_runner is None:
        return {"error": "Model not trained yet. Run ai/train_raman_predictor.py"}
    
    # Ensure input shape
    tensor_input = torch.tensor(input_series, dtype=torch.float32)
    if len(tensor_input.shape) == 1:
        tensor_input = tensor_input.unsqueeze(0) # Add batch dim
        
    prediction = raman_runner.run(tensor_input)
    return {"concentration": float(prediction[0][0])}

@svc.api(input=JSON(), output=JSON())
def retrain(input_data: dict):
    """Trigger retraining (mocked for async execution)."""
    # In prod, this would launch a K8s job or Celery task
    return {"status": "Retraining started", "job_id": "job-123"}

@svc.api(input=JSON(), output=JSON())
def version(input_data: dict):
    return {"version": "v1.0.0", "framework": "pytorch"}
