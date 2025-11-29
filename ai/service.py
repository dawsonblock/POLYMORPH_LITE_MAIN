import bentoml
import numpy as np
import torch
import redis
import json
import hashlib
from bentoml.io import JSON, NumpyNdarray

# Initialize Redis client (fail gracefully if not available)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=0.1)
    redis_client.ping()
except Exception:
    print("Warning: Redis not available. Caching disabled.")
    redis_client = None

@bentoml.service(name="raman_service")
class RamanService:
    def __init__(self):
        # Try to load the trained model, else fallback
        try:
            # In new BentoML, we can load the model directly or use a runner if needed.
            # Runners are deprecated, so we load the model into memory.
            # Assuming PyTorch model.
            self.model = bentoml.pytorch.load_model("raman_predictor:latest")
            self.model.eval() # Set to eval mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Warning: No trained model found or load failed ({e}). Using dummy mode.")
            self.model = None

    @bentoml.api
    def predict(self, input_series: np.ndarray) -> dict:
        if self.model is None:
            return {"error": "Model not trained yet. Run ai/train_raman_predictor.py"}
        
        # Ensure input shape
        tensor_input = torch.tensor(input_series, dtype=torch.float32)
        if len(tensor_input.shape) == 1:
            tensor_input = tensor_input.unsqueeze(0) # Add batch dim
            
        # Check cache
        if redis_client:
            try:
                # Create a hash of the input array
                input_hash = hashlib.sha256(input_series.tobytes()).hexdigest()
                cached_result = redis_client.get(f"raman_pred:{input_hash}")
                if cached_result:
                    return json.loads(cached_result)
            except Exception as e:
                print(f"Cache read error: {e}")

        with torch.no_grad():
            prediction = self.model(tensor_input)
        
        result = {"concentration": float(prediction[0][0])}
        
        # Write to cache
        if redis_client:
            try:
                redis_client.setex(f"raman_pred:{input_hash}", 3600, json.dumps(result))
            except Exception as e:
                print(f"Cache write error: {e}")
                
        return result

    @bentoml.api
    def retrain(self, input_data: dict) -> dict:
        """Trigger retraining (mocked for async execution)."""
        # In prod, this would launch a K8s job or Celery task
        return {"status": "Retraining started", "job_id": "job-123"}

    @bentoml.api
    def version(self, input_data: dict) -> dict:
        return {"version": "v1.0.0", "framework": "pytorch", "service_api": "v1.4+"}
