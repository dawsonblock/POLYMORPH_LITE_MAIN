import bentoml
import numpy as np
import torch
import redis
import json
import hashlib
from bentoml.io import JSON, NumpyNdarray
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from io import BytesIO
import base64

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
        # Load model version registry
        self.version_registry = self._load_version_registry()
        self.current_version = self.version_registry.get("current_version", "1.0.0")
        
        # Try to load the trained polymorph model
        try:
            model_file = self.version_registry["models"][self.current_version]["model_file"]
            model_path = Path("ai/models") / model_file
            
            if model_path.exists():
                # Load polymorph detection model
                from ai.training.train_polymorph import PolymorphCNN, TrainingConfig
                config = TrainingConfig()
                self.poly_model = PolymorphCNN(config)
                self.poly_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.poly_model.eval()
                print(f"Polymorph model v{self.current_version} loaded successfully.")
            else:
                print(f"Warning: Model file {model_path} not found. Using dummy mode.")
                self.poly_model = None
        except Exception as e:
            print(f"Warning: Polymorph model load failed ({e}). Using dummy mode.")
            self.poly_model = None
            
        # Also keep legacy model for backward compatibility
        try:
            self.model = bentoml.pytorch.load_model("raman_predictor:latest")
            self.model.eval()
        except Exception:
            self.model = None
    
    def _load_version_registry(self) -> Dict:
        """Load model version registry."""
        registry_path = Path("ai/model_version.json")
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        return {"current_version": "1.0.0", "models": {}}

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
        """Get service version information."""
        return {
            "service_version": "v4.0.0",
            "model_version": self.current_version,
            "framework": "pytorch",
            "service_api": "v1.4+",
            "features": [
                "polymorph_detection",
                "legacy_prediction",
                "model_versioning",
                "report_generation"
            ],
            "model_info": self.version_registry.get("models", {}).get(self.current_version, {})
        }
    
    @bentoml.api
    def polymorph_detect(self, input_data: dict) -> dict:
        """
        Detect polymorphs from Raman spectrum.
        
        Args:
            input_data: {
                "spectrum": [list of intensities],
                "wavelengths": [list of wavelengths],
                "metadata": {optional metadata}
            }
            
        Returns:
            {
                "polymorph_detected": bool,
                "polymorph_id": int,
                "polymorph_name": str,
                "confidence": float,
                "signature_vector": [list],
                "alternative_forms": [list of alternatives]
            }
        """
        if self.poly_model is None:
            return {
                "error": "Polymorph model not available",
                "polymorph_detected": False
            }
        
        try:
            # Extract spectrum
            spectrum = np.array(input_data.get("spectrum", []))
            
            if len(spectrum) == 0:
                return {"error": "Empty spectrum provided", "polymorph_detected": False}
            
            # Normalize spectrum
            spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)
            
            # Prepare input
            tensor_input = torch.FloatTensor(spectrum).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            
            # Run inference
            with torch.no_grad():
                output = self.poly_model(tensor_input)
                probabilities = torch.softmax(output, dim=1)[0]
                
            # Get prediction
            confidence, predicted_class = torch.max(probabilities, dim=0)
            confidence = float(confidence)
            predicted_class = int(predicted_class)
            
            # Check if confident enough
            min_threshold = self.version_registry.get("metadata", {}).get("min_confidence_threshold", 0.7)
            polymorph_detected = confidence >= min_threshold
            
            # Get alternative forms
            alternatives = []
            top_k_values, top_k_indices = torch.topk(probabilities, k=min(3, len(probabilities)))
            for conf, idx in zip(top_k_values[1:], top_k_indices[1:]):
                if float(conf) > 0.1:  # Only include if > 10% confidence
                    alternatives.append({
                        "polymorph_id": int(idx),
                        "polymorph_name": f"Form_{int(idx)}",
                        "confidence": float(conf)
                    })
            
            result = {
                "polymorph_detected": polymorph_detected,
                "polymorph_id": predicted_class,
               "polymorph_name": f"Form_{predicted_class}",
                "confidence": confidence,
                "signature_vector": output[0].tolist(),  # Raw logits as signature
                "alternative_forms": alternatives,
                "model_version": self.current_version,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add metadata if provided
            if "metadata" in input_data:
                result["input_metadata"] = input_data["metadata"]
            
            return result
            
        except Exception as e:
            return {
                "error": f"Polymorph detection failed: {str(e)}",
                "polymorph_detected": False
            }
    
    @bentoml.api
    def polymorph_report(self, input_data: dict) -> dict:
        """
        Generate polymorph detection report.
        
        Args:
            input_data: {
                "detection_result": {result from polymorph_detect},
                "format": "json" or "pdf",
                "include_spectrum": bool
            }
            
        Returns:
            {
                "format": "json" or "pdf",
                "data": {report data} or base64-encoded PDF
            }
        """
        detection_result = input_data.get("detection_result", {})
        report_format = input_data.get("format", "json")
        
        if report_format == "json":
            # JSON report
            report = {
                "report_id": hashlib.sha256(
                    json.dumps(detection_result, sort_keys=True).encode()
                ).hexdigest()[:16],
                "generated_at": datetime.now().isoformat(),
                "model_version": self.current_version,
                "detection_summary": {
                    "polymorph_detected": detection_result.get("polymorph_detected", False),
                    "primary_form": {
                        "id": detection_result.get("polymorph_id"),
                        "name": detection_result.get("polymorph_name"),
                        "confidence": detection_result.get("confidence")
                    },
                    "alternative_forms": detection_result.get("alternative_forms", [])
                },
                "raw_results": detection_result
            }
            
            return {
                "format": "json",
                "data": report
            }
        else:
            # PDF report (placeholder - would use reportlab or weasyprint)
            return {
                "format": "pdf",
                "data": "<base64_encoded_pdf_placeholder>",
                "note": "PDF generation requires reportlab - not implemented in this version"
            }
