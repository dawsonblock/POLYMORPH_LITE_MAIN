from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yaml

class Step(BaseModel):
    type: str
    params: Dict[str, Any] = {}

RecipeStep = Step

import uuid

class Recipe(BaseModel):
    id: Optional[uuid.UUID] = None
    name: str
    version: str = "1.0"
    metadata: Dict[str, Any] = {}
    steps: List[Step]

    @staticmethod
    def from_yaml(path: str) -> "Recipe":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        d["steps"] = [Step(**s) for s in d.get("steps", [])]
        return Recipe(**d)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recipe to dictionary for serialization."""
        return {
            "id": str(self.id) if self.id else None,
            "name": self.name,
            "version": self.version,
            "metadata": self.metadata,
            "steps": [{"type": s.type, "params": s.params} for s in self.steps]
        }
