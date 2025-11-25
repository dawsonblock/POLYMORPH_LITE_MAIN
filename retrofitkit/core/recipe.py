from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yaml

class Step(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class Recipe(BaseModel):
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
