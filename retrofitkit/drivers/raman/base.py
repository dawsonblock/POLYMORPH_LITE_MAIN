from typing import Dict, Any
class RamanBase:
    async def read_frame(self) -> Dict[str, Any]:
        raise NotImplementedError
