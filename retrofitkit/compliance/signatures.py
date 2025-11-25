import os, json, base64, time
from dataclasses import dataclass
from typing import Dict, Any
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from pydantic import BaseModel

PRIV = "/mnt/data/Polymorph4_Retrofit_Kit_v1/config/keys/private.pem"
PUB = "/mnt/data/Polymorph4_Retrofit_Kit_v1/config/keys/public.pem"

class SignatureRequest(BaseModel):
    record_id: int
    reason: str

class Signer:
    def __init__(self):
        pass

    def _load_keys(self):
        with open(PRIV, "rb") as f:
            priv = serialization.load_pem_private_key(f.read(), password=None)
        with open(PUB, "rb") as f:
            pub = serialization.load_pem_public_key(f.read())
        return priv, pub

    def sign_record(self, req: SignatureRequest, signer_email: str):
        payload = json.dumps({"record_id": req.record_id, "reason": req.reason, "signer": signer_email, "ts": time.time()}, sort_keys=True).encode()
        priv, pub = self._load_keys()
        sig = priv.sign(payload, padding.PKCS1v15(), hashes.SHA256())
        b64sig = base64.b64encode(sig).decode()
        return {"signed": True, "signature": b64sig, "payload": json.loads(payload.decode())}
