import os, json, base64, time
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import BaseModel

# FIX: Use environment variable or relative path, never absolute hardcoded /mnt
DATA_DIR = os.getenv("P4_DATA_DIR", "data")
KEY_DIR = os.path.join(DATA_DIR, "config", "keys")
PRIV_KEY_PATH = os.path.join(KEY_DIR, "private.pem")
PUB_KEY_PATH = os.path.join(KEY_DIR, "public.pem")

class SignatureRequest(BaseModel):
    record_id: int
    reason: str

class Signer:
    def __init__(self):
        self._ensure_keys_exist()

    def _ensure_keys_exist(self):
        if not os.path.exists(KEY_DIR):
            # Only auto-generate in non-production if missing, otherwise warn
            if os.getenv("ENVIRONMENT") != "production":
                os.makedirs(KEY_DIR, exist_ok=True)
            else:
                if not os.path.exists(PRIV_KEY_PATH):
                    raise RuntimeError(f"Production Signing Keys missing at {PRIV_KEY_PATH}")

    def _load_keys(self) -> Tuple[Any, Any]:
        if not os.path.exists(PRIV_KEY_PATH):
            raise FileNotFoundError(f"Private key not found: {PRIV_KEY_PATH}")
            
        with open(PRIV_KEY_PATH, "rb") as f:
            priv = serialization.load_pem_private_key(f.read(), password=None)
        with open(PUB_KEY_PATH, "rb") as f:
            pub = serialization.load_pem_public_key(f.read())
        return priv, pub

    def sign_record(self, req: SignatureRequest, signer_email: str):
        payload = json.dumps({
            "record_id": req.record_id,
            "reason": req.reason,
            "signer": signer_email,
            "ts": time.time()
        }, sort_keys=True).encode()
        
        priv, pub = self._load_keys()
        sig = priv.sign(payload, padding.PKCS1v15(), hashes.SHA256())
        b64sig = base64.b64encode(sig).decode()
        return {
            "signed": True,
            "signature": b64sig,
            "payload": json.loads(payload.decode())
        }
