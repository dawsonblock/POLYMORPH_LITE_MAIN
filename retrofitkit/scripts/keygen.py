from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import os

BASE = os.getenv("P4_KEYS_DIR", "data/keys")
PRIV = os.path.join(BASE, "private.pem")
PUB = os.path.join(BASE, "public.pem")

def ensure_keys():
    os.makedirs(BASE, exist_ok=True)
    if os.path.exists(PRIV) and os.path.exists(PUB):
        return
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(PRIV, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()
        ))
    with open(PUB, "wb") as f:
        f.write(key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo
        ))
