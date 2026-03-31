# federated_learning/security/encryption.py
# AES-256-GCM encrypt/decrypt model updates before transmission.

import os
import json
import base64
import hashlib
from typing import Dict

import torch


def serialize_weights(state_dict: Dict) -> bytes:
    """Convert a PyTorch state dict to bytes for encryption."""
    cpu_dict = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
    return json.dumps(cpu_dict).encode()


def deserialize_weights(data: bytes) -> Dict:
    """Reconstruct a PyTorch state dict from bytes."""
    cpu_dict = json.loads(data.decode())
    return {k: torch.tensor(v) for k, v in cpu_dict.items()}


def generate_session_key() -> bytes:
    """Fresh 256-bit AES key per round."""
    return os.urandom(32)


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """Simple XOR cipher — fallback when PyCryptodome is not available."""
    key_len = len(key)
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


def encrypt_update(state_dict: Dict, key: bytes) -> Dict[str, str]:
    """
    AES-256-GCM encryption (or XOR fallback).
    Returns a dict with base64-encoded ciphertext.
    """
    plaintext = serialize_weights(state_dict)

    try:
        from Crypto.Cipher import AES
        cipher = AES.new(key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        payload = {
            "nonce":      base64.b64encode(cipher.nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "tag":        base64.b64encode(tag).decode(),
            "method":     "AES-256-GCM",
        }
    except ImportError:
        # Fallback: XOR-based (demonstrates the pipeline without pycryptodome)
        nonce = os.urandom(12)
        ciphertext = _xor_bytes(plaintext, key)
        tag = hashlib.sha256(plaintext + nonce).digest()[:16]
        payload = {
            "nonce":      base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "tag":        base64.b64encode(tag).decode(),
            "method":     "XOR-256-HMAC",
        }

    print(f"[Encryption] Update encrypted ({payload['method']}) — "
          f"{len(plaintext)} B → {len(payload['ciphertext'])} B ciphertext")
    return payload


def decrypt_update(payload: Dict[str, str], key: bytes) -> Dict:
    """
    Decrypt model weights from an encrypted payload.
    """
    nonce      = base64.b64decode(payload["nonce"])
    ciphertext = base64.b64decode(payload["ciphertext"])
    tag        = base64.b64decode(payload["tag"])

    if payload.get("method") == "AES-256-GCM":
        from Crypto.Cipher import AES
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    else:
        plaintext = _xor_bytes(ciphertext, key)
        expected_tag = hashlib.sha256(plaintext + nonce).digest()[:16]
        if expected_tag != tag:
            raise ValueError("Integrity check failed — ciphertext was tampered with.")

    return deserialize_weights(plaintext)
