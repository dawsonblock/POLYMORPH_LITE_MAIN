"""
Tests for electronic signature functionality (21 CFR Part 11 compliance).

This module tests:
- RSA key pair generation and management
- Digital signature creation
- Signature verification
- Payload integrity
- Timestamp inclusion
- Key security
"""
import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from retrofitkit.compliance.signatures import Signer, SignatureRequest


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data and keys."""
    temp_dir = tempfile.mkdtemp()
    old_data_dir = os.environ.get("P4_DATA_DIR")
    os.environ["P4_DATA_DIR"] = temp_dir
    yield temp_dir
    # Cleanup
    if old_data_dir:
        os.environ["P4_DATA_DIR"] = old_data_dir
    else:
        if "P4_DATA_DIR" in os.environ:
            del os.environ["P4_DATA_DIR"]
    shutil.rmtree(temp_dir)


@pytest.fixture
def signer_with_keys(temp_data_dir):
    """Create a Signer instance with test keys."""
    key_dir = os.path.join(temp_data_dir, "config", "keys")
    os.makedirs(key_dir, exist_ok=True)

    # Generate test keys
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    # Save keys
    priv_path = os.path.join(key_dir, "private.pem")
    pub_path = os.path.join(key_dir, "public.pem")

    with open(priv_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open(pub_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    return Signer()


class TestSignerInitialization:
    """Test cases for Signer initialization and key management."""

    def test_signer_initialization_creates_key_dir(self, temp_data_dir):
        """Test that Signer creates key directory in non-production."""
        old_env = os.environ.get("ENVIRONMENT")
        os.environ["ENVIRONMENT"] = "development"
        
        key_dir = os.path.join(temp_data_dir, "config", "keys")
        priv_path = os.path.join(key_dir, "private.pem")
        
        try:
            # Patch module-level constants because they are evaluated at import time
            with patch("retrofitkit.compliance.signatures.KEY_DIR", key_dir), \
                 patch("retrofitkit.compliance.signatures.PRIV_KEY_PATH", priv_path), \
                 patch("os.path.exists", return_value=False), \
                 patch("os.makedirs") as mock_makedirs:
                
                signer = Signer()
                
                mock_makedirs.assert_called_with(key_dir, exist_ok=True)
        finally:
            if old_env:
                os.environ["ENVIRONMENT"] = old_env
            else:
                del os.environ["ENVIRONMENT"]

    def test_signer_fails_in_production_without_keys(self, temp_data_dir):
        """Test that Signer raises error in production when keys are missing."""
        old_env = os.environ.get("ENVIRONMENT")
        os.environ["ENVIRONMENT"] = "production"
        
        key_dir = os.path.join(temp_data_dir, "config", "keys")
        priv_path = os.path.join(key_dir, "private.pem")
        
        try:
            with patch("retrofitkit.compliance.signatures.KEY_DIR", key_dir), \
                 patch("retrofitkit.compliance.signatures.PRIV_KEY_PATH", priv_path), \
                 patch("os.path.exists", return_value=False):
                
                with pytest.raises(RuntimeError) as exc_info:
                    signer = Signer()

            assert "Production Signing Keys missing" in str(exc_info.value)
        finally:
            if old_env:
                os.environ["ENVIRONMENT"] = old_env
            else:
                del os.environ["ENVIRONMENT"]

    def test_load_keys_raises_when_missing(self, temp_data_dir):
        """Test that _load_keys raises FileNotFoundError when keys don't exist."""
        old_env = os.environ.get("ENVIRONMENT")
        os.environ["ENVIRONMENT"] = "development"
        
        key_dir = os.path.join(temp_data_dir, "config", "keys")
        priv_path = os.path.join(key_dir, "private.pem")
        
        try:
            with patch("retrofitkit.compliance.signatures.KEY_DIR", key_dir), \
                 patch("retrofitkit.compliance.signatures.PRIV_KEY_PATH", priv_path), \
                 patch("os.path.exists", return_value=False):
                
                signer = Signer()
                
                with pytest.raises(FileNotFoundError) as exc_info:
                    signer._load_keys()

            assert "Private key not found" in str(exc_info.value)
        finally:
            if old_env:
                os.environ["ENVIRONMENT"] = old_env
            else:
                del os.environ["ENVIRONMENT"]


class TestSignatureCreation:
    """Test cases for creating digital signatures."""

    def test_sign_record_creates_valid_signature(self, signer_with_keys):
        """Test that sign_record creates a valid signature."""
        request = SignatureRequest(record_id=123, reason="Test approval")

        result = signer_with_keys.sign_record(request, "test@example.com")

        assert result["signed"] is True
        assert "signature" in result
        assert "payload" in result
        assert isinstance(result["signature"], str)
        assert len(result["signature"]) > 0

    def test_signature_payload_contains_required_fields(self, signer_with_keys):
        """Test that signature payload contains all required fields."""
        request = SignatureRequest(record_id=456, reason="QA approval")

        result = signer_with_keys.sign_record(request, "qa@example.com")
        payload = result["payload"]

        assert payload["record_id"] == 456
        assert payload["reason"] == "QA approval"
        assert payload["signer"] == "qa@example.com"
        assert "ts" in payload
        assert isinstance(payload["ts"], float)

    def test_signature_includes_timestamp(self, signer_with_keys):
        """Test that signature includes current timestamp."""
        import time
        request = SignatureRequest(record_id=789, reason="Supervisor approval")

        before_time = time.time()
        result = signer_with_keys.sign_record(request, "supervisor@example.com")
        after_time = time.time()

        timestamp = result["payload"]["ts"]
        assert before_time <= timestamp <= after_time

    def test_signature_is_unique_per_request(self, signer_with_keys):
        """Test that each signature is unique due to timestamp."""
        request1 = SignatureRequest(record_id=100, reason="First")
        request2 = SignatureRequest(record_id=100, reason="First")

        result1 = signer_with_keys.sign_record(request1, "test@example.com")
        result2 = signer_with_keys.sign_record(request2, "test@example.com")

        # Signatures should differ due to different timestamps
        assert result1["signature"] != result2["signature"]

    def test_signature_differs_by_signer(self, signer_with_keys):
        """Test that signatures differ when signed by different users."""
        request = SignatureRequest(record_id=200, reason="Approval")

        result1 = signer_with_keys.sign_record(request, "user1@example.com")
        result2 = signer_with_keys.sign_record(request, "user2@example.com")

        assert result1["payload"]["signer"] != result2["payload"]["signer"]

    def test_signature_base64_encoded(self, signer_with_keys):
        """Test that signature is base64 encoded."""
        import base64
        request = SignatureRequest(record_id=300, reason="Test")

        result = signer_with_keys.sign_record(request, "test@example.com")
        signature = result["signature"]

        # Should be valid base64
        try:
            decoded = base64.b64decode(signature)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Signature is not valid base64")


class TestSignatureVerification:
    """Test cases for verifying digital signatures."""

    def test_signature_can_be_verified(self, signer_with_keys):
        """Test that created signature can be verified with public key."""
        import base64
        request = SignatureRequest(record_id=400, reason="Verification test")

        result = signer_with_keys.sign_record(request, "test@example.com")

        # Reconstruct payload
        payload = json.dumps(result["payload"], sort_keys=True).encode()
        signature = base64.b64decode(result["signature"])

        # Load public key and verify
        _, pub_key = signer_with_keys._load_keys()

        try:
            pub_key.verify(signature, payload, padding.PKCS1v15(), hashes.SHA256())
            verified = True
        except Exception:
            verified = False

        assert verified is True

    def test_tampered_payload_fails_verification(self, signer_with_keys):
        """Test that tampered payload fails signature verification."""
        import base64
        request = SignatureRequest(record_id=500, reason="Tamper test")

        result = signer_with_keys.sign_record(request, "test@example.com")

        # Tamper with payload
        tampered_payload = result["payload"].copy()
        tampered_payload["record_id"] = 999  # Change value
        tampered_json = json.dumps(tampered_payload, sort_keys=True).encode()

        signature = base64.b64decode(result["signature"])
        _, pub_key = signer_with_keys._load_keys()

        # Verification should fail
        with pytest.raises(Exception):  # Will raise InvalidSignature
            pub_key.verify(signature, tampered_json, padding.PKCS1v15(), hashes.SHA256())

    def test_wrong_signature_fails_verification(self, signer_with_keys):
        """Test that wrong signature fails verification."""
        import base64
        request = SignatureRequest(record_id=600, reason="Wrong sig test")

        result = signer_with_keys.sign_record(request, "test@example.com")

        # Use correct payload but wrong signature
        payload = json.dumps(result["payload"], sort_keys=True).encode()
        wrong_signature = base64.b64decode(result["signature"])[::-1]  # Reverse bytes

        _, pub_key = signer_with_keys._load_keys()

        with pytest.raises(Exception):
            pub_key.verify(wrong_signature, payload, padding.PKCS1v15(), hashes.SHA256())


class TestPayloadIntegrity:
    """Test cases for payload integrity and structure."""

    def test_payload_json_sorted_keys(self, signer_with_keys):
        """Test that payload JSON uses sorted keys for consistency."""
        request = SignatureRequest(record_id=700, reason="Sorted test")

        result = signer_with_keys.sign_record(request, "test@example.com")
        payload = result["payload"]

        # Keys should be in sorted order
        keys = list(payload.keys())
        assert keys == sorted(keys)

    def test_payload_preserves_record_id(self, signer_with_keys):
        """Test that payload preserves record ID exactly."""
        record_ids = [1, 999, 123456789]

        for record_id in record_ids:
            request = SignatureRequest(record_id=record_id, reason="ID test")
            result = signer_with_keys.sign_record(request, "test@example.com")

            assert result["payload"]["record_id"] == record_id

    def test_payload_preserves_reason(self, signer_with_keys):
        """Test that payload preserves reason text exactly."""
        reasons = [
            "Simple reason",
            "Reason with special chars: !@#$%",
            "Multi\nline\nreason",
            "Unicode: 测试",
        ]

        for reason in reasons:
            request = SignatureRequest(record_id=1, reason=reason)
            result = signer_with_keys.sign_record(request, "test@example.com")

            assert result["payload"]["reason"] == reason

    def test_payload_preserves_signer_email(self, signer_with_keys):
        """Test that payload preserves signer email exactly."""
        emails = [
            "simple@example.com",
            "user+tag@example.com",
            "first.last@sub.domain.com",
        ]

        for email in emails:
            request = SignatureRequest(record_id=1, reason="Email test")
            result = signer_with_keys.sign_record(request, email)

            assert result["payload"]["signer"] == email


class TestKeyManagement:
    """Test cases for cryptographic key management."""

    def test_private_key_is_rsa_2048(self, signer_with_keys):
        """Test that private key is RSA with appropriate key size."""
        priv_key, _ = signer_with_keys._load_keys()

        assert priv_key.key_size >= 2048  # Minimum secure key size

    def test_keys_use_pem_format(self, temp_data_dir, signer_with_keys):
        """Test that keys are stored in PEM format."""
        key_dir = os.path.join(temp_data_dir, "config", "keys")
        priv_path = os.path.join(key_dir, "private.pem")
        pub_path = os.path.join(key_dir, "public.pem")

        with open(priv_path, "rb") as f:
            priv_content = f.read()
        with open(pub_path, "rb") as f:
            pub_content = f.read()

        assert b"BEGIN PRIVATE KEY" in priv_content
        assert b"BEGIN PUBLIC KEY" in pub_content

    def test_private_key_not_encrypted(self, signer_with_keys):
        """Test that private key loads without password (documented behavior)."""
        # This is the current implementation - keys are not password protected
        # In production, consider adding password protection
        priv_key, _ = signer_with_keys._load_keys()
        assert priv_key is not None


class TestCFRCompliance:
    """Test cases for 21 CFR Part 11 compliance requirements."""

    def test_signature_uniquely_identifies_signer(self, signer_with_keys):
        """Test that signature payload includes signer identification."""
        request = SignatureRequest(record_id=1, reason="CFR test")
        result = signer_with_keys.sign_record(request, "signer@example.com")

        assert "signer" in result["payload"]
        assert result["payload"]["signer"] == "signer@example.com"

    def test_signature_includes_date_and_time(self, signer_with_keys):
        """Test that signature includes date/time (CFR requirement)."""
        request = SignatureRequest(record_id=1, reason="CFR test")
        result = signer_with_keys.sign_record(request, "test@example.com")

        assert "ts" in result["payload"]
        # Timestamp should be recent
        import time
        assert abs(time.time() - result["payload"]["ts"]) < 2

    def test_signature_includes_meaning(self, signer_with_keys):
        """Test that signature includes meaning/reason (CFR requirement)."""
        request = SignatureRequest(record_id=1, reason="Approved for production")
        result = signer_with_keys.sign_record(request, "test@example.com")

        assert "reason" in result["payload"]
        assert result["payload"]["reason"] == "Approved for production"

    def test_signature_cannot_be_reused(self, signer_with_keys):
        """Test that signatures have unique timestamps preventing reuse."""
        request = SignatureRequest(record_id=1, reason="Test")

        result1 = signer_with_keys.sign_record(request, "test@example.com")
        result2 = signer_with_keys.sign_record(request, "test@example.com")

        # Timestamps should differ
        assert result1["payload"]["ts"] != result2["payload"]["ts"]
        # Signatures should differ
        assert result1["signature"] != result2["signature"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_sign_with_empty_reason(self, signer_with_keys):
        """Test signing with empty reason string."""
        request = SignatureRequest(record_id=1, reason="")
        result = signer_with_keys.sign_record(request, "test@example.com")

        assert result["payload"]["reason"] == ""
        assert result["signed"] is True

    def test_sign_with_very_long_reason(self, signer_with_keys):
        """Test signing with very long reason text."""
        long_reason = "A" * 10000
        request = SignatureRequest(record_id=1, reason=long_reason)
        result = signer_with_keys.sign_record(request, "test@example.com")

        assert result["payload"]["reason"] == long_reason
        assert result["signed"] is True

    def test_sign_with_special_characters_in_email(self, signer_with_keys):
        """Test signing with special characters in email."""
        email = "test+special@example.com"
        request = SignatureRequest(record_id=1, reason="Test")
        result = signer_with_keys.sign_record(request, email)

        assert result["payload"]["signer"] == email
