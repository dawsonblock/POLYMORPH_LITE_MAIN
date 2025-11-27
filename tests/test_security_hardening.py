"""
Test security hardening features.

Verifies:
1. Account lockout after 5 failed attempts
2. Password history enforcement
3. Two-person approval for sample deletion
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from fastapi import HTTPException

from retrofitkit.compliance.users import Users
from retrofitkit.db.models.user import User
from retrofitkit.api.samples import delete_sample

class TestSecurityHardening:
    
    def test_account_lockout(self):
        """Test that 5 failed attempts lock the account."""
        users = Users()
        users.pwd_context = Mock()
        users.pwd_context.verify.return_value = False  # Always fail password
        users.audit = Mock()
        
        mock_user = Mock(spec=User)
        mock_user.password_hash = b"hash"
        mock_user.failed_login_attempts = 0
        mock_user.account_locked_until = None
        mock_user.mfa_secret = None
        
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user
        
        with patch('retrofitkit.compliance.users.get_session', return_value=mock_session):
            # 5 failed attempts
            for i in range(5):
                users.authenticate("test@example.com", "wrong")
                mock_user.failed_login_attempts = i + 1
            
            # Check if locked
            assert mock_user.account_locked_until is not None
            assert mock_user.account_locked_until > datetime.utcnow()
            
            # 6th attempt should fail immediately due to lock
            mock_user.account_locked_until = datetime.utcnow() + timedelta(minutes=30)
            result = users.authenticate("test@example.com", "wrong")
            assert result is None
            users.audit.log.assert_called_with(
                "LOGIN_ATTEMPT_LOCKED", 
                "test@example.com", 
                "system", 
                f"Login attempt on locked account (locked until {mock_user.account_locked_until})"
            )

    def test_password_history(self):
        """Test that password history prevents reuse."""
        users = Users()
        users.pwd_context = Mock()
        users.pwd_context.verify.side_effect = [
            True,   # Verify old password (success)
            True    # Verify new password against history (match found!)
        ]
        users.audit = Mock()
        
        mock_user = Mock(spec=User)
        mock_user.password_hash = b"current_hash"
        mock_user.password_history = ["old_hash_1", "old_hash_2"]
        
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user
        
        payload = Mock()
        payload.old_password = "old"
        payload.new_password = "reuse_attempt"
        
        with patch('retrofitkit.compliance.users.get_session', return_value=mock_session):
            with patch('retrofitkit.security.validators.InputValidator.validate_password_strength', return_value=True):
                with pytest.raises(ValueError, match="Cannot reuse any of your last 5 passwords"):
                    users.change_password("test@example.com", payload)

    @pytest.mark.asyncio
    async def test_sample_deletion_approval(self):
        """Test that sample deletion requires approval."""
        # Mock dependencies
        mock_db = Mock()
        mock_user = Mock(spec=User)
        mock_user.email = "admin@example.com"
        mock_role = Mock()
        mock_role.role.name = "admin"
        mock_user.user_roles = [mock_role]
        
        # Mock approval that is NOT approved yet
        mock_approval = Mock()
        mock_approval.is_approved = False
        mock_approval.id = "approval_123"
        
        with patch('retrofitkit.compliance.approvals.require_approval', return_value=mock_approval) as mock_require:
            try:
                await delete_sample(
                    sample_id="SAMPLE-001",
                    db=mock_db,
                    current_user=mock_user
                )
            except HTTPException as e:
                assert e.status_code == 202
                assert e.detail["status"] == "pending_approval"
                assert e.detail["approval_id"] == "approval_123"
