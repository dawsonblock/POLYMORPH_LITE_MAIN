import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from retrofitkit.core.workflows.db_logger import DatabaseLogger
from scripts.unified_cli import cmd_audit

# --- Audit Integration Tests ---

def test_db_logger_audit_integration():
    session_factory = MagicMock()
    session = MagicMock()
    session_factory.return_value = session
    
    # Mock AuditEvent query to return None (genesis)
    session.query.return_value.order_by.return_value.first.return_value = None
    
    logger = DatabaseLogger(session_factory)
    logger.run_id = "RUN-TEST"
    logger.execution_id = "123"
    
    # Mock WorkflowExecution
    execution = MagicMock()
    session.query.return_value.get.return_value = execution
    
    # Mock write_audit_event
    with patch("retrofitkit.compliance.audit.write_audit_event") as mock_write:
        logger.log_step_complete(1, "test_step", "success")
        
        mock_write.assert_called_once()
        call_args = mock_write.call_args[1]
        assert call_args["event_type"] == "STEP_COMPLETE"
        assert call_args["entity_id"] == "RUN-TEST"

# --- CLI Audit Tests ---

@pytest.mark.asyncio
async def test_cli_audit_verify():
    args = MagicMock()
    args.subcommand = "verify"
    config = MagicMock()
    
    with patch("retrofitkit.compliance.audit.verify_audit_chain") as mock_verify:
        mock_verify.return_value = {"valid": True, "entries_checked": 10, "errors": []}
        
        with patch("retrofitkit.db.session.SessionLocal") as mock_session_local:
            mock_session_local.return_value = MagicMock()
            await cmd_audit(args, config)
            mock_verify.assert_called_once()
