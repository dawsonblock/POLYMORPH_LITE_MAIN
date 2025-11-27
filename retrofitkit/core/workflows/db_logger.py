"""
Database Logger for Workflow Execution.

Handles persistence of run status, step logs, and audit events.
"""
import logging
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from retrofitkit.db.models.workflow import WorkflowExecution
from retrofitkit.db.models.audit import AuditEvent
# from retrofitkit.compliance.audit import AuditLogger # Removed: not implemented yet

logger = logging.getLogger(__name__)

class DatabaseLogger:
    """
    Logs workflow execution events to the database.
    """
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.execution_id: Optional[uuid.UUID] = None
        self.run_id: Optional[str] = None
        self.operator_email: Optional[str] = None

    def log_run_start(self,
                      workflow_version_id: uuid.UUID,
                      operator_email: str,
                      run_metadata: Dict[str, Any] = None) -> str:
        """
        Create a new WorkflowExecution record.
        
        Returns:
            The generated run_id (string).
        """
        self.operator_email = operator_email
        self.run_id = f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        session = self.session_factory()
        try:
            execution = WorkflowExecution(
                run_id=self.run_id,
                workflow_version_id=workflow_version_id,
                operator=operator_email,
                status="running",
                started_at=datetime.now(timezone.utc),
                run_metadata=run_metadata or {}
            )
            session.add(execution)
            session.commit()
            session.refresh(execution)
            self.execution_id = execution.id

            logger.info(f"Started run {self.run_id} for workflow {workflow_version_id}")
            return self.run_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log run start: {e}")
            raise
        finally:
            session.close()

    def log_step_start(self, step_index: int, step_name: str):
        """Log the start of a step (could update execution status or add a log entry)."""
        # For now, we might just log to standard log or update a 'current_step' field if we added one.
        # The WorkflowExecution model stores 'results' which is a JSON blob. We can append to that.
        # But updating a JSON blob on every step might be heavy.
        # Let's just log to standard logger for now, and maybe update status.
        logger.info(f"[{self.run_id}] Starting step {step_index}: {step_name}")

    def log_step_complete(self, step_index: int, step_name: str, result: Any):
        """Log step completion and result."""
        session = self.session_factory()
        try:
            execution = session.get(WorkflowExecution, self.execution_id)
            if execution:
                # Append result to results JSON
                current_results = dict(execution.results) if execution.results else {}
                current_results[f"step_{step_index}_{step_name}"] = result
                execution.results = current_results
                session.commit()

                # Secure Audit Log
                try:
                    from retrofitkit.compliance.audit import write_audit_event
                    write_audit_event(
                        db=session,
                        actor_id=self.operator_email or "system",
                        event_type="STEP_COMPLETE",
                        entity_type="workflow_run",
                        entity_id=self.run_id,
                        payload={"step": step_index, "name": step_name, "result": str(result)[:100]} # Truncate result for log
                    )
                except ImportError:
                    pass # Fallback if audit module issues
        except Exception as e:
            logger.error(f"Failed to log step completion: {e}")
        finally:
            session.close()

    def log_run_complete(self, status: str, error: str = None):
        """Finalize the run record."""
        session = self.session_factory()
        try:
            execution = session.get(WorkflowExecution, self.execution_id)
            if execution:
                execution.status = status
                execution.completed_at = datetime.now(timezone.utc)
                if error:
                    execution.error_message = error
                session.commit()

                # Secure Audit Log
                try:
                    from retrofitkit.compliance.audit import write_audit_event
                    write_audit_event(
                        db=session,
                        actor_id=self.operator_email or "system",
                        event_type=f"RUN_{status.upper()}",
                        entity_type="workflow_run",
                        entity_id=self.run_id,
                        payload={"error": error} if error else {}
                    )
                except ImportError:
                    pass

                logger.info(f"Run {self.run_id} completed with status: {status}")
        except Exception as e:
            logger.error(f"Failed to log run completion: {e}")
        finally:
            session.close()

    def log_audit_event(self, event: str, details: str):
        """Log a secure audit event linked to this run."""
        # This should ideally use the centralized AuditLogger to ensure chain of custody
        # For now, we'll do a direct insert if AuditLogger isn't easily injectable,
        # but let's try to use the model directly for simplicity in this phase.

        session = self.session_factory()
        try:
            audit = AuditEvent(
                ts=time.time(),
                event=event,
                actor=self.operator_email or "system",
                subject=self.run_id or "unknown",
                details=details,
                hash="pending", # Placeholder, real system needs hash chain logic
                prev_hash="pending"
            )
            session.add(audit)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
        finally:
            session.close()
