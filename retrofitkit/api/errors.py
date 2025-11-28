"""
Comprehensive error handling for POLYMORPH-LITE API.

Provides:
- Custom exception classes
- Global exception handlers
- Error response models
- Middleware for error tracking
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
import traceback
from retrofitkit.core.logging_config import get_logger

log = get_logger("api.errors")


# Custom Exception Classes
class PolymorphError(Exception):
    """Base exception for all POLYMORPH errors."""
    def __init__(self, message: str, code: str = "POLYMORPH_ERROR", details: Dict[str, Any] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class DatabaseError(PolymorphError):
    """Database operation failed."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "DATABASE_ERROR", details)


class AuthenticationError(PolymorphError):
    """Authentication failed."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class AuthorizationError(PolymorphError):
    """Authorization/permission denied."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)


class APIDataValidationError(PolymorphError):
    """Data validation failed."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class DeviceError(PolymorphError):
    """Hardware device error."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "DEVICE_ERROR", details)


class WorkflowError(PolymorphError):
    """Workflow execution error."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "WORKFLOW_ERROR", details)


class SafetyError(PolymorphError):
    """Safety interlock triggered."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "SAFETY_ERROR", details)


class ConfigurationError(PolymorphError):
    """Configuration error."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


# Error Response Models
class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    path: Optional[str] = None
    timestamp: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: ErrorDetail
    request_id: Optional[str] = None
    trace_id: Optional[str] = None


# Exception Handlers
async def polymorph_error_handler(request: Request, exc: PolymorphError) -> JSONResponse:
    """Handle custom POLYMORPH errors."""
    from datetime import datetime, timezone

    log.error(
        "polymorph_error",
        code=exc.code,
        message=exc.message,
        details=exc.details,
        path=request.url.path
    )

    error_detail = ErrorDetail(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        path=str(request.url.path),
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    # Map error types to HTTP status codes
    status_map = {
        "AUTHENTICATION_ERROR": status.HTTP_401_UNAUTHORIZED,
        "AUTHORIZATION_ERROR": status.HTTP_403_FORBIDDEN,
        "VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "DATABASE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "DEVICE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "WORKFLOW_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "SAFETY_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "CONFIGURATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }

    status_code = status_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    return JSONResponse(
        status_code=status_code,
        content={"error": error_detail.model_dump()}
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    from datetime import datetime, timezone

    log.warning(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )

    error_detail = ErrorDetail(
        code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
        path=str(request.url.path),
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_detail.model_dump()}
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    from datetime import datetime, timezone

    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    log.warning(
        "validation_error",
        errors=errors,
        path=request.url.path
    )

    error_detail = ErrorDetail(
        code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"errors": errors},
        path=str(request.url.path),
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": error_detail.model_dump()}
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all uncaught exceptions."""
    from datetime import datetime, timezone
    import os

    # Get traceback
    tb = traceback.format_exc()

    log.error(
        "unhandled_exception",
        exception_type=type(exc).__name__,
        message=str(exc),
        traceback=tb,
        path=request.url.path
    )

    # In production, don't expose internal errors
    if os.getenv("ENVIRONMENT", "development") == "production":
        message = "An internal error occurred. Please contact support."
        details = None
    else:
        message = str(exc)
        details = {
            "type": type(exc).__name__,
            "traceback": tb
        }

    error_detail = ErrorDetail(
        code="INTERNAL_ERROR",
        message=message,
        details=details,
        path=str(request.url.path),
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": error_detail.model_dump()}
    )


def register_exception_handlers(app):
    """Register all exception handlers with FastAPI app."""
    from fastapi.exceptions import RequestValidationError

    app.add_exception_handler(PolymorphError, polymorph_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    log.info("exception_handlers_registered")
