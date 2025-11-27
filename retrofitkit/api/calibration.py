"""
Instrument Calibration API endpoints.

Provides calibration logging, certificate storage, and upcoming calibration reminders.
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel, UUID4, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta, timezone
import os
import shutil

from retrofitkit.database.models import (
    CalibrationEntry, DeviceStatus, get_session
)
from retrofitkit.compliance.audit import Audit
from retrofitkit.api.dependencies import get_current_user

router = APIRouter(prefix="/api/calibration", tags=["calibration"])

# Directory for storing calibration certificates
CERTIFICATES_DIR = os.getenv("P4_DATA_DIR", "data") + "/calibration_certificates"
os.makedirs(CERTIFICATES_DIR, exist_ok=True)

# ============================================================================
# Pydantic Models
# ============================================================================

class CalibrationCreate(BaseModel):
    device_id: str
    calibration_date: Optional[datetime] = None
    next_due_date: Optional[date] = None
    status: str = 'valid'
    results: Dict[str, Any] = {}


class CalibrationResponse(BaseModel):
    id: UUID4
    device_id: str
    calibration_date: datetime
    performed_by: str
    next_due_date: Optional[date]
    status: str
    results: Dict[str, Any]
    certificate_path: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DeviceStatusResponse(BaseModel):
    id: UUID4
    device_id: str
    status: str
    last_calibration_date: Optional[date]
    next_calibration_due: Optional[date]
    health_score: Optional[float]
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# CALIBRATION ENDPOINTS
# ============================================================================

@router.post("/", response_model=CalibrationResponse, status_code=status.HTTP_201_CREATED)
async def add_calibration_entry(
    calibration: CalibrationCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add a new calibration record."""
    session = get_session()
    audit = Audit()

    try:
        # Default calibration date to now if not provided
        calib_date = calibration.calibration_date or datetime.now(timezone.utc)

        new_entry = CalibrationEntry(
            device_id=calibration.device_id,
            calibration_date=calib_date,
            performed_by=current_user["email"],
            next_due_date=calibration.next_due_date,
            status=calibration.status,
            results=calibration.results
        )

        session.add(new_entry)

        # Update or create device status
        device_status = session.query(DeviceStatus).filter(
            DeviceStatus.device_id == calibration.device_id
        ).first()

        if device_status:
            device_status.last_calibration_date = calib_date.date()
            device_status.next_calibration_due = calibration.next_due_date
            device_status.status = 'operational' if calibration.status == 'valid' else 'maintenance'
            device_status.updated_at = datetime.now(timezone.utc)
        else:
            device_status = DeviceStatus(
                device_id=calibration.device_id,
                last_calibration_date=calib_date.date(),
                next_calibration_due=calibration.next_due_date,
                status='operational' if calibration.status == 'valid' else 'maintenance'
            )
            session.add(device_status)

        session.commit()
        session.refresh(new_entry)

        # Audit log (non-blocking)
        try:
            audit.log(
                "CALIBRATION_PERFORMED",
                current_user["email"],
                calibration.device_id,
                f"Calibration performed on {calibration.device_id}, status: {calibration.status}"
            )
        except Exception:
            pass

        return new_entry

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding calibration: {str(e)}"
        )
    finally:
        session.close()


@router.get("/device/{device_id}", response_model=List[CalibrationResponse])
async def get_device_calibration_history(device_id: str):
    """Get calibration history for a specific device."""
    session = get_session()

    try:
        calibrations = session.query(CalibrationEntry).filter(
            CalibrationEntry.device_id == device_id
        ).order_by(CalibrationEntry.calibration_date.desc()).all()

        return calibrations

    finally:
        session.close()


@router.get("/upcoming", response_model=List[Dict])
async def get_upcoming_calibrations(days: int = 30):
    """Get devices due for calibration within N days."""
    session = get_session()

    try:
        cutoff_date = date.today() + timedelta(days=days)

        upcoming = session.query(DeviceStatus).filter(
            DeviceStatus.next_calibration_due <= cutoff_date,
            DeviceStatus.next_calibration_due >= date.today()
        ).all()

        return [
            {
                "device_id": device.device_id,
                "next_calibration_due": device.next_calibration_due.isoformat(),
                "days_until_due": (device.next_calibration_due - date.today()).days,
                "last_calibration_date": device.last_calibration_date.isoformat() if device.last_calibration_date else None,
                "status": device.status
            }
            for device in upcoming
        ]

    finally:
        session.close()


@router.get("/overdue")
async def get_overdue_calibrations():
    """Get devices with overdue calibrations."""
    session = get_session()

    try:
        overdue = session.query(DeviceStatus).filter(
            DeviceStatus.next_calibration_due < date.today()
        ).all()

        return {
            "count": len(overdue),
            "devices": [
                {
                    "device_id": device.device_id,
                    "next_calibration_due": device.next_calibration_due.isoformat(),
                    "days_overdue": (date.today() - device.next_calibration_due).days,
                    "last_calibration_date": device.last_calibration_date.isoformat() if device.last_calibration_date else None,
                    "status": device.status
                }
                for device in overdue
            ]
        }

    finally:
        session.close()


@router.post("/{calibration_id}/attach-certificate")
async def attach_certificate(
    calibration_id: UUID4,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload calibration certificate (PDF or image)."""
    session = get_session()
    audit = Audit()

    try:
        # Verify calibration entry exists
        calibration = session.query(CalibrationEntry).filter(
            CalibrationEntry.id == calibration_id
        ).first()
        if not calibration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calibration entry {calibration_id} not found"
            )

        # Save file
        file_extension = os.path.splitext(file.filename)[1]
        safe_filename = f"{calibration.device_id}_{calibration.calibration_date.strftime('%Y%m%d')}_{calibration_id}{file_extension}"
        file_path = os.path.join(CERTIFICATES_DIR, safe_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Update calibration entry
        calibration.certificate_path = file_path
        session.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "CALIBRATION_CERTIFICATE_ATTACHED",
                current_user["email"],
                str(calibration_id),
                f"Attached certificate {safe_filename} to calibration {calibration_id}"
            )
        except Exception:
            pass

        return {
            "message": "Certificate attached successfully",
            "file_path": file_path
        }

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error attaching certificate: {str(e)}"
        )
    finally:
        session.close()


@router.get("/{calibration_id}", response_model=CalibrationResponse)
async def get_calibration_entry(calibration_id: UUID4):
    """Get specific calibration entry details."""
    session = get_session()

    try:
        calibration = session.query(CalibrationEntry).filter(
            CalibrationEntry.id == calibration_id
        ).first()
        if not calibration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calibration entry {calibration_id} not found"
            )
        return calibration

    finally:
        session.close()


# ============================================================================
# DEVICE STATUS ENDPOINTS
# ============================================================================

@router.get("/status/{device_id}", response_model=DeviceStatusResponse)
async def get_device_status(device_id: str):
    """Get current status of a device."""
    session = get_session()

    try:
        device_status = session.query(DeviceStatus).filter(
            DeviceStatus.device_id == device_id
        ).first()
        if not device_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No status found for device '{device_id}'"
            )
        return device_status

    finally:
        session.close()


@router.get("/status", response_model=List[DeviceStatusResponse])
async def list_device_statuses(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List all device statuses."""
    session = get_session()

    try:
        query = session.query(DeviceStatus)

        if status:
            query = query.filter(DeviceStatus.status == status)

        devices = query.limit(limit).offset(offset).all()
        return devices

    finally:
        session.close()


@router.put("/status/{device_id}")
async def update_device_status(
    device_id: str,
    new_status: str,
    health_score: Optional[float] = None,
    current_user: dict = Depends(get_current_user)
):
    """Update device operational status."""
    session = get_session()
    audit = Audit()

    try:
        device_status = session.query(DeviceStatus).filter(
            DeviceStatus.device_id == device_id
        ).first()

        if not device_status:
            # Create new status entry
            device_status = DeviceStatus(
                device_id=device_id,
                status=new_status,
                health_score=health_score
            )
            session.add(device_status)
        else:
            device_status.status = new_status
            if health_score is not None:
                device_status.health_score = health_score
            device_status.updated_at = datetime.now(timezone.utc)

        session.commit()

        # Audit log (non-blocking)
        try:
            audit.log(
                "DEVICE_STATUS_UPDATED",
                current_user["email"],
                device_id,
                f"Device {device_id} status updated to {new_status}"
            )
        except Exception:
            pass

        return {
            "message": f"Device status updated to {new_status}",
            "device_id": device_id
        }

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating device status: {str(e)}"
        )
    finally:
        session.close()
