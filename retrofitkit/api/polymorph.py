"""
Polymorph API endpoints for detection, tracking, and reporting.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import uuid
import time
import httpx

from retrofitkit.db.session import get_db
from retrofitkit.api.dependencies import get_current_user
from retrofitkit.core.config import get_config
from retrofitkit.db.models.polymorph import PolymorphEvent, PolymorphSignature, PolymorphReport

router = APIRouter(prefix="/api/polymorph", tags=["polymorph"])


class PolymorphDetectionRequest(BaseModel):
    """Request for polymorph detection."""
    spectrum: List[float]
    wavelengths: List[float]
    metadata: Optional[Dict[str, Any]] = None


class PolymorphReportRequest(BaseModel):
    """Request for report generation."""
    event_id: str
    format: str = "json"  # "json" or "pdf"
    include_spectrum: bool = True


@router.post("/detect")
async def detect_polymorph(
    request: PolymorphDetectionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Detect polymorphs from Raman spectrum.
    
    Sends spectrum to AI service and logs detection event.
    """
    config = get_config()
    ai_url = config.ai.service_url
    
    # Call AI service
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{ai_url}/polymorph_detect",
                json={
                    "spectrum": request.spectrum,
                    "wavelengths": request.wavelengths,
                    "metadata": request.metadata
                },
                timeout=5.0
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="AI service error")
                
            detection_result = response.json()
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="AI service timeout")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"AI service failed: {str(e)}")
    
    # Log detection event to database
    event_id = str(uuid.uuid4())
    
    event = PolymorphEvent(
        event_id=event_id,
        detected_at=time.time(),
        polymorph_id=detection_result.get("polymorph_id", 0),
        polymorph_name=detection_result.get("polymorph_name", "Unknown"),
        confidence=detection_result.get("confidence", 0.0),
        model_version=detection_result.get("model_version", "1.0.0"),
        operator_email=current_user.get("email", "unknown"),
        metadata=request.metadata or {}
    )
    
    db.add(event)
    await db.flush()  # Get event ID
    
    # Store signature
    signature = PolymorphSignature(
        signature_id=str(uuid.uuid4()),
        event_id=event_id,
        polymorph_id=detection_result.get("polymorph_id", 0),
        signature_vector=detection_result.get("signature_vector", []),
        alternative_forms=detection_result.get("alternative_forms", []),
        created_at=time.time()
    )
    
    db.add(signature)
    await db.commit()
    
    # Return enriched result
    return {
        **detection_result,
        "event_id": event_id,
        "logged_at": time.time()
    }


@router.get("/events")
async def list_polymorph_events(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """List polymorph detection events."""
    
    stmt = select(PolymorphEvent)\
        .order_by(PolymorphEvent.detected_at.desc())\
        .limit(limit)\
        .offset(offset)
        
    result = await db.execute(stmt)
    events = result.scalars().all()
    
    # Count total
    count_stmt = select(func.count(PolymorphEvent.id))
    count_result = await db.execute(count_stmt)
    total = count_result.scalar_one()

    
    return {
        "events": [
            {
                "event_id": e.event_id,
                "polymorph_name": e.polymorph_name,
                "confidence": e.confidence,
                "detected_at": e.detected_at,
                "operator": e.operator_email
            }
            for e in events
        ],
        "total": total
    }


@router.get("/events/{event_id}")
async def get_polymorph_event(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get detailed information about a polymorph event."""
    
    stmt = select(PolymorphEvent).where(PolymorphEvent.event_id == event_id)
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    sig_stmt = select(PolymorphSignature).where(PolymorphSignature.event_id == event_id)
    sig_result = await db.execute(sig_stmt)
    signature = sig_result.scalar_one_or_none()
    
    return {
        "event": {
            "event_id": event.event_id,
            "polymorph_id": event.polymorph_id,
            "polymorph_name": event.polymorph_name,
            "confidence": event.confidence,
            "model_version": event.model_version,
            "detected_at": event.detected_at,
            "operator_email": event.operator_email,
            "metadata": event.metadata
        },
        "signature": {
            "signature_id": signature.signature_id,
            "signature_vector": signature.signature_vector,
            "alternative_forms": signature.alternative_forms
        } if signature else None
    }


@router.post("/report")
async def generate_polymorph_report(
    request: PolymorphReportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Generate polymorph detection report."""
    
    # Get event
    stmt = select(PolymorphEvent).where(PolymorphEvent.event_id == request.event_id)
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Call AI service for report generation
    config = get_config()
    ai_url = config.ai.service_url
    
    detection_result = {
        "polymorph_detected": True,
        "polymorph_id": event.polymorph_id,
        "polymorph_name": event.polymorph_name,
        "confidence": event.confidence,
        "model_version": event.model_version
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ai_url}/polymorph_report",
            json={
                "detection_result": detection_result,
                "format": request.format,
                "include_spectrum": request.include_spectrum
            },
            timeout=10.0
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Report generation failed")
            
        report_data = response.json()
    
    # Store report
    report_id = str(uuid.uuid4())
    report = PolymorphReport(
        report_id=report_id,
        event_id=request.event_id,
        report_format=request.format,
        report_data=str(report_data.get("data", {})),
        generated_at=time.time(),
        generated_by=current_user.get("email", "unknown")
    )
    
    db.add(report)
    await db.commit()
    
    return {
        "report_id": report_id,
        "format": request.format,
        "data": report_data.get("data"),
        "generated_at": time.time()
    }


@router.get("/statistics")
async def get_polymorph_statistics(
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get polymorph detection statistics."""
    
    total_stmt = select(func.count(PolymorphEvent.id))
    total_result = await db.execute(total_stmt)
    total_events = total_result.scalar_one()
    
    # Count by polymorph type
    # Group by
    stmt = select(
        PolymorphEvent.polymorph_name,
        func.count(PolymorphEvent.id).label('count')
    ).group_by(PolymorphEvent.polymorph_name)
    
    result = await db.execute(stmt)
    by_type = result.all()
    
    # Average confidence
    avg_stmt = select(func.avg(PolymorphEvent.confidence))
    avg_result = await db.execute(avg_stmt)
    avg_confidence = avg_result.scalar_one() or 0.0
    
    return {
        "total_detections": total_events,
        "average_confidence": float(avg_confidence),
        "by_polymorph_type": [
            {"name": row.polymorph_name, "count": row.count}
            for row in by_type
        ]
    }
