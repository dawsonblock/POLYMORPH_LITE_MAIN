"""
Analytics API endpoints for system performance metrics.
Includes mock data generation for demonstration purposes.
"""
from fastapi import APIRouter, Depends, Query
from typing import List, Dict, Any
import random
from datetime import datetime, timedelta
from retrofitkit.api.security import get_current_user

router = APIRouter(prefix="/analytics", tags=["analytics"])

def _generate_mock_trends(days: int = 30) -> List[Dict[str, Any]]:
    """Generate realistic-looking trend data for the demo."""
    data = []
    base_yield = 92.0
    now = datetime.now()
    
    for i in range(days):
        date = now - timedelta(days=days-i)
        # Add some random variance and a slight upward trend
        daily_yield = min(99.9, max(85.0, base_yield + (i * 0.1) + random.uniform(-2.0, 2.0)))
        
        # Correlate errors with lower yield
        errors = 0
        if daily_yield < 90.0:
            errors = random.randint(1, 3)
            
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "yield": round(daily_yield, 1),
            "efficiency": round(random.uniform(88.0, 96.0), 1),
            "errors": errors,
            "runs": random.randint(5, 12)
        })
    return data

@router.get("/summary")
def get_summary(user=Depends(get_current_user)):
    """Get high-level KPI summary."""
    return {
        "processEfficiency": 94.2,
        "averageRuntime": 45.5, # minutes
        "errorRate": 1.2, # percent
        "maintenanceAlerts": 0,
        "utilizationRate": 78.5,
        "qualityMetrics": {
            "yield": 96.8,
            "purity": 99.2,
            "consistency": 95.5
        }
    }

@router.get("/trends")
def get_trends(days: int = Query(30, ge=7, le=90), user=Depends(get_current_user)):
    """Get historical trend data."""
    return _generate_mock_trends(days)

@router.get("/error_distribution")
def get_error_distribution(user=Depends(get_current_user)):
    """Get breakdown of error types."""
    return [
        {"name": "Timeout", "value": 12, "color": "#f59e0b"}, # Amber
        {"name": "Safety Trip", "value": 3, "color": "#ef4444"}, # Red
        {"name": "Comms Loss", "value": 5, "color": "#3b82f6"}, # Blue
        {"name": "User Abort", "value": 8, "color": "#8b5cf6"}, # Purple
    ]
