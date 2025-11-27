"""
Inventory Management API endpoints.

Provides CRUD operations for inventory items, stock lots, and vendors.
Includes low-stock alerts and expiration tracking.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta

from retrofitkit.database.models import (
    InventoryItem, StockLot, Vendor, get_session
)
from retrofitkit.compliance.audit import Audit
from retrofitkit.compliance.tokens import get_current_user

router = APIRouter(prefix="/api/inventory", tags=["inventory"])

# ============================================================================
# Pydantic Models
# ============================================================================

class InventoryItemCreate(BaseModel):
    item_code: str
    name: str
    category: Optional[str] = None
    unit: Optional[str] = None
    min_stock: int = 0
    reorder_point: int = 0
    location: Optional[str] = None


class InventoryItemResponse(BaseModel):
    id: UUID4
    item_code: str
    name: str
    category: Optional[str]
    unit: Optional[str]
    min_stock: int
    current_stock: int
    reorder_point: int
    location: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class StockLotCreate(BaseModel):
    lot_number: str
    item_code: str  # Will lookup item_id from this
    vendor_id: Optional[UUID4] = None
    quantity: int
    expiration_date: Optional[date] = None


class StockLotResponse(BaseModel):
    id: UUID4
    lot_number: str
    item_id: UUID4
    vendor_id: Optional[UUID4]
    quantity: int
    quantity_remaining: int
    received_date: date
    expiration_date: Optional[date]
    status: str

    class Config:
        from_attributes = True


class VendorCreate(BaseModel):
    vendor_id: str
    name: str
    contact_info: Dict[str, Any] = {}


class VendorResponse(BaseModel):
    id: UUID4
    vendor_id: str
    name: str
    contact_info: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# INVENTORY ITEM ENDPOINTS
# ============================================================================

@router.post("/items", response_model=InventoryItemResponse, status_code=status.HTTP_201_CREATED)
async def create_inventory_item(
    item: InventoryItemCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new inventory item."""
    session = get_session()
    audit = Audit()

    try:
        existing = session.query(InventoryItem).filter(
            InventoryItem.item_code == item.item_code
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Item '{item.item_code}' already exists"
            )

        new_item = InventoryItem(
            item_code=item.item_code,
            name=item.name,
            category=item.category,
            unit=item.unit,
            min_stock=item.min_stock,
            reorder_point=item.reorder_point,
            location=item.location,
            created_by=current_user["email"]
        )

        session.add(new_item)
        session.commit()
        session.refresh(new_item)

        audit.log(
            "INVENTORY_ITEM_CREATED",
            current_user["email"],
            item.item_code,
            f"Created inventory item {item.item_code}"
        )

        return new_item

    finally:
        session.close()


@router.get("/items", response_model=List[InventoryItemResponse])
async def list_inventory_items(
    category: Optional[str] = None,
    low_stock_only: bool = False,
    limit: int = 100,
    offset: int = 0
):
    """List inventory items with optional filtering."""
    session = get_session()

    try:
        query = session.query(InventoryItem)

        if category:
            query = query.filter(InventoryItem.category == category)

        if low_stock_only:
            query = query.filter(InventoryItem.current_stock < InventoryItem.reorder_point)

        items = query.order_by(InventoryItem.name).limit(limit).offset(offset).all()
        return items

    finally:
        session.close()


@router.get("/items/{item_code}", response_model=InventoryItemResponse)
async def get_inventory_item(item_code: str):
    """Get inventory item details."""
    session = get_session()

    try:
        item = session.query(InventoryItem).filter(
            InventoryItem.item_code == item_code
        ).first()
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item '{item_code}' not found"
            )
        return item

    finally:
        session.close()


@router.get("/alerts/low-stock")
async def get_low_stock_alerts():
    """Get items below reorder point."""
    session = get_session()

    try:
        low_stock_items = session.query(InventoryItem).filter(
            InventoryItem.current_stock < InventoryItem.reorder_point
        ).all()

        return {
            "count": len(low_stock_items),
            "items": [
                {
                    "item_code": item.item_code,
                    "name": item.name,
                    "current_stock": item.current_stock,
                    "reorder_point": item.reorder_point,
                    "shortage": item.reorder_point - item.current_stock
                }
                for item in low_stock_items
            ]
        }

    finally:
        session.close()


@router.get("/alerts/expiring")
async def get_expiring_lots(days: int = 30):
    """Get stock lots expiring within N days."""
    session = get_session()

    try:
        cutoff_date = date.today() + timedelta(days=days)

        expiring_lots = session.query(StockLot).filter(
            StockLot.expiration_date <= cutoff_date,
            StockLot.expiration_date >= date.today(),
            StockLot.status == 'active'
        ).all()

        return {
            "count": len(expiring_lots),
            "lots": [
                {
                    "lot_number": lot.lot_number,
                    "item_id": str(lot.item_id),
                    "expiration_date": lot.expiration_date.isoformat(),
                    "days_until_expiry": (lot.expiration_date - date.today()).days,
                    "quantity_remaining": lot.quantity_remaining
                }
                for lot in expiring_lots
            ]
        }

    finally:
        session.close()


# ============================================================================
# STOCK LOT ENDPOINTS
# ============================================================================

@router.post("/lots", response_model=StockLotResponse, status_code=status.HTTP_201_CREATED)
async def create_stock_lot(
    lot: StockLotCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add a new stock lot."""
    session = get_session()
    audit = Audit()

    try:
        # Lookup item
        item = session.query(InventoryItem).filter(
            InventoryItem.item_code == lot.item_code
        ).first()
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item '{lot.item_code}' not found"
            )

        # Check if lot number already exists
        existing = session.query(StockLot).filter(
            StockLot.lot_number == lot.lot_number
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Lot '{lot.lot_number}' already exists"
            )

        new_lot = StockLot(
            lot_number=lot.lot_number,
            item_id=item.id,
            vendor_id=lot.vendor_id,
            quantity=lot.quantity,
            quantity_remaining=lot.quantity,
            expiration_date=lot.expiration_date
        )

        session.add(new_lot)

        # Update item stock count
        item.current_stock += lot.quantity

        session.commit()
        session.refresh(new_lot)

        audit.log(
            "STOCK_LOT_RECEIVED",
            current_user["email"],
            lot.lot_number,
            f"Received lot {lot.lot_number} for item {lot.item_code}, quantity: {lot.quantity}"
        )

        return new_lot

    finally:
        session.close()


@router.get("/lots", response_model=List[StockLotResponse])
async def list_stock_lots(
    item_code: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List stock lots."""
    session = get_session()

    try:
        query = session.query(StockLot)

        if item_code:
            item = session.query(InventoryItem).filter(
                InventoryItem.item_code == item_code
            ).first()
            if item:
                query = query.filter(StockLot.item_id == item.id)

        if status:
            query = query.filter(StockLot.status == status)

        lots = query.order_by(StockLot.received_date.desc()).limit(limit).offset(offset).all()
        return lots

    finally:
        session.close()


@router.post("/lots/{lot_number}/consume")
async def consume_stock(
    lot_number: str,
    quantity: int,
    current_user: dict = Depends(get_current_user)
):
    """Consume stock from a lot."""
    session = get_session()
    audit = Audit()

    try:
        lot = session.query(StockLot).filter(StockLot.lot_number == lot_number).first()
        if not lot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lot '{lot_number}' not found"
            )

        if lot.quantity_remaining < quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient stock. Available: {lot.quantity_remaining}, Requested: {quantity}"
            )

        # Update lot
        lot.quantity_remaining -= quantity
        if lot.quantity_remaining == 0:
            lot.status = 'depleted'

        # Update item stock
        item = session.query(InventoryItem).filter(InventoryItem.id == lot.item_id).first()
        if item:
            item.current_stock -= quantity

        session.commit()

        audit.log(
            "STOCK_CONSUMED",
            current_user["email"],
            lot_number,
            f"Consumed {quantity} units from lot {lot_number}"
        )

        return {
            "message": f"Consumed {quantity} units",
            "remaining": lot.quantity_remaining
        }

    finally:
        session.close()


# ============================================================================
# VENDOR ENDPOINTS
# ============================================================================

@router.post("/vendors", response_model=VendorResponse, status_code=status.HTTP_201_CREATED)
async def create_vendor(
    vendor: VendorCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new vendor."""
    session = get_session()
    audit = Audit()

    try:
        existing = session.query(Vendor).filter(Vendor.vendor_id == vendor.vendor_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Vendor '{vendor.vendor_id}' already exists"
            )

        new_vendor = Vendor(
            vendor_id=vendor.vendor_id,
            name=vendor.name,
            contact_info=vendor.contact_info
        )

        session.add(new_vendor)
        session.commit()
        session.refresh(new_vendor)

        audit.log(
            "VENDOR_CREATED",
            current_user["email"],
            vendor.vendor_id,
            f"Created vendor {vendor.vendor_id}"
        )

        return new_vendor

    finally:
        session.close()


@router.get("/vendors", response_model=List[VendorResponse])
async def list_vendors(limit: int = 100, offset: int = 0):
    """List all vendors."""
    session = get_session()

    try:
        vendors = session.query(Vendor).order_by(Vendor.name).limit(limit).offset(offset).all()
        return vendors

    finally:
        session.close()


@router.get("/vendors/{vendor_id}", response_model=VendorResponse)
async def get_vendor(vendor_id: str):
    """Get vendor details."""
    session = get_session()

    try:
        vendor = session.query(Vendor).filter(Vendor.vendor_id == vendor_id).first()
        if not vendor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vendor '{vendor_id}' not found"
            )
        return vendor

    finally:
        session.close()
