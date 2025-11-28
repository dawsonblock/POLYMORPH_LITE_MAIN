"""
Inventory management models for stock tracking.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Date, DateTime, ForeignKey, JSON, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from retrofitkit.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utc_today():
    return datetime.now(timezone.utc).date()


class Vendor(Base):
    """Vendor/supplier information."""
    __tablename__ = 'vendors'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vendor_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    contact_info = Column(JSON, default=dict)  # email, phone, address, etc.

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    stock_lots = relationship("StockLot", back_populates="vendor")


class InventoryItem(Base):
    """Inventory item master data."""
    __tablename__ = 'inventory_items'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_code = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=True, index=True)  # reagent, consumable, standard
    unit = Column(String(50), nullable=True)  # mL, g, ea, etc.

    # Stock management
    min_stock = Column(Integer, default=0)
    current_stock = Column(Integer, default=0, index=True)
    reorder_point = Column(Integer, default=0)

    # Location
    location = Column(String(255), nullable=True, index=True)

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    created_by = Column(String(255), nullable=False)

    # Relationships
    stock_lots = relationship("StockLot", back_populates="item", cascade="all, delete-orphan")


class StockLot(Base):
    """Individual stock lot with expiration tracking."""
    __tablename__ = 'stock_lots'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lot_number = Column(String(255), unique=True, nullable=False, index=True)

    # Foreign keys
    item_id = Column(UUID(as_uuid=True), ForeignKey('inventory_items.id'), nullable=False, index=True)
    vendor_id = Column(UUID(as_uuid=True), ForeignKey('vendors.id'), nullable=True, index=True)

    # Quantities
    quantity = Column(Integer, nullable=False)
    quantity_remaining = Column(Integer, nullable=False)

    # Dates
    received_date = Column(Date, default=utc_today, index=True)
    expiration_date = Column(Date, nullable=True, index=True)

    # Status
    status = Column(String(50), default='active', index=True)  # active, depleted, expired, quarantined

    extra_data = Column(JSON, default=dict)

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    item = relationship("InventoryItem", back_populates="stock_lots")
    vendor = relationship("Vendor", back_populates="stock_lots")
