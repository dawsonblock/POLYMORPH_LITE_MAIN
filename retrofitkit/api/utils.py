"""
Common utilities for API endpoints.

Provides reusable functions for pagination, error handling, and validation.
"""
from typing import TypeVar, Generic, List
from pydantic import BaseModel
from sqlalchemy.orm import Query

T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response model."""
    items: List[T]
    total: int
    limit: int
    offset: int
    has_more: bool

    class Config:
        from_attributes = True


def paginate(query: Query, limit: int = 100, offset: int = 0, max_limit: int = 1000) -> dict:
    """
    Paginate a SQLAlchemy query.

    Args:
        query: SQLAlchemy query object
        limit: Number of items per page
        offset: Number of items to skip
        max_limit: Maximum allowed limit

    Returns:
        Dictionary with items, total, limit, offset, has_more
    """
    # Enforce max limit
    limit = min(limit, max_limit)

    # Get total count
    total = query.count()

    # Get paginated items
    items = query.limit(limit).offset(offset).all()

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total
    }


def validate_pagination_params(limit: int, offset: int, max_limit: int = 1000) -> tuple:
    """
    Validate and sanitize pagination parameters.

    Args:
        limit: Requested limit
        offset: Requested offset
        max_limit: Maximum allowed limit

    Returns:
        Tuple of (validated_limit, validated_offset)

    Raises:
        ValueError: If parameters are invalid
    """
    if limit < 1:
        raise ValueError("Limit must be at least 1")

    if offset < 0:
        raise ValueError("Offset must be non-negative")

    if limit > max_limit:
        raise ValueError(f"Limit cannot exceed {max_limit}")

    return min(limit, max_limit), offset
