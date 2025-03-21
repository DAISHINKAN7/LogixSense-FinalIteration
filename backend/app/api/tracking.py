# backend/app/api/tracking.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional

from ..utils.data_processor import DataProcessor
import os

router = APIRouter(
    prefix="/api/tracking",
    tags=["tracking"],
    responses={404: {"description": "Not found"}},
)

# Initialize data processor
data_path = os.path.join("data", "31122024045505_CELEXP_RECPT_0115082024.csv")
data_processor = DataProcessor(data_path)

@router.get("/{tracking_id}")
async def get_shipment_tracking(tracking_id: str) -> Dict[str, Any]:
    """
    Get tracking information for a shipment.
    
    Args:
        tracking_id: The tracking ID or AWB number
        
    Returns:
        Shipment tracking details
    """
    try:
        tracking_data = data_processor.get_shipment_tracking(tracking_id)
        
        if tracking_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Shipment with tracking ID {tracking_id} not found"
            )
        
        return tracking_data
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving tracking information: {str(e)}"
        )

@router.get("/status/{status}")
async def get_shipments_by_status(
    status: str,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get shipments by status.
    
    Args:
        status: The shipment status to filter by
        limit: Maximum number of results to return
        
    Returns:
        List of shipments matching the status
    """
    # In a real implementation, this would query a database
    # For the prototype, we'll return mock data
    try:
        # Mock statuses
        valid_statuses = ["in_transit", "delivered", "customs", "processing"]
        
        if status.lower() not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid options are: {', '.join(valid_statuses)}"
            )
        
        # Mock data for different statuses
        mock_shipments = []
        
        base_shipment = {
            "origin": "Delhi, India",
            "destination": "New York, USA",
            "weight": "235.5 kg",
            "service": "Express Air Freight"
        }
        
        # Generate different shipments based on status
        if status.lower() == "in_transit":
            for i in range(min(limit, 5)):
                mock_shipments.append({
                    "id": f"AWB1098{3762 + i}",
                    "status": "In Transit",
                    "current_location": "Dubai, UAE",
                    "estimated_arrival": "2025-03-18",
                    **base_shipment
                })
        elif status.lower() == "delivered":
            for i in range(min(limit, 5)):
                mock_shipments.append({
                    "id": f"AWB1098{3255 + i}",
                    "status": "Delivered",
                    "delivery_date": "2025-03-14",
                    "recipient": "John Smith",
                    **base_shipment
                })
        elif status.lower() == "customs":
            for i in range(min(limit, 3)):
                mock_shipments.append({
                    "id": f"AWB1098{3445 + i}",
                    "status": "Customs Clearance",
                    "current_location": "JFK Airport, USA",
                    "hold_reason": "Documentation Review",
                    "estimated_clearance": "2025-03-17",
                    **base_shipment
                })
        elif status.lower() == "processing":
            for i in range(min(limit, 4)):
                mock_shipments.append({
                    "id": f"AWB1098{3390 + i}",
                    "status": "Processing",
                    "current_location": "Delhi Air Cargo Terminal, India",
                    "estimated_departure": "2025-03-19",
                    **base_shipment
                })
        
        return {
            "status": status,
            "count": len(mock_shipments),
            "shipments": mock_shipments
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving shipments: {str(e)}"
        )