# backend/app/api/tracking.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List

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
    try:
        # Valid statuses
        valid_statuses = ["in_transit", "delivered", "customs", "processing"]
        
        if status.lower() not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid options are: {', '.join(valid_statuses)}"
            )
        
        # Get shipments from data processor using real data
        shipments = data_processor.get_shipments_by_status(status, limit)
        
        return {
            "status": status,
            "count": len(shipments),
            "shipments": shipments
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving shipments: {str(e)}"
        )

@router.get("/recent")
async def get_recent_shipments(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent shipments tracked through the system.
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of recent shipment data
    """
    try:
        # Get combination of different status shipments
        in_transit = data_processor.get_shipments_by_status("in_transit", limit=2)
        delivered = data_processor.get_shipments_by_status("delivered", limit=2) 
        processing = data_processor.get_shipments_by_status("processing", limit=1)
        
        # Combine and limit to requested number
        recent_shipments = in_transit + delivered + processing
        return recent_shipments[:limit]
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving recent shipments: {str(e)}"
        )

@router.get("/sample-awbs")
async def get_sample_awbs(count: int = 5) -> List[str]:
    """
    Get sample AWB numbers from the dataset for testing.
    
    Args:
        count: Number of AWBs to return
        
    Returns:
        List of AWB numbers
    """
    try:
        # Hardcoded samples that we know work (as a fallback)
        hardcoded_samples = ["23258167594", "23258167605"]
        
        # If the data processor isn't initialized or has no data, return hardcoded samples
        if data_processor is None or data_processor.df is None or data_processor.df.empty:
            return hardcoded_samples[:count]
        
        # Try to get AWB numbers from the data
        if 'AWB_NO' in data_processor.df.columns:
            # Get non-null AWB numbers and convert to strings
            valid_awbs = []
            
            # Use iterrows for safer processing
            for _, row in data_processor.df.head(100).iterrows():
                if pd.notna(row['AWB_NO']):
                    try:
                        awb = str(int(row['AWB_NO']))
                        valid_awbs.append(awb)
                        # Once we have enough samples, break
                        if len(valid_awbs) >= count:
                            break
                    except:
                        continue
            
            # If we found valid AWBs, return them
            if valid_awbs:
                return valid_awbs[:count]
        
        # If we couldn't get AWBs from the data, return hardcoded samples
        return hardcoded_samples[:count]
    
    except Exception as e:
        # Log the error but don't crash
        print(f"Error getting sample AWBs: {str(e)}")
        return ["23258167594", "23258167605"]

@router.get("/available-awbs")
async def get_available_awbs(limit: int = 20) -> Dict[str, Any]:
    """
    Get a list of available AWB numbers from the dataset.
    
    Args:
        limit: Maximum number of AWBs to return
        
    Returns:
        Dictionary with list of AWB numbers and count
    """
    try:
        if data_processor is None or data_processor.df is None or data_processor.df.empty:
            return {"count": 0, "awbs": []}
        
        # A list to store AWB numbers we've verified work
        valid_awbs = []
        
        # Get a larger sample to filter through
        sample_df = data_processor.df.head(limit * 5)
        
        # Use iterrows for safer processing
        for _, row in sample_df.iterrows():
            if pd.notna(row['AWB_NO']):
                try:
                    awb = str(int(row['AWB_NO']))
                    
                    # Test if this AWB actually returns data
                    tracking_data = data_processor.get_shipment_tracking(awb)
                    if tracking_data is not None:
                        valid_awbs.append(awb)
                    
                    # Once we have enough samples, break
                    if len(valid_awbs) >= limit:
                        break
                except:
                    continue
        
        return {
            "count": len(valid_awbs),
            "awbs": valid_awbs
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving available AWBs: {str(e)}"
        )

@router.get("/awb-info")
async def get_awb_info() -> Dict[str, Any]:
    """
    Get information about AWB numbers in the dataset.
    
    Returns:
        Dictionary with information about AWB numbers
    """
    try:
        if data_processor is None or data_processor.df is None or data_processor.df.empty:
            return {"count": 0, "info": "No data available"}
        
        # Get basic stats about AWB_NO column
        total_count = len(data_processor.df)
        non_null_count = data_processor.df['AWB_NO'].notna().sum()
        
        # Get examples of different lengths
        sample_awbs = []
        
        # Convert to string and get sample of different lengths
        data_processor.df['AWB_STR'] = data_processor.df['AWB_NO'].astype(str)
        
        # Get lengths of AWB numbers
        awb_lengths = data_processor.df['AWB_STR'].str.len().value_counts().to_dict()
        
        # Get examples of each length
        examples_by_length = {}
        for length, count in awb_lengths.items():
            examples = data_processor.df[data_processor.df['AWB_STR'].str.len() == length]['AWB_STR'].head(3).tolist()
            examples_by_length[str(length)] = {
                "count": int(count),
                "examples": examples
            }
        
        return {
            "total_records": total_count,
            "non_null_awbs": int(non_null_count),
            "lengths": examples_by_length
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "message": "Error analyzing AWB numbers"
        }