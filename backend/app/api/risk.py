from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import pandas as pd 
import numpy as np 
from app.utils.data_processor import DataProcessor
from fastapi import Body

router = APIRouter(prefix="/api/risk", tags=["Risk Assessment"])

# This will be injected from the main app
data_processor = None

def set_data_processor(processor: DataProcessor):
    global data_processor
    data_processor = processor

@router.get("/assessment")
async def get_risk_assessment() -> Dict[str, Any]:
    """
    Generate a comprehensive risk assessment based on actual shipment data analysis.
    
    Returns:
        JSON object with risk assessment data including:
        - Overall risk level and score
        - Risk factors with impact and probability
        - Risk by region
        - Risk trend over time
        - Risk category breakdown
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        risk_assessment = data_processor.generate_risk_assessment()
        return risk_assessment
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating risk assessment: {str(e)}")


@router.get("/by-destination")
async def get_risk_by_destination() -> List[Dict[str, Any]]:
    """
    Get detailed risk breakdown by destination.
    
    Returns:
        List of destinations with risk scores and contributing factors
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        df = data_processor.df
        if df is None or df.empty:
            return []
        
        # Get top destinations by shipment count
        if 'DSTNTN' not in df.columns:
            return []
        
        destination_counts = df['DSTNTN'].value_counts().head(10)
        destinations = []
        
        for dest, count in destination_counts.items():
            # Get destination-specific data
            dest_df = df[df['DSTNTN'] == dest]
            
            # Calculate metrics for this destination
            risk_metrics = {}
            
            # Processing time (if available)
            processing_days = None
            if 'BAG_DT' in dest_df.columns and 'FLT_DT' in dest_df.columns:
                dest_df['bag_dt_clean'] = pd.to_datetime(dest_df['BAG_DT'], errors='coerce')
                dest_df['flt_dt_clean'] = pd.to_datetime(dest_df['FLT_DT'], errors='coerce')
                
                valid_dates = dest_df.dropna(subset=['bag_dt_clean', 'flt_dt_clean'])
                if not valid_dates.empty:
                    valid_dates['proc_days'] = (valid_dates['flt_dt_clean'] - valid_dates['bag_dt_clean']).dt.total_seconds() / (24 * 3600)
                    valid_dates = valid_dates[valid_dates['proc_days'].between(0, 30)]  # Filter unrealistic values
                    
                    if not valid_dates.empty:
                        processing_days = valid_dates['proc_days'].mean()
            
            # Weight statistics
            weights = None
            if 'GRSS_WGHT' in dest_df.columns:
                weights = pd.to_numeric(dest_df['GRSS_WGHT'], errors='coerce').dropna()
                if not weights.empty:
                    weights = weights.mean()
            
            # Value statistics
            values = None
            if 'FOB_VAL' in dest_df.columns:
                values = dest_df['FOB_VAL'].apply(lambda x: 
                    float(str(x).replace(',', '')) if isinstance(x, str) else float(x) if pd.notna(x) else 0
                )
                if not values.empty:
                    values = values.mean()
            
            # Calculate risk score based on available metrics
            risk_score = 50  # Default
            factors = []
            
            if processing_days is not None:
                proc_risk = min(90, max(30, processing_days * 10))
                factors.append({"name": "Processing Time", "score": round(proc_risk, 1)})
                risk_score = proc_risk
            
            if weights is not None:
                # Compare to overall average
                avg_weight = pd.to_numeric(df['GRSS_WGHT'], errors='coerce').dropna().mean()
                weight_ratio = weights / avg_weight if avg_weight > 0 else 1
                weight_risk = min(80, max(30, 50 * weight_ratio))
                factors.append({"name": "Shipment Weight", "score": round(weight_risk, 1)})
                
                # Adjust risk score
                risk_score = risk_score * 0.7 + weight_risk * 0.3
            
            if values is not None:
                # Compare to overall average
                df['value_clean'] = df['FOB_VAL'].apply(lambda x: 
                    float(str(x).replace(',', '')) if isinstance(x, str) else float(x) if pd.notna(x) else 0
                )
                avg_value = df['value_clean'].mean()
                value_ratio = values / avg_value if avg_value > 0 else 1
                value_risk = min(85, max(30, 50 * value_ratio))
                factors.append({"name": "Shipment Value", "score": round(value_risk, 1)})
                
                # Adjust risk score
                risk_score = risk_score * 0.6 + value_risk * 0.4
            
            destinations.append({
                "destination": dest,
                "shipmentCount": int(count),
                "riskScore": round(risk_score, 1),
                "riskLevel": "High" if risk_score >= 70 else "Medium" if risk_score >= 45 else "Low",
                "factors": factors
            })
        
        # Sort by risk score
        destinations.sort(key=lambda x: x["riskScore"], reverse=True)
        return destinations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing destination risk: {str(e)}")

@router.get("/high-risk-shipments")
async def get_high_risk_shipments(limit: int = 5):
    """
    Get high-risk shipments based on multiple risk factors.
    
    Args:
        limit: Maximum number of shipments to return
        
    Returns:
        List of high-risk shipments with risk scores and details
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        df = data_processor.df
        if df is None or df.empty:
            return []
        
        # Create a simplified version that's less likely to cause errors
        # Base risk on destination, weight, and value
        
        # Create a clean working copy
        risk_df = df.copy()
        
        # Calculate basic risk score (default medium risk)
        risk_df['risk_score'] = 50
        
        # 1. Add risk based on destination
        if 'DSTNTN' in risk_df.columns:
            # Count shipments by destination to identify high-volume destinations
            dest_counts = risk_df['DSTNTN'].value_counts()
            high_volume_dests = dest_counts.nlargest(5).index.tolist()
            
            # Mark shipments to high-volume destinations as higher risk
            risk_df.loc[risk_df['DSTNTN'].isin(high_volume_dests), 'risk_score'] += 15
        
        # 2. Add risk based on weight
        if 'GRSS_WGHT' in risk_df.columns:
            # Convert to numeric
            risk_df['weight_value'] = pd.to_numeric(risk_df['GRSS_WGHT'], errors='coerce').fillna(0)
            
            # Get 75th percentile as threshold for heavy shipments
            high_weight = risk_df['weight_value'].quantile(0.75)
            
            # Mark heavy shipments as higher risk
            risk_df.loc[risk_df['weight_value'] > high_weight, 'risk_score'] += 15
        
        # 3. Add risk based on value
        if 'FOB_VAL' in risk_df.columns:
            # Clean value field
            risk_df['value_clean'] = risk_df['FOB_VAL'].apply(lambda x: 
                float(str(x).replace(',', '')) if isinstance(x, str) else float(x) if pd.notna(x) else 0
            )
            
            # Get 75th percentile for high-value threshold
            high_value = risk_df['value_clean'].quantile(0.75)
            
            # Mark high-value shipments as higher risk
            risk_df.loc[risk_df['value_clean'] > high_value, 'risk_score'] += 20
        
        # Cap risk score at 95
        risk_df['risk_score'] = risk_df['risk_score'].clip(upper=95)
        
        # Sort by risk score and get top high-risk shipments
        high_risk_df = risk_df.sort_values('risk_score', ascending=False).head(limit)
        
        # Build response
        high_risk_shipments = []
        
        for _, row in high_risk_df.iterrows():
            # Format the data
            awb = str(row['AWB_NO']) if pd.notna(row.get('AWB_NO')) else ''
            destination = str(row['DSTNTN']) if pd.notna(row.get('DSTNTN')) else 'Unknown'
            country = str(row['CONSGN_COUNTRY']) if pd.notna(row.get('CONSGN_COUNTRY')) else ''
            risk_score = float(row['risk_score'])
            
            # Format risk factors
            risk_factors = []
            
            # Add appropriate risk factors
            if pd.notna(row.get('DSTNTN')) and row['DSTNTN'] in high_volume_dests:
                risk_factors.append("High-Volume Destination")
            
            if 'weight_value' in row and pd.notna(row['weight_value']) and row['weight_value'] > high_weight:
                risk_factors.append("Heavy Weight Shipment")
            
            if 'value_clean' in row and pd.notna(row['value_clean']) and row['value_clean'] > high_value:
                risk_factors.append("High Value Shipment")
            
            # Format weight
            weight = f"{float(row['GRSS_WGHT']):.1f} kg" if pd.notna(row.get('GRSS_WGHT')) else "Unknown"
            
            # Format value
            value = row.get('FOB_VAL', '')
            if pd.notna(value):
                if isinstance(value, str):
                    value_formatted = value
                else:
                    value_formatted = f"{float(value):,.2f}"
            else:
                value_formatted = "Unknown"
            
            # Determine risk level
            risk_level = "High" if risk_score >= 70 else "Medium"
            
            high_risk_shipments.append({
                "id": f"AWB{awb}",
                "destination": f"{destination}{', ' + country if country else ''}",
                "riskScore": round(risk_score, 1),
                "riskLevel": risk_level,
                "riskFactors": risk_factors,
                "weight": weight,
                "value": value_formatted
            })
        
        return high_risk_shipments
    
    except Exception as e:
        import traceback
        print(f"Error in high-risk-shipments: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error identifying high-risk shipments: {str(e)}")
    
@router.post("/feedback")
async def submit_risk_feedback(
    shipment_id: str = Body(...),
    risk_level: str = Body(...),
    comments: str = Body(None)
) -> Dict[str, Any]:
    """
    Submit feedback on risk assessment to improve the ML model.
    
    Args:
        shipment_id: The shipment ID (AWB number)
        risk_level: The actual risk level experienced ('Low', 'Medium', 'High')
        comments: Optional feedback comments
        
    Returns:
        Status information
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        # Check if RiskAnalyzer is available
        if not hasattr(data_processor, "risk_analyzer"):
            # In a real implementation, we would store this feedback in a database
            # For now, just return success
            return {
                "status": "success",
                "message": "Feedback received. Note: Risk analyzer not initialized yet."
            }
        
        # Pass feedback to the risk analyzer
        success = data_processor.risk_analyzer.add_feedback(
            shipment_id=shipment_id,
            actual_risk=risk_level,
            comments=comments
        )
        
        if success:
            # Try to apply learning from feedback
            data_processor.risk_analyzer.apply_feedback_learning()
            
            return {
                "status": "success",
                "message": "Thank you for your feedback. Our models will use this to improve."
            }
        else:
            return {
                "status": "warning",
                "message": "Feedback received, but shipment ID not found in our records."
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")