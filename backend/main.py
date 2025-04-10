# backend/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.responses import StreamingResponse

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

from app.utils.data_processor import DataProcessor
from app.api import tracking
from app.api import risk
from app.api import global_shipping
from app.api import forecasting
from app.api import analytics

app = FastAPI(
    title="LogixSense API",
    description="Backend API for LogixSense Logistics Analytics Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data processor
data_path = os.path.join("data", "31122024045505_CELEXP_RECPT_0115082024.csv")
data_processor = None

# Modified assistant import handling
assistant = None

def load_assistant_module():
    """Load the assistant module safely with error handling"""
    try:
        from app.api import assistant as assistant_module
        print("Assistant module imported successfully")
        return assistant_module
    except SyntaxError as e:
        print(f"SyntaxError in assistant module: {e}")
        print(f"File: {e.filename}, Line: {e.lineno}, Text: {e.text}")
        print(traceback.format_exc())
        return None
    except Exception as e:
        print(f"Error loading assistant module: {str(e)}, Type: {type(e)}")
        print(traceback.format_exc())
        return None

from fastapi.routing import APIRoute

# Update your startup function to prevent creating fallback routes if assistant loaded
@app.on_event("startup")
async def startup_db_client():
    global data_processor, assistant
    try:
        data_processor = DataProcessor(data_path)
        print(f"Successfully initialized data processor")
        
        # Pass the data processor to the routers
        risk.set_data_processor(data_processor)
        global_shipping.set_data_processor(data_processor)
        forecasting.set_data_processor(data_processor)
        
        # Try to load the assistant module
        assistant = load_assistant_module()
        if assistant:
            # Clean up any existing fallback routes
            remove_existing_assistant_routes()
            
            # Initialize the AI Assistant with the data processor
            assistant.initialize_services(data_processor)
            
            # Add the assistant router
            app.include_router(assistant.router)
            print("AI Assistant services initialized successfully")
            print("Assistant router included in API routes")
            return  # Exit early to prevent creating fallback routes
        
        # Only create fallback routes if assistant failed to load
        print("WARNING: Assistant module could not be loaded - assistant functionality will be unavailable")
        create_fallback_assistant_routes()
            
    except Exception as e:
        print(f"Error initializing data processor: {e}")
        print(traceback.format_exc())
        create_fallback_assistant_routes()

# Add this function to remove existing routes
def remove_existing_assistant_routes():
    """Remove any existing assistant routes to avoid conflicts"""
    routes_to_remove = []
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/api/assistant/"):
            routes_to_remove.append(route)
    
    for route in routes_to_remove:
        app.routes.remove(route)
        print(f"Removed existing route: {route.path}")

# Update your create_fallback_assistant_routes function
def create_fallback_assistant_routes():
    """Create fallback routes if the assistant module fails to load"""
    # First remove any existing assistant routes
    remove_existing_assistant_routes()
    
    # Now add the fallback routes
    @app.get("/api/assistant/status")
    async def assistant_status_fallback():
        return {
            "status": "offline",
            "message": "Assistant module could not be loaded due to syntax errors. Please fix the assistant.py file."
        }
    
    @app.post("/api/assistant/query")
    async def assistant_query_fallback(request: Request):
        body = await request.json()  # Get the request body
        return {
            "answer": "The AI assistant is currently unavailable due to syntax errors in the assistant.py module. Please check the server logs and fix the issues.",
            "error": "Module loading error"
        }
    
    @app.get("/api/assistant/suggestions")
    async def assistant_suggestions_fallback():
        return {
            "suggestions": [
                "Fix the assistant module to enable suggestions",
                "Check the server logs for error details"
            ]
        }
    
    @app.post("/api/assistant/init-vector-index")
    async def assistant_init_vector_index_fallback():
        raise HTTPException(
            status_code=503, 
            detail="Assistant module could not be loaded. Vector index initialization is unavailable."
        )
    
    print("Fallback assistant routes created due to module loading errors")

# Create a router for dashboard endpoints
from fastapi import APIRouter

dashboard_router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"]
)

@dashboard_router.get("/metrics")
async def get_dashboard_metrics():
    """
    Get all key metrics for the dashboard in a single API call
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        # Collect data with proper error handling for each section
        
        # 1. Get basic summary statistics
        try:
            summary_stats = data_processor.get_summary_statistics()
        except Exception as e:
            print(f"Error getting summary statistics: {str(e)}")
            summary_stats = {"activeShipments": 0, "totalWeight": 0, "avgDeliveryTime": 0}
        
        # 2. Get the top destinations
        try:
            # Get destination data from the data processor
            top_destinations_raw = data_processor.get_destination_analysis()
            
            # Clean and validate the destinations data
            top_destinations = []
            for item in top_destinations_raw:
                if isinstance(item, dict) and 'name' in item and 'value' in item:
                    # Convert value to int/float if it's a string
                    value = item['value']
                    if isinstance(value, str):
                        try:
                            # Try to convert to number
                            value = float(value)
                        except ValueError:
                            # If conversion fails, skip this item
                            continue
                    
                    # Add to cleaned data
                    top_destinations.append({
                        'name': str(item['name']),
                        'value': int(value)  # Convert to integer for counts
                    })
        except Exception as e:
            print(f"Error getting destination analysis: {str(e)}")
            print(traceback.format_exc())
            
            # Generate fallback destination data
            if data_processor is not None and not data_processor.df.empty and 'DSTNTN' in data_processor.df.columns:
                try:
                    # Calculate destinations manually
                    destinations = data_processor.df['DSTNTN'].fillna('Unknown').value_counts().head(8)
                    top_destinations = [{'name': str(name), 'value': int(count)} for name, count in destinations.items()]
                except Exception:
                    top_destinations = []
            else:
                top_destinations = []
        
        # 3. Get recent shipments
        try:
            recent_shipments = await get_recent_shipments(limit=5)
        except Exception as e:
            print(f"Error getting recent shipments: {str(e)}")
            recent_shipments = []
        
        # 4. Get weight distribution
        try:
            # Try to get from the data processor
            weight_distribution_raw = data_processor.get_weight_distribution()
            
            # Clean and validate the weight distribution data
            weight_distribution = []
            for item in weight_distribution_raw:
                if isinstance(item, dict) and 'name' in item and 'value' in item:
                    # Convert value to float if it's a string
                    value = item['value']
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            # If conversion fails, skip this item
                            continue
                    
                    # Add to cleaned data
                    weight_distribution.append({
                        'name': str(item['name']),
                        'value': float(value)
                    })
                    
            # If we still have no data, calculate it manually
            if not weight_distribution and data_processor is not None and not data_processor.df.empty:
                try:
                    df = data_processor.df
                    if 'GRSS_WGHT' in df.columns:
                        # Define weight ranges
                        weight_bins = [0, 50, 200, 500, float('inf')]
                        weight_labels = ['0-50 kg', '51-200 kg', '201-500 kg', '501+ kg']
                        
                        # Convert weights to numeric
                        weights = pd.to_numeric(df['GRSS_WGHT'], errors='coerce')
                        
                        # Create weight categories
                        categories = pd.cut(
                            weights.dropna(), 
                            bins=weight_bins, 
                            labels=weight_labels, 
                            right=False
                        )
                        
                        # Calculate percentages
                        counts = categories.value_counts(normalize=True) * 100
                        
                        # Create the result
                        weight_distribution = [
                            {'name': str(name), 'value': float(value)}
                            for name, value in counts.items()
                        ]
                except Exception as e:
                    print(f"Error calculating manual weight distribution: {str(e)}")
                    weight_distribution = []
        except Exception as e:
            print(f"Error getting weight distribution: {str(e)}")
            print(traceback.format_exc())
            weight_distribution = []
            
        # Ensure we have some sample data if all else fails
        if not weight_distribution:
            weight_distribution = [
                {'name': '0-50 kg', 'value': 45.0},
                {'name': '51-200 kg', 'value': 35.0},
                {'name': '201-500 kg', 'value': 15.0},
                {'name': '501+ kg', 'value': 5.0}
            ]
        
        # 5. Get monthly trends
        try:
            monthly_trends = data_processor.get_monthly_trends()
        except Exception as e:
            print(f"Error getting monthly trends: {str(e)}")
            monthly_trends = []
        
        # 6. Get commodity breakdown - has a known issue
        try:
            # Fix the commodity breakdown calculation right here
            if data_processor is not None and not data_processor.df.empty:
                # Manual calculation to work around the error
                df = data_processor.df
                if 'COMM_DESC' in df.columns:
                    # Group by commodity description and count shipments
                    comm_counts = (
                        df['COMM_DESC']
                        .fillna('Unknown')
                        .value_counts()
                        .head(5)
                    )
                    
                    # Calculate percentages
                    total_count = len(df)
                    comm_percentages = (comm_counts / total_count * 100).round(1)
                    
                    # Convert to format needed
                    commodity_breakdown = []
                    for name, value in comm_percentages.items():
                        commodity_breakdown.append({
                            'name': name,
                            'value': float(value)
                        })
                    
                    # Add "Others" category
                    top_total = sum(item['value'] for item in commodity_breakdown)
                    others = float(100) - float(top_total)
                    
                    if others > 0:
                        commodity_breakdown.append({
                            'name': 'Others',
                            'value': float(others)
                        })
                else:
                    commodity_breakdown = []
            else:
                commodity_breakdown = []
        except Exception as e:
            print(f"Error calculating commodity breakdown: {str(e)}")
            commodity_breakdown = []
        
        # 7. Get carrier distribution
        try:
            # Calculate carrier distribution directly here to avoid calling the endpoint
            if data_processor is not None and not data_processor.df.empty:
                df = data_processor.df
                if 'ARLN_DESC' in df.columns:
                    # Group by airline description and count shipments
                    carrier_counts = (
                        df['ARLN_DESC']
                        .fillna('Unknown')
                        .value_counts()
                        .head(8)
                    )
                    
                    # Convert to the format needed
                    carrier_data = []
                    for name, value in carrier_counts.items():
                        carrier_data.append({
                            'name': name,
                            'value': int(value)
                        })
                    
                    # Calculate "Others" category for remaining carriers
                    total = len(df)
                    top_total = sum(item['value'] for item in carrier_data)
                    others = int(total) - int(top_total)
                    
                    if others > 0:
                        carrier_data.append({
                            'name': 'Others',
                            'value': int(others)
                        })
                else:
                    carrier_data = []
            else:
                carrier_data = []
        except Exception as e:
            print(f"Error calculating carrier distribution: {str(e)}")
            carrier_data = []
        
        # 8. Generate alerts from risk data
        try:
            risk_data = data_processor.generate_risk_assessment()
            
            # Check for potential NaN values in risk data
            def clean_json_value(value):
                if isinstance(value, float) and (pd.isna(value) or np.isnan(value) or np.isinf(value)):
                    return 0.0
                elif isinstance(value, dict):
                    return {k: clean_json_value(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [clean_json_value(item) for item in value]
                return value
            
            # Clean risk data
            risk_data = clean_json_value(risk_data)
            
            alerts = generate_alerts_from_risk(risk_data)
        except Exception as e:
            print(f"Error generating alerts: {str(e)}")
            print(traceback.format_exc())
            alerts = []
        
        # Calculate estimated revenue
        estimated_revenue = estimate_revenue(summary_stats)
        
        # Calculate trends
        trends = calculate_trends()
        
        # Build the response - clean any potential NaN values
        response_data = {
            "summary": {
                "activeShipments": summary_stats.get("activeShipments", 0),
                "totalWeight": summary_stats.get("totalWeight", 0),
                "avgDeliveryTime": summary_stats.get("avgDeliveryTime", 0),
                "totalRevenue": estimated_revenue,
                "trends": trends
            },
            "destinations": top_destinations,
            "recentShipments": recent_shipments,
            "weightDistribution": weight_distribution,
            "commodityBreakdown": commodity_breakdown,
            "monthlyTrends": monthly_trends,
            "carrierData": carrier_data,
            "alerts": alerts
        }
        
        # Clean the response to make it JSON compatible
        def clean_response(obj):
            if isinstance(obj, dict):
                return {k: clean_response(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_response(item) for item in obj]
            elif isinstance(obj, float):
                if pd.isna(obj) or np.isnan(obj) or np.isinf(obj):
                    return 0.0
                return obj
            return obj
        
        clean_data = clean_response(response_data)
        return clean_data
        
    except Exception as e:
        print(f"Error in dashboard metrics: {str(e)}")
        print(traceback.format_exc())
        
        # Return a minimal response instead of raising an error
        # This ensures the frontend gets something to display
        return {
            "summary": {
                "activeShipments": 0,
                "totalWeight": 0,
                "avgDeliveryTime": 0,
                "totalRevenue": "₹ 0",
                "trends": {
                    "activeShipmentsTrend": 0,
                    "weightTrend": 0,
                    "deliveryTimeTrend": 0,
                    "revenueTrend": 0
                }
            },
            "destinations": [],
            "recentShipments": [],
            "weightDistribution": [],
            "commodityBreakdown": [],
            "monthlyTrends": [],
            "carrierData": [],
            "alerts": []
        }

@dashboard_router.get("/carriers")
async def get_carrier_distribution():
    """
    Get distribution of shipments by carrier (airline)
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        # Access the dataframe directly to calculate carrier distribution
        df = data_processor.df
        
        if df is None or df.empty or 'ARLN_DESC' not in df.columns:
            return []
        
        # Group by airline description and count shipments
        carrier_counts = (
            df['ARLN_DESC']
            .fillna('Unknown')
            .value_counts()
            .head(8)
        )
        
        # Convert to the format needed
        result = []
        for name, count in carrier_counts.items():
            result.append({
                'name': str(name),
                'value': int(count)
            })
        
        # Calculate "Others" category for remaining carriers
        total = int(len(df))
        top_total = sum(item['value'] for item in result)
        others = total - top_total
        
        if others > 0:
            result.append({
                'name': 'Others', 
                'value': int(others)
            })
        
        return result
    
    except Exception as e:
        print(f"Error getting carrier distribution: {str(e)}")
        print(traceback.format_exc())
        # Return empty list instead of raising an error
        return []

# Helper functions for dashboard

def generate_alerts_from_risk(risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate alerts from risk assessment data"""
    alerts = []
    
    # Add anomaly alerts
    if risk_data.get("anomalies"):
        for anomaly in risk_data["anomalies"][:2]:  # Only take top 2
            alerts.append({
                "id": anomaly.get("id", "unknown"),
                "type": "warning",
                "message": f"Anomaly detected in shipment {anomaly.get('id')} to {anomaly.get('destination')} (Score: {anomaly.get('anomalyScore')})"
            })
    
    # Add high risk region alert
    if risk_data.get("riskByRegion") and len(risk_data["riskByRegion"]) > 0:
        highest_risk_region = risk_data["riskByRegion"][0]
        if highest_risk_region.get("riskScore", 0) > 70:
            alerts.append({
                "id": f"region-{highest_risk_region.get('region')}",
                "type": "error",
                "message": f"High risk level ({highest_risk_region.get('riskScore')}) detected for region: {highest_risk_region.get('region')}"
            })
    
    # Add overall risk alert
    if risk_data.get("overallRisk") and risk_data.get("riskScore"):
        risk_type = "error" if risk_data["overallRisk"] == "High" else \
                    "warning" if risk_data["overallRisk"] == "Medium" else "success"
        
        alerts.append({
            "id": "overall-risk",
            "type": risk_type,
            "message": f"Overall logistics risk level: {risk_data['overallRisk']} ({risk_data['riskScore']})"
        })
    
    return alerts

def estimate_revenue(summary_stats: Dict[str, Any]) -> str:
    """Estimate revenue based on shipment count and weight"""
    active_shipments = summary_stats.get("activeShipments", 0)
    total_weight = summary_stats.get("totalWeight", 0)
    
    # Calculate estimated revenue using both shipment count and weight
    base_revenue_per_shipment = 8000  # Base revenue in rupees
    revenue_per_kg = 200  # Additional revenue per kg
    
    # Calculate total estimated revenue
    total_revenue = (active_shipments * base_revenue_per_shipment) + (total_weight * revenue_per_kg)
    
    # Format as currency string with commas
    return f"₹ {total_revenue:,.2f}"

def calculate_trends() -> Dict[str, float]:
    """
    Calculate trends for dashboard metrics
    This is a placeholder that would normally calculate actual trends from historical data
    """
    return {
        "activeShipmentsTrend": 3.2,
        "weightTrend": 1.5,
        "deliveryTimeTrend": -0.3,  # Negative is good for delivery time
        "revenueTrend": 2.7
    }

# Include the dashboard router
app.include_router(dashboard_router)

@app.get("/")
async def root():
    return {"message": "Welcome to LogixSense API", "status": "online"}

@app.get("/api/statistics/summary")
async def get_summary_statistics():
    """
    Get summary statistics for the dashboard
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        return data_processor.get_summary_statistics()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/analytics/destinations")
async def get_destination_analysis():
    """
    Get shipment volume by destination
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        return data_processor.get_destination_analysis()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/analytics/weight-distribution")
async def get_weight_distribution():
    """
    Get shipment weight distribution
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        return data_processor.get_weight_distribution()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/analytics/monthly-trends")
async def get_monthly_trends():
    """
    Get shipment and weight trends by month
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        return data_processor.get_monthly_trends()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/analytics/commodity-breakdown")
async def get_commodity_breakdown():
    """
    Get breakdown of shipments by commodity type
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        return data_processor.get_commodity_breakdown()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/shipments/recent")
async def get_recent_shipments(limit: int = 5):
    """
    Get recent shipments for the dashboard using data from the data processor
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        # Get in-transit shipments
        in_transit = data_processor.get_shipments_by_status("in_transit", limit=2)
        
        # Get customs shipments
        customs = data_processor.get_shipments_by_status("customs", limit=1)
        
        # Get delivered shipments
        delivered = data_processor.get_shipments_by_status("delivered", limit=1)
        
        # Get processing shipments
        processing = data_processor.get_shipments_by_status("processing", limit=1)
        
        # Combine all shipments
        all_shipments = []
        all_shipments.extend(in_transit)
        all_shipments.extend(customs)
        all_shipments.extend(delivered)
        all_shipments.extend(processing)
        
        # Format shipments for the dashboard
        result = []
        for shipment in all_shipments[:limit]:
            # Extract AWB number for value calculation
            awb_id = shipment.get("id", "AWB0")
            awb_num = int(awb_id.replace("AWB", "").replace(" ", "")) if "AWB" in awb_id else 0
            
            # Extract weight
            weight_value = shipment.get("weight", "0 kg")
            weight_num = float(weight_value.replace(" kg", "").replace(",", ""))
            
            # Generate reasonable value based on weight and AWB
            value = f"₹ {int((weight_num * 2100) + (awb_num % 1000) * 100):,}"
            
            result.append({
                'id': shipment.get("id", "Unknown"),
                'destination': shipment.get("destination", "Unknown").replace(", Unknown", ""),
                'status': shipment.get("status", "Unknown"),
                'weight': shipment.get("weight", "0 kg"),
                'value': value
            })
        
        return result[:limit]
    
    except Exception as e:
        print(f"Error getting recent shipments: {str(e)}")
        print(traceback.format_exc())
        # Return a fallback dataset if the function fails
        return [
            {
                'id': f'AWB10983762',
                'destination': 'New York, USA',
                'status': 'In Transit',
                'weight': '342.5 kg',
                'value': '₹ 284,500'
            },
            {
                'id': f'AWB10983571',
                'destination': 'Dubai, UAE',
                'status': 'Delivered',
                'weight': '128.3 kg',
                'value': '₹ 95,200'
            }
        ][:limit]

@app.get("/api/risk/assessment")
async def get_risk_assessment():
    """
    Generate a shipment risk assessment
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        return data_processor.generate_risk_assessment()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Include the tracking router
app.include_router(tracking.router)

# Include the risk router
app.include_router(risk.router)

# Include the global shipping router
app.include_router(global_shipping.router)

# Include the forecasting router
app.include_router(forecasting.router)

# Include the analytics router
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])

# Only include the assistant router if the module was loaded successfully
if assistant:
    app.include_router(assistant.router)
    print("Assistant router included in API routes")
else:
    # Create fallback assistant routes
    @app.get("/api/assistant/status")
    async def assistant_status_fallback():
        return {
            "status": "offline",
            "message": "Assistant module could not be loaded due to syntax errors. Please fix the assistant.py file."
        }
    
    @app.post("/api/assistant/query")
    async def assistant_query_fallback(request: Request):
        return {
            "answer": "The AI assistant is currently unavailable due to syntax errors in the assistant.py module. Please check the server logs and fix the issues.",
            "error": "Module loading error"
        }
    
    print("Fallback assistant routes created due to module loading errors")

# Add a new endpoint to populate the Qdrant database
@app.post("/api/assistant/init-vector-db", status_code=201)
async def initialize_vector_database():
    """
    Initialize the vector database with data from the CSV file.
    This endpoint should be called once to set up the AI Assistant.
    """
    try:
        if data_processor is None:
            raise HTTPException(status_code=500, detail="Data processor not initialized")
        
        try:
            from app.services.qdrant_service import QdrantService
            
            # Initialize Qdrant service
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_collection = os.getenv("QDRANT_COLLECTION", "logistics_data")
            qdrant_service = QdrantService(qdrant_url, qdrant_collection)
            
            # Ingest logistics data
            success = qdrant_service.ingest_logistics_data(data_processor.df)
            
            if success:
                # Get collection info
                collection_info = qdrant_service.get_collection_info()
                return {
                    "message": "Vector database initialized successfully",
                    "collection_info": collection_info
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to initialize vector database")
                
        except ImportError as ie:
            # Handle missing dependencies gracefully
            return {
                "message": "Vector database initialization skipped - dependencies not installed",
                "details": "Install qdrant-client and sentence-transformers to enable vector database functionality",
                "error": str(ie)
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing vector database: {str(e)}")

# Add a configuration endpoint for the assistant
@app.get("/api/assistant/config")
async def get_assistant_config():
    """
    Get the current configuration of the AI Assistant
    """
    try:
        # Check if demo mode is enabled
        demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"
        
        # Check for external LLM configuration
        external_llm = {
            "enabled": os.getenv("ENABLE_EXTERNAL_LLM", "false").lower() == "true",
            "provider": os.getenv("LLM_PROVIDER", "huggingface"),
            "model_name": os.getenv("LLM_MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta"),
            "inference_url": os.getenv("LLM_INFERENCE_URL", "")
        }
        
        # Check vector database configuration
        vector_db = {
            "enabled": os.getenv("ENABLE_VECTOR_DB", "false").lower() == "true",
            "provider": "qdrant",
            "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "collection": os.getenv("QDRANT_COLLECTION", "logistics_data")
        }
        
        # Get status from assistant module if available
        assistant_status = {}
        if assistant:
            try:
                assistant_status = assistant.get_status()
            except:
                assistant_status = {"status": "error", "message": "Failed to get status from assistant module"}
        else:
            assistant_status = {"status": "offline", "message": "Assistant module not loaded due to syntax errors"}
        
        return {
            "demo_mode": demo_mode,
            "external_llm": external_llm,
            "vector_db": vector_db,
            "assistant_status": assistant_status,
            "data_processor_available": data_processor is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting assistant config: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)