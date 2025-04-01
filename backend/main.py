# # backend/main.py
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Any, Optional

# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta

# from app.utils.data_processor import DataProcessor
# from app.api import tracking
# from app.api import risk  # Import the risk router
# from app.api import global_shipping  # Import the global shipping router

# app = FastAPI(
#     title="LogixSense API",
#     description="Backend API for LogixSense Logistics Analytics Platform",
#     version="1.0.0"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize data processor
# data_path = os.path.join("data", "31122024045505_CELEXP_RECPT_0115082024.csv")
# data_processor = None

# @app.on_event("startup")
# async def startup_db_client():
#     global data_processor
#     try:
#         data_processor = DataProcessor(data_path)
#         print(f"Successfully initialized data processor")
        
#         # Pass the data processor to the routers
#         risk.set_data_processor(data_processor)
#         global_shipping.set_data_processor(data_processor)
        
#     except Exception as e:
#         print(f"Error initializing data processor: {e}")

# @app.get("/")
# async def root():
#     return {"message": "Welcome to LogixSense API", "status": "online"}

# @app.get("/api/statistics/summary")
# async def get_summary_statistics():
#     """
#     Get summary statistics for the dashboard
#     """
#     try:
#         if data_processor is None:
#             raise HTTPException(status_code=500, detail="Data processor not initialized")
        
#         return data_processor.get_summary_statistics()
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/api/analytics/destinations")
# async def get_destination_analysis():
#     """
#     Get shipment volume by destination
#     """
#     try:
#         if data_processor is None:
#             raise HTTPException(status_code=500, detail="Data processor not initialized")
        
#         return data_processor.get_destination_analysis()
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/api/analytics/weight-distribution")
# async def get_weight_distribution():
#     """
#     Get shipment weight distribution
#     """
#     try:
#         if data_processor is None:
#             raise HTTPException(status_code=500, detail="Data processor not initialized")
        
#         return data_processor.get_weight_distribution()
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/api/analytics/monthly-trends")
# async def get_monthly_trends():
#     """
#     Get shipment and weight trends by month
#     """
#     try:
#         if data_processor is None:
#             raise HTTPException(status_code=500, detail="Data processor not initialized")
        
#         return data_processor.get_monthly_trends()
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/api/analytics/commodity-breakdown")
# async def get_commodity_breakdown():
#     """
#     Get breakdown of shipments by commodity type
#     """
#     try:
#         if data_processor is None:
#             raise HTTPException(status_code=500, detail="Data processor not initialized")
        
#         return data_processor.get_commodity_breakdown()
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/api/shipments/recent")
# async def get_recent_shipments(limit: int = 5):
#     """
#     Get recent shipments for the dashboard
#     """
#     try:
#         # For the prototype, we'll use mock data
#         result = [
#             {
#                 'id': f'AWB10983762',
#                 'destination': 'New York, USA',
#                 'status': 'In Transit',
#                 'weight': '342.5 kg',
#                 'value': '₹ 284,500'
#             },
#             {
#                 'id': f'AWB10983571',
#                 'destination': 'Dubai, UAE',
#                 'status': 'Delivered',
#                 'weight': '128.3 kg',
#                 'value': '₹ 95,200'
#             },
#             {
#                 'id': f'AWB10983445',
#                 'destination': 'Singapore',
#                 'status': 'Customs Clearance',
#                 'weight': '205.0 kg',
#                 'value': '₹ 173,800'
#             },
#             {
#                 'id': f'AWB10983390',
#                 'destination': 'London, UK',
#                 'status': 'Processing',
#                 'weight': '178.2 kg',
#                 'value': '₹ 152,600'
#             },
#             {
#                 'id': f'AWB10983255',
#                 'destination': 'Tokyo, Japan',
#                 'status': 'Delivered',
#                 'weight': '93.7 kg',
#                 'value': '₹ 87,400'
#             }
#         ]
        
#         return result[:limit]
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/api/forecasting/demand")
# async def get_demand_forecast():
#     """
#     Generate a simple demand forecast
#     """
#     try:
#         # Generate dates for the next 6 months
#         today = datetime.now()
#         dates = [(today + timedelta(days=30*i)).strftime("%b %Y") for i in range(6)]
        
#         # Generate forecast data with some randomness
#         base_value = 1800
#         forecast = []
        
#         for i, date in enumerate(dates):
#             # Add a growth trend with some seasonality and randomness
#             value = base_value * (1 + 0.05 * i + 0.1 * np.sin(i) + 0.1 * np.random.random())
#             forecast.append({
#                 "month": date,
#                 "forecast": int(value),
#                 "lower": int(value * 0.8),
#                 "upper": int(value * 1.2)
#             })
        
#         return forecast
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/api/risk/assessment")
# async def get_risk_assessment():
#     """
#     Generate a shipment risk assessment
#     """
#     try:
#         if data_processor is None:
#             raise HTTPException(status_code=500, detail="Data processor not initialized")
        
#         return data_processor.generate_risk_assessment()
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# # Include the tracking router
# app.include_router(tracking.router)

# # Include the risk router
# app.include_router(risk.router)

# # Include the global shipping router
# app.include_router(global_shipping.router)

# # Import and include the analytics router - this should come after data_processor initialization
# from app.api import analytics
# app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.utils.data_processor import DataProcessor
from app.api import tracking
from app.api import risk  # Import the risk router
from app.api import global_shipping  # Import the global shipping router
from app.api import forecasting  # Import the forecasting router

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

@app.on_event("startup")
async def startup_db_client():
    global data_processor
    try:
        data_processor = DataProcessor(data_path)
        print(f"Successfully initialized data processor")
        
        # Pass the data processor to the routers
        risk.set_data_processor(data_processor)
        global_shipping.set_data_processor(data_processor)
        forecasting.set_data_processor(data_processor)  # Initialize forecasting models
        
    except Exception as e:
        print(f"Error initializing data processor: {e}")

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
    Get recent shipments for the dashboard
    """
    try:
        # For the prototype, we'll use mock data
        result = [
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
            },
            {
                'id': f'AWB10983445',
                'destination': 'Singapore',
                'status': 'Customs Clearance',
                'weight': '205.0 kg',
                'value': '₹ 173,800'
            },
            {
                'id': f'AWB10983390',
                'destination': 'London, UK',
                'status': 'Processing',
                'weight': '178.2 kg',
                'value': '₹ 152,600'
            },
            {
                'id': f'AWB10983255',
                'destination': 'Tokyo, Japan',
                'status': 'Delivered',
                'weight': '93.7 kg',
                'value': '₹ 87,400'
            }
        ]
        
        return result[:limit]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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

# Import and include the analytics router - this should come after data_processor initialization
from app.api import analytics
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)