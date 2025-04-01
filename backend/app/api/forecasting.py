# backend/app/api/forecasting.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from enum import Enum
import os
import json
from datetime import datetime, timedelta

from app.models.forecasting import ForecastingModels

router = APIRouter(prefix="/api/forecasting", tags=["Forecasting"])

# Global reference to the forecasting models
forecasting_models = None

def set_data_processor(data_processor):
    """Set the data processor for the forecasting models."""
    global forecasting_models
    forecasting_models = ForecastingModels(data_processor)

class ForecastType(str, Enum):
    """Enum for the types of forecasts available."""
    DEMAND = "demand"
    WEIGHT = "weight"
    VALUE = "value"
    CARRIER = "carrier"
    SEASONAL = "seasonal"
    PROCESSING = "processing"

class ModelType(str, Enum):
    """Enum for the types of models available."""
    ML = "ml"
    ARIMA = "arima"
    HISTORICAL = "historical"

@router.get("/demand")
async def get_demand_forecast(model: Optional[str] = Query(None, description="Model type to use (ml, arima, historical)")):
    """
    Get demand forecast for the next 6 months.
    
    Args:
        model: Optional model type to use
    
    Returns:
        Forecast data with model information
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Get forecast data
        forecast_data = forecasting_models.get_forecast(ForecastType.DEMAND, model)
        
        return JSONResponse(content=forecast_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating demand forecast: {str(e)}")

@router.get("/weight")
async def get_weight_forecast(model: Optional[str] = Query(None, description="Model type to use (ml, arima, historical)")):
    """
    Get weight forecast for the next 6 months.
    
    Args:
        model: Optional model type to use
    
    Returns:
        Forecast data with model information
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Get forecast data
        forecast_data = forecasting_models.get_forecast(ForecastType.WEIGHT, model)
        
        return JSONResponse(content=forecast_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating weight forecast: {str(e)}")

@router.get("/value")
async def get_value_forecast(model: Optional[str] = Query(None, description="Model type to use (ml, arima, historical)")):
    """
    Get shipment value forecast for the next 6 months.
    
    Args:
        model: Optional model type to use
    
    Returns:
        Forecast data with model information
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Get forecast data
        forecast_data = forecasting_models.get_forecast(ForecastType.VALUE, model)
        
        return JSONResponse(content=forecast_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating value forecast: {str(e)}")

@router.get("/carrier")
async def get_carrier_forecast(model: Optional[str] = Query(None, description="Model type to use (ml, historical)")):
    """
    Get carrier utilization forecast for the next 6 months.
    
    Args:
        model: Optional model type to use
    
    Returns:
        Forecast data with carrier distribution
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Get forecast data
        forecast_data = forecasting_models.get_forecast(ForecastType.CARRIER, model)
        
        return JSONResponse(content=forecast_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating carrier forecast: {str(e)}")

@router.get("/seasonal")
async def get_seasonal_analysis(model: Optional[str] = Query(None, description="Model type to use (arima, historical)")):
    """
    Get seasonal analysis and forecast for the next 6 months.
    
    Args:
        model: Optional model type to use
    
    Returns:
        Seasonal analysis with forecast data
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Get forecast data
        forecast_data = forecasting_models.get_forecast(ForecastType.SEASONAL, model)
        
        return JSONResponse(content=forecast_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating seasonal analysis: {str(e)}")

@router.get("/processing")
async def get_processing_time_forecast(model: Optional[str] = Query(None, description="Model type to use (ml, arima, historical)")):
    """
    Get processing time forecast for the next 6 months.
    
    Args:
        model: Optional model type to use
    
    Returns:
        Forecast data for processing time
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Get forecast data
        forecast_data = forecasting_models.get_forecast(ForecastType.PROCESSING, model)
        
        return JSONResponse(content=forecast_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating processing time forecast: {str(e)}")

@router.get("/models/{forecast_type}")
async def get_all_models(forecast_type: ForecastType):
    """
    Get all available models and their accuracies for a specific forecast type.
    
    Args:
        forecast_type: Type of forecast (demand, weight, value, carrier, seasonal, processing)
    
    Returns:
        Dictionary with model data and accuracies
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Get all models for the forecast type
        models_data = forecasting_models.get_all_models(forecast_type)
        
        return JSONResponse(content=models_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving models: {str(e)}")

@router.get("/train/{forecast_type}")
async def train_model(forecast_type: ForecastType):
    """
    Train or retrain models for a specific forecast type.
    
    Args:
        forecast_type: Type of forecast to train models for
    
    Returns:
        Status message with training results
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Train the specified model
        if forecast_type == ForecastType.DEMAND:
            result = forecasting_models.train_demand_forecast_model()
        elif forecast_type == ForecastType.WEIGHT:
            result = forecasting_models.train_weight_forecast_model()
        elif forecast_type == ForecastType.VALUE:
            result = forecasting_models.train_value_forecast_model()
        elif forecast_type == ForecastType.CARRIER:
            result = forecasting_models.train_carrier_forecast_model()
        elif forecast_type == ForecastType.SEASONAL:
            result = forecasting_models.train_seasonal_analysis_model()
        elif forecast_type == ForecastType.PROCESSING:
            result = forecasting_models.train_processing_time_model()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown forecast type: {forecast_type}")
        
        # Return success message with available models
        available_models = [model for model in result.keys() if model != 'default_model']
        
        return {
            "status": "success",
            "message": f"Successfully trained models for {forecast_type}",
            "available_models": available_models,
            "default_model": result.get('default_model')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")

@router.get("/dashboard")
async def get_forecasting_dashboard():
    """
    Get a comprehensive dashboard with forecasts for all types.
    
    Returns:
        Dictionary with forecasts for all types using their default models
    """
    try:
        if forecasting_models is None:
            raise HTTPException(status_code=500, detail="Forecasting models not initialized")
        
        # Get forecasts for all types
        demand_forecast = forecasting_models.get_forecast(ForecastType.DEMAND)
        weight_forecast = forecasting_models.get_forecast(ForecastType.WEIGHT)
        value_forecast = forecasting_models.get_forecast(ForecastType.VALUE)
        seasonal_analysis = forecasting_models.get_forecast(ForecastType.SEASONAL)
        processing_forecast = forecasting_models.get_forecast(ForecastType.PROCESSING)
        
        # Return comprehensive dashboard data
        return {
            "demand": demand_forecast,
            "weight": weight_forecast,
            "value": value_forecast,
            "seasonal": seasonal_analysis,
            "processing": processing_forecast
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")