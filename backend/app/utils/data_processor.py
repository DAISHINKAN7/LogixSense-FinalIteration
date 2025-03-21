# backend/app/utils/data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os

class DataProcessor:
    """Utility class for processing logistics data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the DataProcessor with the path to the CSV data.
        
        Args:
            data_path: Path to the CSV file containing logistics data
        """
        self.data_path = data_path
        self.df = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and preprocess the CSV data."""
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Clean column names
            self.df.columns = self.df.columns.str.strip()
            
            # Convert date columns to datetime
            date_columns = ['TDG_DT', 'BAG_DT', 'SB_DT', 'FLT_DT']
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            # Clean weight columns and convert to numeric
            weight_columns = ['GRSS_WGHT', 'ACTL_CHRGBL_WGHT']
            for col in weight_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            print(f"Successfully loaded dataset with {len(self.df)} records")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Create an empty dataframe with expected columns as fallback
            self.df = pd.DataFrame()
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics from the logistics data.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.df is None or self.df.empty:
            return {
                "activeShipments": 0,
                "totalWeight": 0,
                "avgDeliveryTime": 0,
                "topDestinations": []
            }
        
        # Calculate basic statistics
        total_shipments = len(self.df)
        total_weight = self.df['GRSS_WGHT'].sum() if 'GRSS_WGHT' in self.df.columns else 0
        
        # Get top destinations
        top_destinations = []
        if 'DSTNTN' in self.df.columns:
            top_destinations = (
                self.df['DSTNTN']
                .value_counts()
                .reset_index()
                .rename(columns={'index': 'destination', 'DSTNTN': 'count'})
                .head(5)
                .to_dict('records')
            )
        
        # Average delivery time calculation would require actual delivery data
        # Using a placeholder for the prototype
        avg_delivery_time = 4.5
        
        return {
            "activeShipments": total_shipments,
            "totalWeight": float(total_weight),
            "avgDeliveryTime": avg_delivery_time,
            "topDestinations": top_destinations
        }
    
    def get_destination_analysis(self) -> List[Dict[str, Any]]:
        """
        Analyze shipment volumes by destination.
        
        Returns:
            List of dictionaries with destination analysis
        """
        if self.df is None or self.df.empty or 'DSTNTN' not in self.df.columns:
            return []
        
        # Group by destination and count shipments
        destination_counts = (
            self.df['DSTNTN']
            .value_counts()
            .reset_index()
            .rename(columns={'index': 'name', 'DSTNTN': 'value'})
            .head(10)
        )
        
        # Calculate "Others" category for remaining destinations
        total = self.df['DSTNTN'].count()
        top_total = destination_counts['value'].sum()
        others = total - top_total
        
        if others > 0:
            others_row = pd.DataFrame([{'name': 'Others', 'value': others}])
            destination_counts = pd.concat([destination_counts, others_row])
        
        return destination_counts.to_dict('records')
    
    def get_weight_distribution(self) -> List[Dict[str, Any]]:
        """
        Analyze shipment weight distribution.
        
        Returns:
            List of dictionaries with weight distribution
        """
        if self.df is None or self.df.empty or 'GRSS_WGHT' not in self.df.columns:
            return []
        
        # Define weight ranges
        weight_bins = [0, 50, 200, 500, float('inf')]
        weight_labels = ['0-50 kg', '51-200 kg', '201-500 kg', '501+ kg']
        
        # Create weight categories
        df_with_bins = self.df.copy()
        df_with_bins['weight_category'] = pd.cut(
            df_with_bins['GRSS_WGHT'], 
            bins=weight_bins, 
            labels=weight_labels, 
            right=False
        )
        
        # Count shipments in each weight category
        weight_distribution = (
            df_with_bins['weight_category']
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .reset_index()
            .rename(columns={'index': 'name', 'weight_category': 'value'})
            .sort_values('value', ascending=False)
        )
        
        return weight_distribution.to_dict('records')
    
    def get_monthly_trends(self) -> List[Dict[str, Any]]:
        """
        Calculate monthly shipment and weight trends.
        
        Returns:
            List of dictionaries with monthly trends
        """
        if self.df is None or self.df.empty or 'BAG_DT' not in self.df.columns:
            return []
        
        # Filter out rows with missing dates
        date_df = self.df.dropna(subset=['BAG_DT'])
        
        # Add month column for grouping
        date_df['month'] = date_df['BAG_DT'].dt.strftime('%b')
        date_df['month_num'] = date_df['BAG_DT'].dt.month
        
        # Group by month and calculate metrics
        monthly_data = (
            date_df.groupby(['month', 'month_num'])
            .agg(
                shipments=('AWB_NO', 'count'),
                weight=('GRSS_WGHT', 'sum')
            )
            .reset_index()
            .sort_values('month_num')
        )
        
        # Format the response
        result = monthly_data.apply(
            lambda x: {
                'name': x['month'],
                'shipments': int(x['shipments']),
                'weight': float(x['weight'])
            }, 
            axis=1
        ).tolist()
        
        return result
    
    def get_commodity_breakdown(self) -> List[Dict[str, Any]]:
        """
        Analyze shipments by commodity type.
        
        Returns:
            List of dictionaries with commodity breakdown
        """
        if self.df is None or self.df.empty or 'COMM_DESC' not in self.df.columns:
            return []
        
        # Group by commodity description and count shipments
        commodity_counts = (
            self.df['COMM_DESC']
            .fillna('Unknown')
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .reset_index()
            .rename(columns={'index': 'name', 'COMM_DESC': 'value'})
            .sort_values('value', ascending=False)
            .head(5)
        )
        
        # Calculate "Others" category for remaining commodities
        top_total = commodity_counts['value'].sum()
        others = 100 - top_total
        
        if others > 0:
            others_row = pd.DataFrame([{'name': 'Others', 'value': others}])
            commodity_counts = pd.concat([commodity_counts, others_row])
        
        return commodity_counts.to_dict('records')
    
    def generate_risk_assessment(self) -> Dict[str, Any]:
        """
        Generate a risk assessment based on shipment data.
        
        Returns:
            Dictionary containing risk assessment data
        """
        if self.df is None or self.df.empty:
            return {
                "overallRisk": "Medium",
                "riskFactors": [],
                "riskByRegion": []
            }
        
        # In a real implementation, this would use ML models
        # For the prototype, we'll create synthetic risk data based on actual regions
        
        # Get unique regions/countries
        regions = []
        if 'CONSGN_COUNTRY' in self.df.columns:
            regions = self.df['CONSGN_COUNTRY'].dropna().unique()
        
        # Create risk scores by region
        region_risks = []
        for region in regions[:10]:  # Limit to top 10 regions
            region_risks.append({
                "region": region,
                "riskScore": round(20 + 60 * np.random.random(), 1),
                "shipmentCount": int(self.df[self.df['CONSGN_COUNTRY'] == region].shape[0])
            })
        
        # Sort by risk score
        region_risks.sort(key=lambda x: x["riskScore"], reverse=True)
        
        # Generate risk factors
        risk_factors = [
            {"factor": "Weather Disruptions", "impact": "High", "probability": "Medium"},
            {"factor": "Customs Delays", "impact": "Medium", "probability": "High"},
            {"factor": "Transportation Strikes", "impact": "High", "probability": "Low"},
            {"factor": "Capacity Constraints", "impact": "Medium", "probability": "Medium"},
        ]
        
        return {
            "overallRisk": "Medium",
            "riskFactors": risk_factors,
            "riskByRegion": region_risks
        }
    
    def get_shipment_tracking(self, tracking_id: str) -> Optional[Dict[str, Any]]:
        """
        Get shipment tracking details.
        
        Args:
            tracking_id: The tracking ID or AWB number
        
        Returns:
            Dictionary with shipment tracking details or None if not found
        """
        # In a real implementation, this would query a database
        # For the prototype, we'll return mock data
        tracking_id = str(tracking_id).strip()
        
        # Check if the tracking ID exists in our data
        if self.df is None or self.df.empty:
            return None
        
        # Convert AWB_NO to string for comparison
        self.df['AWB_STR'] = self.df['AWB_NO'].astype(str)
        
        shipment = self.df[self.df['AWB_STR'] == tracking_id]
        if len(shipment) == 0:
            return None
        
        # Get the first matching shipment
        shipment = shipment.iloc[0]
        
        # Create a mock tracking result
        origin = shipment.get('STTN_OF_ORGN', 'Delhi, India')
        destination = shipment.get('DSTNTN', 'New York, USA')
        country = shipment.get('CONSGN_COUNTRY', 'USA')
        
        # Generate random dates for the tracking timeline
        now = datetime.now()
        created_date = now - timedelta(days=3)
        received_date = created_date + timedelta(hours=15)
        departed_date = received_date + timedelta(hours=2)
        connection_date = departed_date + timedelta(hours=10)
        estimated_arrival = connection_date + timedelta(hours=30)
        
        return {
            "id": f"AWB{tracking_id}",
            "status": "In Transit",
            "origin": {
                "location": origin,
                "departureTime": departed_date.isoformat(),
                "facility": f"{origin} Air Cargo Terminal"
            },
            "destination": {
                "location": f"{destination}, {country}",
                "estimatedArrival": estimated_arrival.isoformat(),
                "facility": f"{destination} Airport Cargo Terminal"
            },
            "currentLocation": {
                "location": "Dubai, UAE",
                "timestamp": connection_date.isoformat(),
                "status": "In Transit - Connection"
            },
            "details": {
                "weight": f"{shipment.get('GRSS_WGHT', 200):.1f} kg",
                "packages": int(shipment.get('NO_OF_PKGS', 1)),
                "dimensions": "120x80x75 cm",
                "type": shipment.get('COMM_DESC', 'Commercial Goods'),
                "service": "Express Air Freight"
            },
            "timeline": [
                {"status": "Order Created", "location": origin, "timestamp": created_date.isoformat(), "isCompleted": True},
                {"status": "Package Received", "location": origin, "timestamp": received_date.isoformat(), "isCompleted": True},
                {"status": "Departed Origin", "location": origin, "timestamp": departed_date.isoformat(), "isCompleted": True},
                {"status": "Arrived at Connection", "location": "Dubai, UAE", "timestamp": connection_date.isoformat(), "isCompleted": True},
                {"status": "Departed Connection", "location": "Dubai, UAE", "timestamp": None, "isCompleted": False},
                {"status": "Customs Clearance", "location": destination, "timestamp": None, "isCompleted": False},
                {"status": "Out for Delivery", "location": destination, "timestamp": None, "isCompleted": False},
                {"status": "Delivered", "location": destination, "timestamp": None, "isCompleted": False}
            ],
            "updates": [
                {"type": "info", "message": f"Shipment has departed from {origin}", "timestamp": departed_date.isoformat()},
                {"type": "warning", "message": "Slight delay at Dubai due to weather conditions", "timestamp": (connection_date + timedelta(minutes=45)).isoformat()},
                {"type": "info", "message": "Shipment is scheduled to depart Dubai in the next 2 hours", "timestamp": (connection_date + timedelta(hours=4)).isoformat()}
            ]
        }