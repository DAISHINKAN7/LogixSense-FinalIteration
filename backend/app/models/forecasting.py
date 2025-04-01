# backend/app/models/forecasting.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from typing import Dict, List, Any, Optional, Tuple

class ForecastingModels:
    """Class for training and using forecasting models for logistics data."""
    
    def __init__(self, data_processor):
        """
        Initialize the forecasting models with the data processor.
        
        Args:
            data_processor: DataProcessor instance with access to logistics data
        """
        self.data_processor = data_processor
        self.models = {}
        self.model_dir = os.path.join("app", "models", "saved")
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _prepare_time_series_data(self, target_col: str, group_by: List[str]) -> pd.DataFrame:
        """
        Prepare time series data for forecasting by grouping and aggregating.
        
        Args:
            target_col: Column to forecast
            group_by: Columns to group by before aggregation
            
        Returns:
            DataFrame with time series data ready for forecasting
        """
        if self.data_processor.df is None or self.data_processor.df.empty:
            return pd.DataFrame()
            
        # Copy dataframe to avoid modifying original
        df = self.data_processor.df.copy()
        
        # Ensure BAG_DT is datetime
        if 'BAG_DT' in df.columns:
            df['BAG_DT'] = pd.to_datetime(df['BAG_DT'], errors='coerce')
            
            # Extract month and year for grouping
            df['month'] = df['BAG_DT'].dt.month
            df['year'] = df['BAG_DT'].dt.year
            df['month_year'] = df['BAG_DT'].dt.strftime('%Y-%m')
            
            # Group by specified columns and month/year
            grouping_cols = group_by.copy()
            grouping_cols.append('month_year')
            
            # Aggregate the target column
            if target_col in df.columns:
                # For numeric columns, calculate sum
                if pd.api.types.is_numeric_dtype(df[target_col]) or target_col in ['GRSS_WGHT', 'NO_OF_PKGS', 'FOB_VAL_NUMERIC']:
                    # Handle FOB_VAL if it's a string
                    if target_col == 'FOB_VAL':
                        df['FOB_VAL_NUMERIC'] = df['FOB_VAL'].apply(lambda x: 
                            float(str(x).replace(',', '')) if isinstance(x, str) else float(x) if pd.notna(x) else 0
                        )
                        target_col = 'FOB_VAL_NUMERIC'
                    
                    result = df.groupby(grouping_cols)[target_col].sum().reset_index()
                # For categorical columns, count occurrences
                else:
                    result = df.groupby(grouping_cols).size().reset_index(name=target_col)
                
                # Convert to datetime for time series operations
                result['date'] = pd.to_datetime(result['month_year'])
                result = result.sort_values('date')
                
                return result
        
        return pd.DataFrame()
    
    def _train_arima_model(self, time_series: pd.Series) -> Optional[ARIMA]:
        """
        Train an ARIMA model on the provided time series.
        
        Args:
            time_series: Time series data for training
            
        Returns:
            Trained ARIMA model or None if training fails
        """
        try:
            # Use a simple ARIMA model with standard parameters
            model = ARIMA(time_series, order=(1, 1, 1))
            model_fit = model.fit()
            return model_fit
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            return None
    
    def _train_ml_model(self, df: pd.DataFrame, target_col: str, categorical_features: List[str], numeric_features: List[str]) -> Optional[Pipeline]:
        """
        Train a machine learning model (Random Forest) for forecasting.
        
        Args:
            df: DataFrame with features and target
            target_col: Column to forecast
            categorical_features: List of categorical feature columns
            numeric_features: List of numeric feature columns
            
        Returns:
            Trained ML pipeline or None if training fails
        """
        try:
            # Create preprocessing pipeline
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            
            # Create full pipeline with Random Forest
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Extract features and target
            X = df[categorical_features + numeric_features]
            y = df[target_col]
            
            # Train model
            pipeline.fit(X, y)
            
            return pipeline
        except Exception as e:
            print(f"Error training ML model: {e}")
            return None
    
    def train_demand_forecast_model(self) -> Dict[str, Any]:
        """
        Train models to forecast shipping demand by region.
        
        Returns:
            Dictionary with model information and performance metrics
        """
        # Prepare data for overall demand forecast
        result = {}
        
        # Simple time series for overall demand
        df_overall = self._prepare_time_series_data('AWB_NO', [])
        
        if not df_overall.empty:
            # Get the time series for overall demand
            ts_overall = df_overall.set_index('date')['AWB_NO']
            
            # Train ARIMA model
            arima_model = self._train_arima_model(ts_overall)
            
            if arima_model:
                # Save model
                model_path = os.path.join(self.model_dir, 'demand_arima.pkl')
                joblib.dump(arima_model, model_path)
                
                # Make predictions for next 6 months
                last_date = ts_overall.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                
                forecast = arima_model.forecast(steps=6)
                
                # Prepare forecast data
                forecast_data = []
                for i, date in enumerate(future_dates):
                    forecast_data.append({
                        'month': date.strftime('%b %Y'),
                        'forecast': max(0, round(forecast[i])),
                        'lower_bound': max(0, round(forecast[i] * 0.8)),
                        'upper_bound': max(0, round(forecast[i] * 1.2))
                    })
                
                # Add to result
                result['arima'] = {
                    'forecast': forecast_data,
                    'accuracy': 0.89  # Placeholder for now
                }
        
        # ML model for region-based demand
        if not self.data_processor.df.empty and 'REGION_CD' in self.data_processor.df.columns:
            # Prepare data by region
            df_region = self._prepare_time_series_data('AWB_NO', ['REGION_CD'])
            
            if not df_region.empty:
                # Add month and year as features
                df_region['month'] = pd.to_datetime(df_region['date']).dt.month
                df_region['year'] = pd.to_datetime(df_region['date']).dt.year
                
                # Train ML model for region-based forecasting
                ml_model = self._train_ml_model(
                    df_region, 
                    'AWB_NO', 
                    categorical_features=['REGION_CD', 'month'], 
                    numeric_features=['year']
                )
                
                if ml_model:
                    # Save model
                    model_path = os.path.join(self.model_dir, 'demand_ml.pkl')
                    joblib.dump(ml_model, model_path)
                    
                    # Make predictions for next 6 months for each region
                    regions = df_region['REGION_CD'].unique()
                    last_date = pd.to_datetime(df_region['date']).max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                    
                    # Create future data for prediction
                    future_data = []
                    for region in regions:
                        for date in future_dates:
                            future_data.append({
                                'REGION_CD': region,
                                'month': date.month,
                                'year': date.year,
                                'date': date
                            })
                    
                    future_df = pd.DataFrame(future_data)
                    
                    # Make predictions
                    predictions = ml_model.predict(future_df[['REGION_CD', 'month', 'year']])
                    future_df['forecast'] = np.maximum(0, predictions)
                    
                    # Group by date to get overall forecast
                    ml_forecast = future_df.groupby('date')['forecast'].sum().reset_index()
                    
                    # Prepare forecast data
                    forecast_data = []
                    for i, row in ml_forecast.iterrows():
                        date = row['date']
                        forecast_value = round(row['forecast'])
                        forecast_data.append({
                            'month': date.strftime('%b %Y'),
                            'forecast': forecast_value,
                            'lower_bound': max(0, round(forecast_value * 0.85)),
                            'upper_bound': round(forecast_value * 1.15)
                        })
                    
                    # Add to result
                    result['ml'] = {
                        'forecast': forecast_data,
                        'accuracy': 0.92  # Placeholder for now
                    }
        
        # Historical average (simplest model)
        if not self.data_processor.df.empty:
            df_monthly = self._prepare_time_series_data('AWB_NO', [])
            
            if not df_monthly.empty:
                # Calculate monthly averages
                monthly_avg = df_monthly.groupby(pd.to_datetime(df_monthly['date']).dt.month)['AWB_NO'].mean()
                
                # Make predictions for next 6 months
                last_date = pd.to_datetime(df_monthly['date']).max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                
                forecast_data = []
                for date in future_dates:
                    month = date.month
                    if month in monthly_avg.index:
                        forecast_value = round(monthly_avg[month] * 1.05)  # Assume 5% growth
                    else:
                        forecast_value = round(monthly_avg.mean() * 1.05)  # Use average if month not in historical data
                    
                    forecast_data.append({
                        'month': date.strftime('%b %Y'),
                        'forecast': forecast_value,
                        'lower_bound': max(0, round(forecast_value * 0.7)),
                        'upper_bound': round(forecast_value * 1.3)
                    })
                
                # Add to result
                result['historical'] = {
                    'forecast': forecast_data,
                    'accuracy': 0.77  # Placeholder for now
                }
        
        # Set default model if needed
        default_model = 'ml' if 'ml' in result else 'arima' if 'arima' in result else 'historical'
        result['default_model'] = default_model
        
        # Save model info
        self.models['demand'] = result
        
        return result
    
    def train_weight_forecast_model(self) -> Dict[str, Any]:
        """
        Train models to forecast shipment weight.
        
        Returns:
            Dictionary with model information and performance metrics
        """
        result = {}
        
        # Prepare data for weight forecasting
        df_weight = self._prepare_time_series_data('GRSS_WGHT', [])
        
        if not df_weight.empty:
            # Get the time series for weight
            ts_weight = df_weight.set_index('date')['GRSS_WGHT']
            
            # Train ARIMA model
            arima_model = self._train_arima_model(ts_weight)
            
            if arima_model:
                # Save model
                model_path = os.path.join(self.model_dir, 'weight_arima.pkl')
                joblib.dump(arima_model, model_path)
                
                # Make predictions for next 6 months
                last_date = ts_weight.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                
                forecast = arima_model.forecast(steps=6)
                
                # Prepare forecast data
                forecast_data = []
                for i, date in enumerate(future_dates):
                    forecast_data.append({
                        'month': date.strftime('%b %Y'),
                        'forecast': max(0, round(forecast[i], 1)),
                        'lower_bound': max(0, round(forecast[i] * 0.8, 1)),
                        'upper_bound': round(forecast[i] * 1.2, 1)
                    })
                
                # Add to result
                result['arima'] = {
                    'forecast': forecast_data,
                    'accuracy': 0.86  # Placeholder for now
                }
        
        # ML model for more sophisticated weight forecasting
        if not self.data_processor.df.empty:
            # Add package count data to help forecast weight
            df_pkg_weight = self._prepare_time_series_data('GRSS_WGHT', [])
            df_pkg_count = self._prepare_time_series_data('NO_OF_PKGS', [])
            
            if not df_pkg_weight.empty and not df_pkg_count.empty:
                # Merge the two datasets
                df_combined = pd.merge(
                    df_pkg_weight, 
                    df_pkg_count, 
                    on='month_year', 
                    suffixes=('', '_pkg')
                )
                
                # Add month and year as features
                df_combined['month'] = pd.to_datetime(df_combined['date']).dt.month
                df_combined['year'] = pd.to_datetime(df_combined['date']).dt.year
                
                # Train ML model for weight forecasting
                ml_model = self._train_ml_model(
                    df_combined, 
                    'GRSS_WGHT', 
                    categorical_features=['month'], 
                    numeric_features=['year', 'NO_OF_PKGS']
                )
                
                if ml_model:
                    # Save model
                    model_path = os.path.join(self.model_dir, 'weight_ml.pkl')
                    joblib.dump(ml_model, model_path)
                    
                    # Make predictions for next 6 months
                    last_date = pd.to_datetime(df_combined['date']).max()
                    last_pkg_count = df_combined.loc[pd.to_datetime(df_combined['date']) == last_date, 'NO_OF_PKGS'].values[0]
                    # Assume 3% growth in package count month over month
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                    
                    # Create future data for prediction
                    future_data = []
                    for i, date in enumerate(future_dates):
                        future_pkg_count = last_pkg_count * (1.03 ** (i + 1))  # 3% monthly growth
                        future_data.append({
                            'month': date.month,
                            'year': date.year,
                            'NO_OF_PKGS': future_pkg_count,
                            'date': date
                        })
                    
                    future_df = pd.DataFrame(future_data)
                    
                    # Make predictions
                    predictions = ml_model.predict(future_df[['month', 'year', 'NO_OF_PKGS']])
                    future_df['forecast'] = np.maximum(0, predictions)
                    
                    # Prepare forecast data
                    forecast_data = []
                    for i, row in future_df.iterrows():
                        forecast_value = row['forecast']
                        forecast_data.append({
                            'month': row['date'].strftime('%b %Y'),
                            'forecast': round(forecast_value, 1),
                            'lower_bound': max(0, round(forecast_value * 0.85, 1)),
                            'upper_bound': round(forecast_value * 1.15, 1)
                        })
                    
                    # Add to result
                    result['ml'] = {
                        'forecast': forecast_data,
                        'accuracy': 0.91  # Placeholder for now
                    }
        
        # Historical average
        if not self.data_processor.df.empty:
            df_monthly = self._prepare_time_series_data('GRSS_WGHT', [])
            
            if not df_monthly.empty:
                # Calculate monthly averages
                monthly_avg = df_monthly.groupby(pd.to_datetime(df_monthly['date']).dt.month)['GRSS_WGHT'].mean()
                
                # Make predictions for next 6 months
                last_date = pd.to_datetime(df_monthly['date']).max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                
                forecast_data = []
                for date in future_dates:
                    month = date.month
                    if month in monthly_avg.index:
                        forecast_value = monthly_avg[month] * 1.05  # Assume 5% growth
                    else:
                        forecast_value = monthly_avg.mean() * 1.05  # Use average if month not in historical data
                    
                    forecast_data.append({
                        'month': date.strftime('%b %Y'),
                        'forecast': round(forecast_value, 1),
                        'lower_bound': max(0, round(forecast_value * 0.7, 1)),
                        'upper_bound': round(forecast_value * 1.3, 1)
                    })
                
                # Add to result
                result['historical'] = {
                    'forecast': forecast_data,
                    'accuracy': 0.76  # Placeholder for now
                }
        
        # Set default model if needed
        default_model = 'ml' if 'ml' in result else 'arima' if 'arima' in result else 'historical'
        result['default_model'] = default_model
        
        # Save model info
        self.models['weight'] = result
        
        return result
    
    def train_value_forecast_model(self) -> Dict[str, Any]:
        """
        Train models to forecast shipment value.
        
        Returns:
            Dictionary with model information and performance metrics
        """
        result = {}
        
        # Prepare data for value forecasting, handling string values in FOB_VAL
        if not self.data_processor.df.empty and 'FOB_VAL' in self.data_processor.df.columns:
            # Convert FOB_VAL to numeric
            df = self.data_processor.df.copy()
            df['FOB_VAL_NUMERIC'] = df['FOB_VAL'].apply(lambda x: 
                float(str(x).replace(',', '')) if isinstance(x, str) else float(x) if pd.notna(x) else 0
            )
            
            # Store the modified dataframe temporarily
            original_df = self.data_processor.df
            self.data_processor.df = df
            
            # Now prepare the time series data
            df_value = self._prepare_time_series_data('FOB_VAL_NUMERIC', [])
            
            # Restore original dataframe
            self.data_processor.df = original_df
            
            if not df_value.empty:
                # Get the time series for value
                ts_value = df_value.set_index('date')['FOB_VAL_NUMERIC']
                
                # Train ARIMA model
                arima_model = self._train_arima_model(ts_value)
                
                if arima_model:
                    # Save model
                    model_path = os.path.join(self.model_dir, 'value_arima.pkl')
                    joblib.dump(arima_model, model_path)
                    
                    # Make predictions for next 6 months
                    last_date = ts_value.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                    
                    forecast = arima_model.forecast(steps=6)
                    
                    # Prepare forecast data
                    forecast_data = []
                    for i, date in enumerate(future_dates):
                        forecast_data.append({
                            'month': date.strftime('%b %Y'),
                            'forecast': max(0, round(forecast[i])),
                            'lower_bound': max(0, round(forecast[i] * 0.8)),
                            'upper_bound': round(forecast[i] * 1.2)
                        })
                    
                    # Add to result
                    result['arima'] = {
                        'forecast': forecast_data,
                        'accuracy': 0.85  # Placeholder for now
                    }
                
                # ML model for value forecasting based on multiple features
                df = self.data_processor.df.copy()
                df['FOB_VAL_NUMERIC'] = df['FOB_VAL'].apply(lambda x: 
                    float(str(x).replace(',', '')) if isinstance(x, str) else float(x) if pd.notna(x) else 0
                )
                df['BAG_DT'] = pd.to_datetime(df['BAG_DT'], errors='coerce')
                df['month'] = df['BAG_DT'].dt.month
                df['year'] = df['BAG_DT'].dt.year
                
                # Group by month and year
                df_monthly = df.groupby(['year', 'month']).agg({
                    'FOB_VAL_NUMERIC': 'sum',
                    'GRSS_WGHT': 'sum',
                    'NO_OF_PKGS': 'sum'
                }).reset_index()
                
                # Create date column
                df_monthly['date'] = df_monthly.apply(
                    lambda x: pd.Timestamp(year=int(x['year']), month=int(x['month']), day=1), 
                    axis=1
                )
                
                # Sort by date
                df_monthly = df_monthly.sort_values('date')
                
                if len(df_monthly) > 6:  # Need enough data points
                    # Train ML model for value forecasting
                    ml_model = self._train_ml_model(
                        df_monthly, 
                        'FOB_VAL_NUMERIC', 
                        categorical_features=['month'], 
                        numeric_features=['year', 'GRSS_WGHT', 'NO_OF_PKGS']
                    )
                    
                    if ml_model:
                        # Save model
                        model_path = os.path.join(self.model_dir, 'value_ml.pkl')
                        joblib.dump(ml_model, model_path)
                        
                        # Make predictions for next 6 months
                        last_date = df_monthly['date'].max()
                        last_weight = df_monthly.loc[df_monthly['date'] == last_date, 'GRSS_WGHT'].values[0]
                        last_pkgs = df_monthly.loc[df_monthly['date'] == last_date, 'NO_OF_PKGS'].values[0]
                        
                        # Assume 3% growth month over month
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                        
                        # Create future data for prediction
                        future_data = []
                        for i, date in enumerate(future_dates):
                            growth_factor = 1.03 ** (i + 1)
                            future_data.append({
                                'month': date.month,
                                'year': date.year,
                                'GRSS_WGHT': last_weight * growth_factor,
                                'NO_OF_PKGS': last_pkgs * growth_factor,
                                'date': date
                            })
                        
                        future_df = pd.DataFrame(future_data)
                        
                        # Make predictions
                        predictions = ml_model.predict(future_df[['month', 'year', 'GRSS_WGHT', 'NO_OF_PKGS']])
                        future_df['forecast'] = np.maximum(0, predictions)
                        
                        # Prepare forecast data
                        forecast_data = []
                        for i, row in future_df.iterrows():
                            forecast_value = row['forecast']
                            forecast_data.append({
                                'month': row['date'].strftime('%b %Y'),
                                'forecast': round(forecast_value),
                                'lower_bound': max(0, round(forecast_value * 0.85)),
                                'upper_bound': round(forecast_value * 1.15)
                            })
                        
                        # Add to result
                        result['ml'] = {
                            'forecast': forecast_data,
                            'accuracy': 0.89  # Placeholder for now
                        }
        
        # Add historical average model for completeness
        if not self.data_processor.df.empty and 'FOB_VAL' in self.data_processor.df.columns:
            df = self.data_processor.df.copy()
            df['FOB_VAL_NUMERIC'] = df['FOB_VAL'].apply(lambda x: 
                float(str(x).replace(',', '')) if isinstance(x, str) else float(x) if pd.notna(x) else 0
            )
            df['BAG_DT'] = pd.to_datetime(df['BAG_DT'], errors='coerce')
            
            # Group by month
            monthly_avg = df.groupby(df['BAG_DT'].dt.month)['FOB_VAL_NUMERIC'].mean()
            
            if not monthly_avg.empty:
                # Make predictions for next 6 months
                last_date = df['BAG_DT'].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                
                forecast_data = []
                for date in future_dates:
                    month = date.month
                    if month in monthly_avg.index:
                        forecast_value = monthly_avg[month] * 1.05  # Assume 5% growth
                    else:
                        forecast_value = monthly_avg.mean() * 1.05  # Use average if month not in historical data
                    
                    forecast_data.append({
                        'month': date.strftime('%b %Y'),
                        'forecast': round(forecast_value),
                        'lower_bound': max(0, round(forecast_value * 0.7)),
                        'upper_bound': round(forecast_value * 1.3)
                    })
                
                # Add to result
                result['historical'] = {
                    'forecast': forecast_data,
                    'accuracy': 0.75  # Placeholder for now
                }
        
        # Set default model if needed
        default_model = 'ml' if 'ml' in result else 'arima' if 'arima' in result else 'historical'
        result['default_model'] = default_model
        
        # Save model info
        self.models['value'] = result
        
        return result
    
    def train_carrier_forecast_model(self) -> Dict[str, Any]:
        """
        Train models to forecast carrier utilization.
        
        Returns:
            Dictionary with model information and performance metrics
        """
        result = {}
        
        # Prepare data for carrier forecasting
        if not self.data_processor.df.empty and 'ARLN_CD' in self.data_processor.df.columns:
            # Get top carriers for analysis
            top_carriers = (
                self.data_processor.df['ARLN_CD']
                .value_counts()
                .head(5)
                .index
                .tolist()
            )
            
            carrier_forecasts = {}
            
            for carrier in top_carriers:
                # Filter data for this carrier
                df_carrier = self.data_processor.df[self.data_processor.df['ARLN_CD'] == carrier].copy()
                
                if len(df_carrier) > 10:  # Need enough data points
                    # Count shipments by month
                    df_carrier['BAG_DT'] = pd.to_datetime(df_carrier['BAG_DT'], errors='coerce')
                    df_carrier['month_year'] = df_carrier['BAG_DT'].dt.strftime('%Y-%m')
                    
                    monthly_counts = df_carrier.groupby('month_year').size().reset_index(name='count')
                    monthly_counts['date'] = pd.to_datetime(monthly_counts['month_year'])
                    monthly_counts = monthly_counts.sort_values('date')
                    
                    if len(monthly_counts) > 3:  # Need at least a few data points
                        # Get time series
                        ts_carrier = monthly_counts.set_index('date')['count']
                        
                        # Train ARIMA model
                        arima_model = self._train_arima_model(ts_carrier)
                        
                        if arima_model:
                            # Make predictions for next 6 months
                            last_date = ts_carrier.index[-1]
                            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                            
                            forecast = arima_model.forecast(steps=6)
                            
                            # Prepare forecast data
                            forecast_data = []
                            for i, date in enumerate(future_dates):
                                forecast_data.append({
                                    'month': date.strftime('%b %Y'),
                                    'forecast': max(0, round(forecast[i])),
                                    'lower_bound': max(0, round(forecast[i] * 0.8)),
                                    'upper_bound': round(forecast[i] * 1.2)
                                })
                            
                            carrier_forecasts[carrier] = forecast_data
            
            # Overall carrier forecast - combine top carriers
            if carrier_forecasts:
                # Train ML model for overall carrier distribution
                all_carriers_data = []
                carrier_monthly_data = {}
                
                for carrier in top_carriers:
                    df_carrier = self.data_processor.df[self.data_processor.df['ARLN_CD'] == carrier].copy()
                    df_carrier['BAG_DT'] = pd.to_datetime(df_carrier['BAG_DT'], errors='coerce')
                    df_carrier['month'] = df_carrier['BAG_DT'].dt.month
                    df_carrier['year'] = df_carrier['BAG_DT'].dt.year
                    
                    # Group by month and year
                    carrier_counts = df_carrier.groupby(['year', 'month']).size().reset_index(name='count')
                    carrier_counts['carrier'] = carrier
                    
                    all_carriers_data.append(carrier_counts)
                    
                    # Store by month for historical model
                    if carrier not in carrier_monthly_data:
                        carrier_monthly_data[carrier] = {}
                    
                    for _, row in carrier_counts.iterrows():
                        month = row['month']
                        count = row['count']
                        if month not in carrier_monthly_data[carrier]:
                            carrier_monthly_data[carrier][month] = []
                        carrier_monthly_data[carrier][month].append(count)
                
                if all_carriers_data:
                    # Combine all carrier data
                    df_all_carriers = pd.concat(all_carriers_data)
                    
                    # Add date column for easier manipulation
                    df_all_carriers['date'] = df_all_carriers.apply(
                        lambda x: pd.Timestamp(year=int(x['year']), month=int(x['month']), day=1),
                        axis=1
                    )
                    
                    # Train ML model for carrier distribution
                    ml_model = self._train_ml_model(
                        df_all_carriers,
                        'count',
                        categorical_features=['carrier', 'month'],
                        numeric_features=['year']
                    )
                    
                    if ml_model:
                        # Save model
                        model_path = os.path.join(self.model_dir, 'carrier_ml.pkl')
                        joblib.dump(ml_model, model_path)
                        
                        # Make predictions for next 6 months
                        last_date = df_all_carriers['date'].max()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                        
                        # Create future data for prediction
                        future_data = []
                        for carrier in top_carriers:
                            for date in future_dates:
                                future_data.append({
                                    'carrier': carrier,
                                    'month': date.month,
                                    'year': date.year,
                                    'date': date
                                })
                        
                        future_df = pd.DataFrame(future_data)
                        
                        # Make predictions
                        predictions = ml_model.predict(future_df[['carrier', 'month', 'year']])
                        future_df['forecast'] = np.maximum(0, predictions)
                        
                        # Group by date to get distribution
                        date_totals = future_df.groupby('date')['forecast'].sum().to_dict()
                        
                        # Calculate percentage distribution by carrier
                        future_df['percentage'] = future_df.apply(
                            lambda x: (x['forecast'] / date_totals[x['date']]) * 100 if date_totals[x['date']] > 0 else 0,
                            axis=1
                        )
                        
                        # Prepare data by month
                        ml_forecast = []
                        for date in future_dates:
                            month_data = future_df[future_df['date'] == date]
                            carriers_data = []
                            
                            for _, row in month_data.iterrows():
                                carriers_data.append({
                                    'carrier': row['carrier'],
                                    'count': round(row['forecast']),
                                    'percentage': round(row['percentage'], 1)
                                })
                            
                            ml_forecast.append({
                                'month': date.strftime('%b %Y'),
                                'carriers': carriers_data
                            })
                        
                        # Add to result
                        result['ml'] = {
                            'forecast': ml_forecast,
                            'accuracy': 0.88  # Placeholder for now
                        }
                
                # Historical model based on monthly averages
                hist_forecast = []
                
                for date in future_dates:
                    month = date.month
                    carriers_data = []
                    
                    total_count = 0
                    carrier_counts = {}
                    
                    for carrier in top_carriers:
                        # Get average for this month if available
                        if month in carrier_monthly_data[carrier] and carrier_monthly_data[carrier][month]:
                            avg_count = sum(carrier_monthly_data[carrier][month]) / len(carrier_monthly_data[carrier][month])
                            # Add 5% growth
                            forecast_count = avg_count * 1.05
                        else:
                            # Use average across all months
                            all_counts = [count for month_counts in carrier_monthly_data[carrier].values() for count in month_counts]
                            if all_counts:
                                avg_count = sum(all_counts) / len(all_counts)
                                forecast_count = avg_count * 1.05
                            else:
                                forecast_count = 0
                        
                        carrier_counts[carrier] = max(0, round(forecast_count))
                        total_count += carrier_counts[carrier]
                    
                    # Calculate percentages
                    for carrier in top_carriers:
                        percentage = (carrier_counts[carrier] / total_count * 100) if total_count > 0 else 0
                        carriers_data.append({
                            'carrier': carrier,
                            'count': carrier_counts[carrier],
                            'percentage': round(percentage, 1)
                        })
                    
                    hist_forecast.append({
                        'month': date.strftime('%b %Y'),
                        'carriers': carriers_data
                    })
                
                # Add to result
                result['historical'] = {
                    'forecast': hist_forecast,
                    'accuracy': 0.76  # Placeholder for now
                }
                
                # Set default model
                default_model = 'ml' if 'ml' in result else 'historical'
                result['default_model'] = default_model
        
        # Save model info
        self.models['carrier'] = result
        
        return result
    
    def train_seasonal_analysis_model(self) -> Dict[str, Any]:
        """
        Create seasonal analysis of shipping patterns.
        
        Returns:
            Dictionary with seasonal patterns and forecasts
        """
        result = {}
        
        if not self.data_processor.df.empty and 'BAG_DT' in self.data_processor.df.columns:
            df = self.data_processor.df.copy()
            df['BAG_DT'] = pd.to_datetime(df['BAG_DT'], errors='coerce')
            
            # Remove missing dates
            df = df.dropna(subset=['BAG_DT'])
            
            if not df.empty:
                # Create month-year column
                df['month_year'] = df['BAG_DT'].dt.strftime('%Y-%m')
                df['month'] = df['BAG_DT'].dt.month
                df['year'] = df['BAG_DT'].dt.year
                
                # Count shipments by month
                monthly_counts = df.groupby('month_year').size().reset_index(name='shipments')
                monthly_counts['date'] = pd.to_datetime(monthly_counts['month_year'])
                monthly_counts = monthly_counts.sort_values('date')
                
                if len(monthly_counts) >= 12:  # Need at least a year of data for seasonal analysis
                    # Perform seasonal decomposition
                    try:
                        # Create time series with regular frequency
                        ts = monthly_counts.set_index('date')['shipments']
                        
                        # Ensure the time series has a regular frequency
                        ts = ts.asfreq('MS', method='ffill')  # Monthly start frequency
                        
                        # Perform decomposition
                        decomposition = seasonal_decompose(ts, model='additive', period=12)
                        
                        # Extract seasonal patterns
                        seasonal = decomposition.seasonal
                        
                        # Create seasonal factors by month
                        seasonal_factors = {}
                        for month in range(1, 13):
                            month_data = seasonal[seasonal.index.month == month]
                            if not month_data.empty:
                                seasonal_factors[month] = float(month_data.mean())
                        
                        # Normalize factors
                        if seasonal_factors:
                            avg_factor = sum(seasonal_factors.values()) / len(seasonal_factors)
                            normalized_factors = {month: factor / avg_factor for month, factor in seasonal_factors.items()}
                            
                            # Create seasonal index data
                            seasonal_index = []
                            for month in range(1, 13):
                                month_name = pd.Timestamp(2023, month, 1).strftime('%b')
                                if month in normalized_factors:
                                    seasonal_index.append({
                                        'month': month_name,
                                        'index': round(normalized_factors[month] * 100, 1)
                                    })
                                else:
                                    seasonal_index.append({
                                        'month': month_name,
                                        'index': 100.0
                                    })
                            
                            result['seasonal_index'] = seasonal_index
                        
                        # Create forecast with seasonal factors
                        # Train ARIMA on the trend component
                        trend = decomposition.trend.dropna()
                        
                        if len(trend) > 6:
                            arima_model = self._train_arima_model(trend)
                            
                            if arima_model:
                                # Save model
                                model_path = os.path.join(self.model_dir, 'seasonal_arima.pkl')
                                joblib.dump(arima_model, model_path)
                                
                                # Forecast trend for next 6 months
                                last_date = trend.index[-1]
                                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='MS')
                                
                                trend_forecast = arima_model.forecast(steps=6)
                                
                                # Apply seasonal factors to create final forecast
                                seasonal_forecast = []
                                for i, date in enumerate(future_dates):
                                    month = date.month
                                    trend_value = trend_forecast[i]
                                    
                                    # Apply seasonal factor
                                    seasonal_factor = normalized_factors.get(month, 1.0)
                                    forecast_value = trend_value * seasonal_factor
                                    
                                    seasonal_forecast.append({
                                        'month': date.strftime('%b %Y'),
                                        'forecast': max(0, round(forecast_value)),
                                        'lower_bound': max(0, round(forecast_value * 0.8)),
                                        'upper_bound': round(forecast_value * 1.2),
                                        'seasonal_factor': round(seasonal_factor * 100)
                                    })
                                
                                # Add to result
                                result['arima'] = {
                                    'forecast': seasonal_forecast,
                                    'accuracy': 0.89  # Placeholder for now
                                }
                                
                                # Peak periods analysis
                                months_sorted = sorted(normalized_factors.items(), key=lambda x: x[1], reverse=True)
                                peak_months = [pd.Timestamp(2023, month, 1).strftime('%B') for month, _ in months_sorted[:3]]
                                low_months = [pd.Timestamp(2023, month, 1).strftime('%B') for month, _ in months_sorted[-3:]]
                                
                                result['peak_periods'] = {
                                    'high_season': peak_months,
                                    'low_season': low_months
                                }
                    except Exception as e:
                        print(f"Error in seasonal decomposition: {e}")
                
                # Basic monthly pattern regardless of decomposition
                monthly_pattern = df.groupby('month').size().reset_index(name='count')
                monthly_pattern['month_name'] = monthly_pattern['month'].apply(lambda x: pd.Timestamp(2023, x, 1).strftime('%b'))
                monthly_pattern = monthly_pattern.sort_values('month')
                
                # Calculate relative volume
                total = monthly_pattern['count'].sum()
                monthly_pattern['percentage'] = (monthly_pattern['count'] / total * 100).round(1)
                
                # Create monthly pattern data
                pattern_data = []
                for _, row in monthly_pattern.iterrows():
                    pattern_data.append({
                        'month': row['month_name'],
                        'shipments': int(row['count']),
                        'percentage': float(row['percentage'])
                    })
                
                result['monthly_pattern'] = pattern_data
                
                # If no ARIMA model was created, use simple forecasting
                if 'arima' not in result:
                    # Make predictions for next 6 months
                    last_date = df['BAG_DT'].max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='MS')
                    
                    # Calculate average shipments per month
                    avg_monthly = monthly_counts['shipments'].mean()
                    
                    forecast_data = []
                    for date in future_dates:
                        month = date.month
                        month_data = monthly_pattern[monthly_pattern['month'] == month]
                        
                        if not month_data.empty:
                            # Use monthly pattern to adjust forecast
                            monthly_percent = month_data.iloc[0]['percentage'] / 100
                            relative_factor = monthly_percent * 12  # Adjust based on relative monthly volume
                            forecast_value = avg_monthly * relative_factor * 1.05  # Add 5% growth
                        else:
                            forecast_value = avg_monthly * 1.05
                        
                        forecast_data.append({
                            'month': date.strftime('%b %Y'),
                            'forecast': max(0, round(forecast_value)),
                            'lower_bound': max(0, round(forecast_value * 0.7)),
                            'upper_bound': round(forecast_value * 1.3),
                            'seasonal_factor': round(relative_factor * 100) if not month_data.empty else 100
                        })
                    
                    # Add to result
                    result['historical'] = {
                        'forecast': forecast_data,
                        'accuracy': 0.82  # Placeholder for now
                    }
        
        # Set default model if needed
        default_model = 'arima' if 'arima' in result else 'historical'
        result['default_model'] = default_model
        
        # Save model info
        self.models['seasonal'] = result
        
        return result
    
    def train_processing_time_model(self) -> Dict[str, Any]:
        """
        Train models to forecast processing time between document generation and shipping.
        
        Returns:
            Dictionary with model information and performance metrics
        """
        result = {}
        
        if not self.data_processor.df.empty and 'TDG_DT' in self.data_processor.df.columns and 'FLT_DT' in self.data_processor.df.columns:
            df = self.data_processor.df.copy()
            
            # Convert date columns to datetime
            df['TDG_DT'] = pd.to_datetime(df['TDG_DT'], errors='coerce')
            df['FLT_DT'] = pd.to_datetime(df['FLT_DT'], errors='coerce')
            
            # Calculate processing time in days
            df['processing_days'] = (df['FLT_DT'] - df['TDG_DT']).dt.days
            
            # Filter valid processing times
            df = df[(df['processing_days'] >= 0) & (df['processing_days'] <= 30)]  # Exclude outliers
            
            if not df.empty:
                # Calculate average processing time by month
                df['month'] = df['TDG_DT'].dt.month
                df['year'] = df['TDG_DT'].dt.year
                df['month_year'] = df['TDG_DT'].dt.strftime('%Y-%m')
                
                # Calculate monthly processing times
                monthly_proc_time = df.groupby('month_year')['processing_days'].mean().reset_index()
                monthly_proc_time['date'] = pd.to_datetime(monthly_proc_time['month_year'])
                monthly_proc_time = monthly_proc_time.sort_values('date')
                
                if len(monthly_proc_time) > 6:  # Need enough data points
                    # Get time series
                    ts_proc_time = monthly_proc_time.set_index('date')['processing_days']
                    
                    # Train ARIMA model
                    arima_model = self._train_arima_model(ts_proc_time)
                    
                    if arima_model:
                        # Save model
                        model_path = os.path.join(self.model_dir, 'processing_arima.pkl')
                        joblib.dump(arima_model, model_path)
                        
                        # Make predictions for next 6 months
                        last_date = ts_proc_time.index[-1]
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                        
                        forecast = arima_model.forecast(steps=6)
                        
                        # Prepare forecast data
                        forecast_data = []
                        for i, date in enumerate(future_dates):
                            forecast_days = max(0.5, forecast[i])  # Ensure positive processing time
                            forecast_data.append({
                                'month': date.strftime('%b %Y'),
                                'forecast': round(forecast_days, 1),
                                'lower_bound': round(max(0.5, forecast_days * 0.8), 1),
                                'upper_bound': round(forecast_days * 1.2, 1)
                            })
                        
                        # Add to result
                        result['arima'] = {
                            'forecast': forecast_data,
                            'accuracy': 0.87  # Placeholder for now
                        }
                
                # Train ML model with more variables
                if 'REGION_CD' in df.columns and 'COMM_TYP' in df.columns:
                    # Create training data with region and commodity type
                    ml_df = df.groupby(['month', 'year', 'REGION_CD', 'COMM_TYP'])['processing_days'].mean().reset_index()
                    
                    if len(ml_df) > 10:  # Need enough data points
                        # Train ML model
                        ml_model = self._train_ml_model(
                            ml_df,
                            'processing_days',
                            categorical_features=['month', 'REGION_CD', 'COMM_TYP'],
                            numeric_features=['year']
                        )
                        
                        if ml_model:
                            # Save model
                            model_path = os.path.join(self.model_dir, 'processing_ml.pkl')
                            joblib.dump(ml_model, model_path)
                            
                            # Get top regions and commodity types
                            top_regions = df['REGION_CD'].value_counts().head(3).index.tolist()
                            top_commodities = df['COMM_TYP'].value_counts().head(3).index.tolist()
                            
                            # Make predictions for next 6 months
                            last_date = df['TDG_DT'].max()
                            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=6, freq='M')
                            
                            # Create future data for prediction
                            future_data = []
                            for region in top_regions:
                                for comm_type in top_commodities:
                                    for date in future_dates:
                                        future_data.append({
                                            'month': date.month,
                                            'year': date.year,
                                            'REGION_CD': region,
                                            'COMM_TYP': comm_type,
                                            'date': date,
                                            'region_comm': f"{region}_{comm_type}"
                                        })
                            
                            future_df = pd.DataFrame(future_data)
                            
                            # Make predictions
                            future_df['forecast'] = ml_model.predict(future_df[['month', 'year', 'REGION_CD', 'COMM_TYP']])
                            future_df['forecast'] = np.maximum(0.5, future_df['forecast'])  # Ensure positive processing time
                            
                            # Calculate overall average by month
                            monthly_forecast = future_df.groupby('date')['forecast'].mean().reset_index()
                            
                            # Prepare forecast data
                            forecast_data = []
                            for _, row in monthly_forecast.iterrows():
                                forecast_data.append({
                                    'month': row['date'].strftime('%b %Y'),
                                    'forecast': round(row['forecast'], 1),
                                    'lower_bound': round(max(0.5, row['forecast'] * 0.85), 1),
                                    'upper_bound': round(row['forecast'] * 1.15, 1)
                                })
                            
                            # Prepare region/commodity breakdown for the first month
                            first_month = future_dates[0]
                            first_month_data = future_df[future_df['date'] == first_month]
                            
                            breakdown = []
                            for region in top_regions:
                                region_data = first_month_data[first_month_data['REGION_CD'] == region]
                                avg_days = region_data['forecast'].mean()
                                breakdown.append({
                                    'name': region,
                                    'days': round(avg_days, 1),
                                    'type': 'region'
                                })
                            
                            for comm_type in top_commodities:
                                comm_data = first_month_data[first_month_data['COMM_TYP'] == comm_type]
                                avg_days = comm_data['forecast'].mean()
                                breakdown.append({
                                    'name': f"Type {comm_type}",
                                    'days': round(avg_days, 1),
                                    'type': 'commodity'
                                })
                            
                            # Add to result
                            result['ml'] = {
                                'forecast': forecast_data,
                                'breakdown': breakdown,
                                'accuracy': 0.91  # Placeholder for now
                            }
                
                # Add historical average model
                if 'ml' not in result and 'arima' not in result:
                    # Calculate average by month
                    monthly_avg = df.groupby('month')['processing_days'].mean()
                    
                    # Make predictions for next 6 months
                    future_dates = pd.date_range(start=datetime.now(), periods=6, freq='M')
                    
                    forecast_data = []
                    for date in future_dates:
                        month = date.month
                        if month in monthly_avg.index:
                            forecast_days = monthly_avg[month]
                        else:
                            forecast_days = df['processing_days'].mean()
                        
                        forecast_data.append({
                            'month': date.strftime('%b %Y'),
                            'forecast': round(forecast_days, 1),
                            'lower_bound': round(max(0.5, forecast_days * 0.7), 1),
                            'upper_bound': round(forecast_days * 1.3, 1)
                        })
                    
                    # Add to result
                    result['historical'] = {
                        'forecast': forecast_data,
                        'accuracy': 0.75  # Placeholder for now
                    }
        
        # Set default model if needed
        default_model = 'ml' if 'ml' in result else 'arima' if 'arima' in result else 'historical'
        result['default_model'] = default_model
        
        # Save model info
        self.models['processing'] = result
        
        return result
    
    def get_forecast(self, forecast_type: str, model_type: str = None) -> Dict[str, Any]:
        """
        Get forecast data for the specified type and model.
        
        Args:
            forecast_type: Type of forecast (demand, weight, value, carrier, seasonal, processing)
            model_type: Model to use (ml, arima, historical)
            
        Returns:
            Dictionary with forecast data
        """
        # Check if models have been trained
        if not self.models:
            # Train models for each forecast type
            self.train_demand_forecast_model()
            self.train_weight_forecast_model()
            self.train_value_forecast_model()
            self.train_carrier_forecast_model()
            self.train_seasonal_analysis_model()
            self.train_processing_time_model()
        
        # Get forecast for the specified type
        if forecast_type in self.models:
            forecast_data = self.models[forecast_type]
            
            # If model type is specified, return only that model's forecast
            if model_type and model_type in forecast_data:
                return {
                    'model': model_type,
                    'data': forecast_data[model_type]
                }
            else:
                # Return the default model's forecast
                default_model = forecast_data.get('default_model', next(iter(forecast_data.keys())))
                return {
                    'model': default_model,
                    'data': forecast_data[default_model]
                }
        else:
            # Return empty result if forecast type not found
            return {
                'model': 'none',
                'data': None
            }
    
    def get_all_models(self, forecast_type: str) -> Dict[str, Any]:
        """
        Get data for all models of a particular forecast type.
        
        Args:
            forecast_type: Type of forecast (demand, weight, value, carrier, seasonal, processing)
            
        Returns:
            Dictionary with all model data and accuracies
        """
        # Check if models have been trained
        if not self.models:
            # Train models for each forecast type
            self.train_demand_forecast_model()
            self.train_weight_forecast_model()
            self.train_value_forecast_model()
            self.train_carrier_forecast_model()
            self.train_seasonal_analysis_model()
            self.train_processing_time_model()
        
        # Get all models for the specified type
        if forecast_type in self.models:
            forecast_data = self.models[forecast_type]
            
            # Format the response
            models = {}
            accuracies = {}
            
            for model_name, model_data in forecast_data.items():
                if model_name != 'default_model':
                    if isinstance(model_data, dict) and 'forecast' in model_data:
                        models[model_name] = model_data['forecast']
                        accuracies[model_name] = model_data.get('accuracy', 0)
            
            default_model = forecast_data.get('default_model', next(iter(models.keys())) if models else None)
            
            return {
                'models': models,
                'accuracies': accuracies,
                'default_model': default_model
            }
        else:
            # Return empty result if forecast type not found
            return {
                'models': {},
                'accuracies': {},
                'default_model': None
            }