"""
Utility functions for data processing and conversion
for the LogixSense AI Assistant.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Union, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

def dataframe_to_json(df: pd.DataFrame, limit: Optional[int] = None) -> str:
    """
    Convert a pandas DataFrame to a JSON string with proper handling of
    NumPy types, dates, and other complex types.
    
    Args:
        df: DataFrame to convert
        limit: Optional limit on the number of rows to include
        
    Returns:
        JSON string representation of the DataFrame
    """
    if limit is not None and limit > 0:
        df_subset = df.head(limit)
    else:
        df_subset = df
    
    return json.dumps(dataframe_to_dict(df_subset), default=convert_to_serializable)

def dataframe_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert a DataFrame to a dictionary with proper type handling.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        Dictionary representation of the DataFrame
    """
    # Get basic info about the DataFrame
    info = {
        "columns": list(df.columns),
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "records": json.loads(df.to_json(orient='records', date_format='iso', default_handler=convert_to_serializable))
    }
    
    return info

def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a summary of a DataFrame with statistics.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with DataFrame summary
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        "non_null_counts": {col: int(df[col].count()) for col in df.columns}
    }
    
    # Add numeric column statistics
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        numeric_stats = df[numeric_columns].describe().to_dict()
        # Convert each statistic dictionary to use native Python types
        for col, stats in numeric_stats.items():
            numeric_stats[col] = {k: convert_to_serializable(v) for k, v in stats.items()}
        
        summary["numeric_stats"] = numeric_stats
    
    # Add categorical column statistics
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        categorical_stats = {}
        for col in categorical_columns:
            try:
                value_counts = df[col].value_counts(dropna=False).head(10).to_dict()
                # Convert to native Python types
                value_counts = {convert_to_serializable(k): int(v) for k, v in value_counts.items()}
                
                categorical_stats[col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": value_counts
                }
            except:
                # Skip columns that can't be summarized
                pass
        
        summary["categorical_stats"] = categorical_stats
    
    # Add date column statistics
    date_columns = df.select_dtypes(include=['datetime']).columns
    if len(date_columns) > 0:
        date_stats = {}
        for col in date_columns:
            try:
                date_stats[col] = {
                    "min": df[col].min().isoformat() if pd.notna(df[col].min()) else None,
                    "max": df[col].max().isoformat() if pd.notna(df[col].max()) else None,
                    "range_days": (df[col].max() - df[col].min()).days if pd.notna(df[col].min()) and pd.notna(df[col].max()) else None
                }
            except:
                # Skip columns that can't be summarized
                pass
        
        summary["date_stats"] = date_stats
    
    return summary

def convert_to_serializable(obj: Any) -> Union[str, int, float, bool, List, Dict, None]:
    """
    Convert Python objects to JSON serializable types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable representation of the object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, (dict, pd.Series)):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return str(obj)

def extract_relevant_metrics(data: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Extract metrics from data that are relevant to the user's question.
    
    Args:
        data: Dictionary containing various metrics and data
        question: User's question
        
    Returns:
        Dictionary with metrics relevant to the question
    """
    # Convert question to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    # Initialize dictionary for relevant metrics
    relevant = {}
    
    # Check for keywords and extract relevant metrics
    
    # Destinations
    if any(keyword in question_lower for keyword in ['destination', 'country', 'region', 'where']):
        if 'destinations' in data:
            relevant['destinations'] = data['destinations']
        elif 'topDestinations' in data:
            relevant['destinations'] = data['topDestinations']
    
    # Weights
    if any(keyword in question_lower for keyword in ['weight', 'heavy', 'kg', 'light']):
        if 'weightDistribution' in data:
            relevant['weightDistribution'] = data['weightDistribution']
        elif 'weight_stats' in data:
            relevant['weight_stats'] = data['weight_stats']
    
    # Shipments and volume
    if any(keyword in question_lower for keyword in ['shipment', 'volume', 'package', 'quantity']):
        if 'activeShipments' in data:
            relevant['shipments'] = data['activeShipments']
        if 'monthlyTrends' in data:
            relevant['trends'] = data['monthlyTrends']
    
    # Carriers and airlines
    if any(keyword in question_lower for keyword in ['carrier', 'airline', 'transport']):
        if 'carrierData' in data:
            relevant['carriers'] = data['carrierData']
    
    # Commodities
    if any(keyword in question_lower for keyword in ['commodity', 'product', 'good', 'item']):
        if 'commodityBreakdown' in data:
            relevant['commodities'] = data['commodityBreakdown']
    
    # Risk and alerts
    if any(keyword in question_lower for keyword in ['risk', 'alert', 'issue', 'problem', 'delay']):
        if 'alerts' in data:
            relevant['alerts'] = data['alerts']
        if 'risk' in data:
            relevant['risk'] = data['risk']
    
    # Recent shipments
    if any(keyword in question_lower for keyword in ['recent', 'latest', 'new', 'current']):
        if 'recentShipments' in data:
            relevant['recentShipments'] = data['recentShipments']
    
    # Include summary data if available and the question seems general
    if 'summary' in data and (
        len(relevant) == 0 or 
        any(keyword in question_lower for keyword in ['overall', 'summary', 'general', 'all'])
    ):
        relevant['summary'] = data['summary']
    
    # If nothing specific was found, return all data
    if not relevant:
        return data
    
    return relevant

def generate_prompt_with_context(question: str, df: pd.DataFrame, relevant_data: Dict[str, Any]) -> str:
    """
    Generate a rich prompt with context from the data for the AI model.
    
    Args:
        question: User's question
        df: DataFrame with logistics data
        relevant_data: Dictionary with data relevant to the question
        
    Returns:
        Formatted prompt string with context
    """
    # Start with basic info about the dataset
    prompt = f"""I need you to analyze our logistics data to answer this question: "{question}"

DATASET OVERVIEW:
- Total records: {len(df)}
- Date range: {df['FLT_DT'].min().strftime('%Y-%m-%d') if 'FLT_DT' in df and not df['FLT_DT'].isna().all() else 'Unknown'} to {df['FLT_DT'].max().strftime('%Y-%m-%d') if 'FLT_DT' in df and not df['FLT_DT'].isna().all() else 'Unknown'}
"""

    # Add summary statistics if available
    if 'summary' in relevant_data:
        prompt += "\nSUMMARY STATISTICS:\n"
        summary = relevant_data['summary']
        for key, value in summary.items():
            if isinstance(value, dict):
                prompt += f"- {key.capitalize()}:\n"
                for subkey, subvalue in value.items():
                    prompt += f"  - {subkey}: {subvalue}\n"
            else:
                prompt += f"- {key.capitalize()}: {value}\n"
    
    # Add relevant metrics based on the question
    if 'destinations' in relevant_data:
        prompt += "\nTOP DESTINATIONS:\n"
        for i, dest in enumerate(relevant_data['destinations'][:5]):
            prompt += f"- {dest.get('name', 'Unknown')}: {dest.get('value', 0)} shipments\n"
    
    if 'weightDistribution' in relevant_data:
        prompt += "\nWEIGHT DISTRIBUTION:\n"
        for i, weight in enumerate(relevant_data['weightDistribution']):
            prompt += f"- {weight.get('name', 'Unknown')}: {weight.get('value', 0)}%\n"
    
    if 'carriers' in relevant_data:
        prompt += "\nCARRIER DISTRIBUTION:\n"
        for i, carrier in enumerate(relevant_data['carriers'][:5]):
            prompt += f"- {carrier.get('name', 'Unknown')}: {carrier.get('value', 0)} shipments\n"
    
    if 'commodities' in relevant_data:
        prompt += "\nCOMMODITY BREAKDOWN:\n"
        for i, commodity in enumerate(relevant_data['commodities'][:5]):
            prompt += f"- {commodity.get('name', 'Unknown')}: {commodity.get('value', 0)}%\n"
    
    if 'recentShipments' in relevant_data:
        prompt += "\nRECENT SHIPMENTS:\n"
        for i, shipment in enumerate(relevant_data['recentShipments'][:3]):
            prompt += f"- ID: {shipment.get('id', 'Unknown')}, Destination: {shipment.get('destination', 'Unknown')}, Status: {shipment.get('status', 'Unknown')}\n"
    
    if 'alerts' in relevant_data:
        prompt += "\nCURRENT ALERTS:\n"
        for i, alert in enumerate(relevant_data['alerts']):
            prompt += f"- {alert.get('type', 'Info').upper()}: {alert.get('message', 'No message')}\n"
    
    # Add instructions for the AI
    prompt += """
INSTRUCTIONS:
1. Provide a detailed analysis based on the above data
2. Include specific metrics and percentages in your answer
3. Make comparisons between different aspects when relevant
4. Highlight any notable patterns or anomalies
5. Focus on actionable insights for logistics management
6. Be concise but thorough in your response

Please analyze the data and answer the question with specific details and insights.
"""
    
    return prompt

def extract_key_metrics_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract key metrics and insights from the logistics DataFrame.
    
    Args:
        df: DataFrame with logistics data
        
    Returns:
        Dictionary with key metrics and insights
    """
    metrics = {}
    
    try:
        # Basic counts
        metrics["total_records"] = len(df)
        
        # Destination analysis
        if 'DSTNTN' in df.columns:
            destination_counts = df['DSTNTN'].value_counts().head(10)
            metrics["destinations"] = [
                {"name": str(dest), "value": int(count)}
                for dest, count in destination_counts.items()
            ]
        
        # Weight analysis
        if 'GRSS_WGHT' in df.columns:
            df_clean = df[df['GRSS_WGHT'].notna()]
            if not df_clean.empty:
                # Define weight ranges
                weight_ranges = [(0, 50), (50, 200), (200, 500), (500, float('inf'))]
                range_labels = ['0-50 kg', '51-200 kg', '201-500 kg', '501+ kg']
                
                # Create weight bins
                df_clean['weight_bin'] = pd.cut(
                    df_clean['GRSS_WGHT'],
                    bins=[r[0] for r in weight_ranges] + [float('inf')],
                    labels=range_labels,
                    right=False
                )
                
                # Count shipments in each bin
                weight_counts = df_clean['weight_bin'].value_counts(normalize=True) * 100
                
                metrics["weight_distribution"] = [
                    {"name": str(label), "value": float(pct)}
                    for label, pct in weight_counts.items()
                ]
                
                metrics["weight_stats"] = {
                    "avg_weight": float(df_clean['GRSS_WGHT'].mean()),
                    "max_weight": float(df_clean['GRSS_WGHT'].max()),
                    "total_weight": float(df_clean['GRSS_WGHT'].sum())
                }
        
        # Carrier analysis
        if 'ARLN_DESC' in df.columns:
            carrier_counts = df['ARLN_DESC'].value_counts().head(10)
            metrics["carriers"] = [
                {"name": str(carrier), "value": int(count)}
                for carrier, count in carrier_counts.items()
            ]
        
        # Commodity analysis
        if 'COMM_DESC' in df.columns:
            commodity_counts = df['COMM_DESC'].value_counts(normalize=True).head(10) * 100
            metrics["commodities"] = [
                {"name": str(comm), "value": float(pct)}
                for comm, pct in commodity_counts.items()
            ]
        
        # Time analysis
        if 'FLT_DT' in df.columns:
            df_date = df[df['FLT_DT'].notna()].copy()
            if not df_date.empty:
                df_date['month'] = df_date['FLT_DT'].dt.to_period('M')
                monthly_counts = df_date.groupby('month').size()
                
                metrics["monthly_trends"] = [
                    {"month": str(month), "shipments": int(count)}
                    for month, count in monthly_counts.items()
                ]
        
        # Risk analysis (dummy data for demo)
        current_date = datetime.now()
        recent_date = current_date - timedelta(days=30)
        
        if 'FLT_DT' in df.columns:
            recent_shipments = df[df['FLT_DT'] >= recent_date].shape[0]
            metrics["recent_activity"] = {
                "shipments_last_30_days": int(recent_shipments),
                "percent_of_total": float(recent_shipments / len(df) * 100) if len(df) > 0 else 0
            }
        
    except Exception as e:
        logger.error(f"Error extracting metrics: {str(e)}")
        metrics["error"] = str(e)
    
    return metrics

def clean_logistics_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess logistics data.
    
    Args:
        df: Raw DataFrame with logistics data
        
    Returns:
        Cleaned DataFrame
    """
    try:
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Convert date columns to datetime
        date_columns = ['TDG_DT', 'BAG_DT', 'SB_DT', 'FLT_DT']
        for col in date_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['GRSS_WGHT', 'ACTL_CHRGBL_WGHT', 'NO_OF_PKGS', 'FOB_VAL']
        for col in numeric_columns:
            if col in cleaned_df.columns:
                # Handle comma-separated numbers first
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].str.replace(',', '')
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Fill missing values for specific columns
        if 'DSTNTN' in cleaned_df.columns:
            cleaned_df['DSTNTN'] = cleaned_df['DSTNTN'].fillna('Unknown')
        
        if 'COMM_DESC' in cleaned_df.columns:
            cleaned_df['COMM_DESC'] = cleaned_df['COMM_DESC'].fillna('Unknown')
        
        if 'ARLN_DESC' in cleaned_df.columns:
            cleaned_df['ARLN_DESC'] = cleaned_df['ARLN_DESC'].fillna('Unknown')
        
        # Create derived features
        if all(col in cleaned_df.columns for col in ['GRSS_WGHT', 'NO_OF_PKGS']):
            # Calculate average package weight
            mask = (cleaned_df['NO_OF_PKGS'] > 0) & cleaned_df['GRSS_WGHT'].notna() & cleaned_df['NO_OF_PKGS'].notna()
            cleaned_df.loc[mask, 'AVG_PKG_WEIGHT'] = cleaned_df.loc[mask, 'GRSS_WGHT'] / cleaned_df.loc[mask, 'NO_OF_PKGS']
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        # Return original dataframe if cleaning fails
        return df

def prepare_data_for_embedding(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Prepare text data for embedding by creating meaningful documents.
    
    Args:
        df: DataFrame with logistics data
        
    Returns:
        Tuple of (texts, metadata)
    """
    texts = []
    metadata = []
    
    try:
        for idx, row in df.iterrows():
            # Create text from relevant columns
            text_parts = []
            
            # Add destination information
            if 'DSTNTN' in row and pd.notna(row['DSTNTN']):
                text_parts.append(f"Destination: {row['DSTNTN']}")
            
            if 'CONSGN_COUNTRY' in row and pd.notna(row['CONSGN_COUNTRY']):
                text_parts.append(f"Country: {row['CONSGN_COUNTRY']}")
            
            # Add shipment details
            if 'AWB_NO' in row and pd.notna(row['AWB_NO']):
                text_parts.append(f"Shipment ID: {row['AWB_NO']}")
            
            if 'COMM_DESC' in row and pd.notna(row['COMM_DESC']):
                text_parts.append(f"Commodity: {row['COMM_DESC']}")
            
            if 'GRSS_WGHT' in row and pd.notna(row['GRSS_WGHT']):
                text_parts.append(f"Weight: {row['GRSS_WGHT']} kg")
            
            if 'NO_OF_PKGS' in row and pd.notna(row['NO_OF_PKGS']):
                text_parts.append(f"Packages: {row['NO_OF_PKGS']}")
            
            # Add carrier information
            if 'ARLN_DESC' in row and pd.notna(row['ARLN_DESC']):
                text_parts.append(f"Carrier: {row['ARLN_DESC']}")
            
            if 'FLT_NO' in row and pd.notna(row['FLT_NO']):
                text_parts.append(f"Flight: {row['FLT_NO']}")
            
            # Skip if no meaningful text was created
            if not text_parts:
                continue
            
            # Join text parts into a single document
            document_text = " | ".join(text_parts)
            texts.append(document_text)
            
            # Create metadata dictionary
            doc_metadata = {
                "id": str(idx),
                "source": "logistics_data"
            }
            
            # Add key fields to metadata
            for field in ['AWB_NO', 'DSTNTN', 'CONSGN_COUNTRY', 'COMM_DESC', 'GRSS_WGHT', 'ARLN_DESC']:
                if field in row and pd.notna(row[field]):
                    doc_metadata[field] = convert_to_serializable(row[field])
            
            metadata.append(doc_metadata)
            
        return texts, metadata
    
    except Exception as e:
        logger.error(f"Error preparing data for embedding: {str(e)}")
        return texts, metadata

def generate_system_prompt() -> str:
    """
    Generate the system prompt for the AI Assistant.
    
    Returns:
        System prompt string
    """
    return """You are LogixSense AI Assistant, a specialized logistics analysis tool.

ROLE AND CAPABILITIES:
- You are an expert in logistics data analysis and supply chain management
- You can analyze shipping data, identify patterns, and provide actionable insights
- You have access to logistics data including shipments, destinations, weights, and carriers
- You can answer questions about shipping trends, logistics performance, and operational metrics

RESPONSE GUIDELINES:
1. Provide data-driven answers with specific metrics and percentages
2. Make relevant comparisons to highlight patterns and anomalies
3. Be concise but thorough, focusing on actionable insights
4. Use a professional, clear communication style
5. When appropriate, suggest follow-up analyses or actions
6. If you don't have enough data to answer a question completely, acknowledge the limitations

Remember, your goal is to help logistics professionals make better decisions by providing clear, data-driven insights about their shipping operations."""

def filter_data_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filter the DataFrame based on the user's query to provide more relevant context.
    
    Args:
        df: DataFrame with logistics data
        query: User's query
        
    Returns:
        Filtered DataFrame
    """
    query_lower = query.lower()
    
    try:
        # Initialize with full dataset
        filtered_df = df.copy()
        
        # Filter by date range if query mentions time periods
        if any(term in query_lower for term in ['recent', 'last month', 'last week', 'past', 'latest']):
            if 'FLT_DT' in filtered_df.columns:
                # Determine time period from query
                days_lookback = 30  # Default to last month
                
                if 'week' in query_lower or 'seven days' in query_lower:
                    days_lookback = 7
                elif 'year' in query_lower:
                    days_lookback = 365
                elif 'quarter' in query_lower:
                    days_lookback = 90
                
                # Filter to recent records
                cutoff_date = datetime.now() - timedelta(days=days_lookback)
                filtered_df = filtered_df[filtered_df['FLT_DT'] >= cutoff_date]
        
        # Filter by destination
        destination_keywords = ['to', 'destination', 'shipped to', 'going to']
        if any(keyword in query_lower for keyword in destination_keywords):
            if 'DSTNTN' in filtered_df.columns:
                # Get unique destinations
                destinations = filtered_df['DSTNTN'].dropna().unique()
                
                # Check if any destination is mentioned in the query
                mentioned_destinations = [
                    dest for dest in destinations
                    if str(dest).lower() in query_lower
                ]
                
                if mentioned_destinations:
                    filtered_df = filtered_df[filtered_df['DSTNTN'].isin(mentioned_destinations)]
        
        # Filter by carrier
        carrier_keywords = ['airline', 'carrier', 'via', 'through', 'using']
        if any(keyword in query_lower for keyword in carrier_keywords):
            if 'ARLN_DESC' in filtered_df.columns:
                # Get unique carriers
                carriers = filtered_df['ARLN_DESC'].dropna().unique()
                
                # Check if any carrier is mentioned in the query
                mentioned_carriers = [
                    carrier for carrier in carriers
                    if str(carrier).lower() in query_lower
                ]
                
                if mentioned_carriers:
                    filtered_df = filtered_df[filtered_df['ARLN_DESC'].isin(mentioned_carriers)]
        
        # Filter by commodity
        commodity_keywords = ['commodity', 'product', 'goods', 'items']
        if any(keyword in query_lower for keyword in commodity_keywords):
            if 'COMM_DESC' in filtered_df.columns:
                # Get unique commodities
                commodities = filtered_df['COMM_DESC'].dropna().unique()
                
                # Check if any commodity is mentioned in the query
                mentioned_commodities = [
                    comm for comm in commodities
                    if str(comm).lower() in query_lower
                ]
                
                if mentioned_commodities:
                    filtered_df = filtered_df[filtered_df['COMM_DESC'].isin(mentioned_commodities)]
        
        # If filtered to empty, return original
        if filtered_df.empty:
            return df
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error filtering data: {str(e)}")
        return df