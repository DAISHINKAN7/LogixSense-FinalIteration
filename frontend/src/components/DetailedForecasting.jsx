'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  AreaChart, Area, LineChart, Line, BarChart, Bar, 
  PieChart, Pie, Cell, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Package, DollarSign, 
  TruckIcon, Calendar, Clock, RefreshCcw, Download, 
  ChevronRight, AlertTriangle, Zap, BarChart2, 
  Activity
} from 'lucide-react';

// Import custom animation components (assumed to exist similar to EnhancedDashboard)
// If these components don't exist yet, you would need to create them
import ForecastAnimations from './ForecastAnimations';

// API URL - change this to match your backend
const API_URL = 'http://localhost:8000/api';

// Create custom color schemes
const gradientColors = {
  blue: ['#3b82f6', '#1d4ed8'],
  green: ['#10b981', '#047857'],
  amber: ['#f59e0b', '#b45309'],
  purple: ['#8b5cf6', '#6d28d9'],
  pink: ['#ec4899', '#be185d'],
  cyan: ['#06b6d4', '#0e7490']
};

export default function EnhancedDetailedForecasting() {
  const [activeTab, setActiveTab] = useState('demand');
  const [isLoading, setIsLoading] = useState(true);
  const [forecastData, setForecastData] = useState({});
  const [selectedModels, setSelectedModels] = useState({
    demand: 'ml',
    weight: 'ml',
    value: 'ml',
    carrier: 'ml',
    seasonal: 'arima',
    processing: 'ml'
  });
  const [modelOptions, setModelOptions] = useState({
    demand: [],
    weight: [],
    value: [],
    carrier: [],
    seasonal: [],
    processing: []
  });
  const [modelAccuracies, setModelAccuracies] = useState({});
  const [error, setError] = useState(null);

  // Fetch forecast data for a specific type
  const fetchForecast = async (type) => {
    try {
      setIsLoading(true);
      const model = selectedModels[type];
      const response = await fetch(`${API_URL}/forecasting/${type}?model=${model}`);
      
      if (!response.ok) {
        throw new Error(`Error fetching ${type} forecast data: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Update state with the new data
      setForecastData(prevData => ({
        ...prevData,
        [type]: data
      }));
      
      setIsLoading(false);
    } catch (error) {
      console.error(`Error fetching ${type} forecast:`, error);
      setError(error.message);
      setIsLoading(false);
    }
  };

  // Fetch available models for a forecast type
  const fetchModels = async (type) => {
    try {
      const response = await fetch(`${API_URL}/forecasting/models/${type}`);
      
      if (!response.ok) {
        throw new Error(`Error fetching models for ${type}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Update model options
      const modelList = Object.keys(data.models || {});
      setModelOptions(prevOptions => ({
        ...prevOptions,
        [type]: modelList
      }));
      
      // Update accuracies
      setModelAccuracies(prevAccuracies => ({
        ...prevAccuracies,
        [type]: data.accuracies || {}
      }));
      
      // Set default model if available
      if (data.default_model && data.default_model !== selectedModels[type]) {
        setSelectedModels(prevModels => ({
          ...prevModels,
          [type]: data.default_model
        }));
      }
    } catch (error) {
      console.error(`Error fetching models for ${type}:`, error);
    }
  };

  // Fetch initial data
  useEffect(() => {
    const fetchAllData = async () => {
      try {
        // Fetch available models for all forecast types
        const types = ['demand', 'weight', 'value', 'carrier', 'seasonal', 'processing'];
        for (const type of types) {
          await fetchModels(type);
        }
        
        // Fetch forecast for the active tab
        await fetchForecast(activeTab);
      } catch (error) {
        console.error('Error fetching initial data:', error);
        setError('Failed to load forecasting data. Please try again later.');
      }
    };

    fetchAllData();
  }, []);

  // Fetch data when active tab changes
  useEffect(() => {
    if (activeTab) {
      fetchForecast(activeTab);
    }
  }, [activeTab]);

  // Fetch data when selected model changes
  useEffect(() => {
    if (activeTab) {
      fetchForecast(activeTab);
    }
  }, [selectedModels]);

  // Function to handle model selection
  const handleModelChange = (type, value) => {
    setSelectedModels(prevModels => ({
      ...prevModels,
      [type]: value
    }));
  };

  // Get color scheme for forecast type
  const getColorScheme = (type) => {
    switch (type) {
      case 'demand':
        return 'blue';
      case 'weight':
        return 'green';
      case 'value':
        return 'amber';
      case 'carrier':
        return 'cyan';
      case 'seasonal':
        return 'purple';
      case 'processing':
        return 'pink';
      default:
        return 'blue';
    }
  };

  // Get icon for forecast type
  const getIcon = (type) => {
    switch (type) {
      case 'demand':
        return <TrendingUp className="h-5 w-5" />;
      case 'weight':
        return <Package className="h-5 w-5" />;
      case 'value':
        return <DollarSign className="h-5 w-5" />;
      case 'carrier':
        return <TruckIcon className="h-5 w-5" />;
      case 'seasonal':
        return <Calendar className="h-5 w-5" />;
      case 'processing':
        return <Clock className="h-5 w-5" />;
      default:
        return <BarChart2 className="h-5 w-5" />;
    }
  };

  // Get gradient ID for RecHarts
  const getGradientId = (color) => `color${color}Gradient`;

  // Function to refresh data
  const handleRefresh = () => {
    fetchForecast(activeTab);
  };

  // Generate CSV data for download
  const generateCSV = (data) => {
    if (!data || !data.data || !data.data.forecast) return '';
    
    const forecast = data.data.forecast;
    const headers = Object.keys(forecast[0]).join(',');
    const rows = forecast.map(row => Object.values(row).join(',')).join('\n');
    
    return `${headers}\n${rows}`;
  };

  // Handle download
  const handleDownload = () => {
    const data = forecastData[activeTab];
    if (!data) return;
    
    const csv = generateCSV(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', `${activeTab}_forecast.csv`);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Format data for charts based on forecast type
  const formatChartData = (type, data) => {
    if (!data || !data.data || !data.data.forecast) return [];
    
    const forecast = data.data.forecast;
    
    switch (type) {
      case 'demand':
      case 'weight':
      case 'value':
      case 'processing':
        // Standard forecast data
        return forecast.map(item => ({
          month: item.month,
          forecast: item.forecast,
          lowerBound: item.lower_bound,
          upperBound: item.upper_bound
        }));
      
      case 'seasonal':
        // Seasonal data includes seasonal factors
        return forecast.map(item => ({
          month: item.month,
          forecast: item.forecast,
          lowerBound: item.lower_bound,
          upperBound: item.upper_bound,
          seasonalFactor: item.seasonal_factor
        }));
      
      case 'carrier':
        // Carrier data has nested structure
        if (forecast[0] && forecast[0].carriers) {
          // Get first month carrier distribution for pie chart
          const firstMonth = forecast[0];
          return firstMonth.carriers.map(carrier => ({
            name: carrier.carrier,
            value: carrier.percentage
          }));
        }
        return [];
      
      default:
        return forecast;
    }
  };

  // Render specific charts based on forecast type
  const renderChart = (type, data) => {
    if (isLoading) return (
      <div className="h-[400px] w-full bg-slate-100 dark:bg-slate-800 animate-pulse rounded-md flex items-center justify-center">
        <ForecastAnimations.LoadingSpinner />
      </div>
    );
    
    const chartData = formatChartData(type, data);
    const colorScheme = getColorScheme(type);
    
    if (!chartData || chartData.length === 0) {
      return <div className="w-full h-[400px] flex items-center justify-center">No forecast data available</div>;
    }
    
    switch (type) {
      case 'demand':
        return (
          <div className="relative">
            <ForecastAnimations.ShiftingBackground className="opacity-10 rounded-lg" />
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <defs>
                  <linearGradient id={`detailed${getGradientId(colorScheme)}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={gradientColors[colorScheme][0]} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={gradientColors[colorScheme][0]} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="month" 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }}
                  tickFormatter={(value) => value.toLocaleString()}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                    border: 'none',
                    color: 'white'
                  }}
                  formatter={(value) => value.toLocaleString()}
                />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="upperBound" 
                  stroke="transparent"
                  fill={`url(#detailed${getGradientId(colorScheme)})`}
                  fillOpacity={0.2}
                  name="Upper Bound"
                />
                <Area 
                  type="monotone" 
                  dataKey="forecast" 
                  stroke={gradientColors[colorScheme][0]} 
                  fillOpacity={1} 
                  fill={`url(#detailed${getGradientId(colorScheme)})`}
                  strokeWidth={2}
                  name="Forecast Shipments"
                />
                <Area 
                  type="monotone" 
                  dataKey="lowerBound" 
                  stroke="transparent" 
                  fill="transparent"
                  name="Lower Bound"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        );
      
      case 'weight':
        return (
          <div className="relative">
            <ForecastAnimations.ShiftingBackground className="opacity-10 rounded-lg" />
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <defs>
                  <linearGradient id={`detailed${getGradientId(colorScheme)}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={gradientColors[colorScheme][0]} stopOpacity={0.8}/>
                    <stop offset="95%" stopColor={gradientColors[colorScheme][1]} stopOpacity={1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="month" 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                  tickFormatter={(value) => `${value.toLocaleString()} kg`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                    border: 'none',
                    color: 'white'
                  }}
                  formatter={(value) => `${value.toLocaleString()} kg`}
                />
                <Legend />
                <Bar 
                  dataKey="forecast" 
                  fill={`url(#detailed${getGradientId(colorScheme)})`} 
                  name="Forecasted Weight (kg)"
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  dataKey="lowerBound" 
                  fill={gradientColors.green[0]} 
                  name="Minimum Weight"
                  radius={[4, 4, 0, 0]}
                  stackId="stack"
                  opacity={0.3}
                />
                <Bar 
                  dataKey="upperBound" 
                  fill={gradientColors.amber[0]} 
                  name="Maximum Weight"
                  radius={[4, 4, 0, 0]}
                  stackId="stack"
                  opacity={0.3}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      
      case 'value':
        return (
          <div className="relative">
            <ForecastAnimations.ShiftingBackground className="opacity-10 rounded-lg" />
            <ResponsiveContainer width="100%" height={400}>
              <LineChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="month" 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                  tickFormatter={(value) => `₹${(value/1000000).toFixed(1)}M`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                    border: 'none',
                    color: 'white'
                  }}
                  formatter={(value) => `₹${value.toLocaleString()}`}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="forecast" 
                  stroke={gradientColors[colorScheme][0]} 
                  strokeWidth={3}
                  dot={{ fill: gradientColors[colorScheme][0], r: 6 }}
                  activeDot={{ r: 8 }}
                  name="Forecasted Value (₹)"
                />
                <Line 
                  type="monotone" 
                  dataKey="upperBound" 
                  stroke={gradientColors[colorScheme][0]} 
                  strokeWidth={1.5}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Upper Bound"
                />
                <Line 
                  type="monotone" 
                  dataKey="lowerBound" 
                  stroke={gradientColors[colorScheme][0]} 
                  strokeWidth={1.5}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Lower Bound"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        );
      
      case 'carrier':
        return (
          <div className="relative">
            <ForecastAnimations.ShiftingBackground className="opacity-10 rounded-lg" />
            <div className="flex flex-col items-center justify-center h-full pt-4">
              <h3 className="text-lg font-medium mb-2 text-center">
                Carrier Distribution Forecast
              </h3>
              <p className="text-sm text-center mb-4 text-gray-400">
                Projected distribution for {data.data.forecast[0]?.month || 'upcoming period'}
              </p>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={120}
                    paddingAngle={5}
                    dataKey="value"
                    nameKey="name"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    labelLine={{ stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1 }}
                  >
                    {chartData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={gradientColors[Object.keys(gradientColors)[index % 6]][0]} 
                      />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                      borderRadius: '8px',
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                      border: 'none',
                      color: 'white'
                    }}
                    formatter={(value) => `${value.toFixed(1)}%`}
                  />
                  <Legend formatter={(value) => value} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      
      case 'seasonal':
        return (
          <div className="relative">
            <ForecastAnimations.ShiftingBackground className="opacity-10 rounded-lg" />
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="month" 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                />
                <YAxis 
                  yAxisId="left"
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                  tickFormatter={(value) => value.toLocaleString()}
                  domain={['auto', 'auto']}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                  domain={[0, 'dataMax + 20']}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                    border: 'none',
                    color: 'white'
                  }}
                  formatter={(value, name) => [
                    name === 'Seasonal Factor' ? `${value}%` : value.toLocaleString(),
                    name
                  ]}
                />
                <Legend />
                <Area 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="forecast" 
                  stroke={gradientColors.blue[0]} 
                  fill={gradientColors.blue[0]}
                  fillOpacity={0.3}
                  name="Shipment Volume"
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="seasonalFactor" 
                  stroke={gradientColors[colorScheme][0]} 
                  strokeWidth={3}
                  dot={{ fill: gradientColors[colorScheme][0], r: 6 }}
                  name="Seasonal Factor"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        );
      
      case 'processing':
        return (
          <div className="relative">
            <ForecastAnimations.ShiftingBackground className="opacity-10 rounded-lg" />
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="month" 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.7)" 
                  tick={{ fill: 'rgba(255,255,255,0.7)' }} 
                  tickFormatter={(value) => `${value} days`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                    border: 'none',
                    color: 'white'
                  }}
                  formatter={(value) => `${value} days`}
                />
                <Legend />
                <Bar 
                  dataKey="forecast" 
                  fill={gradientColors[colorScheme][0]} 
                  name="Average Processing Days"
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  dataKey="upperBound" 
                  fill="#ef4444" 
                  name="Maximum Days"
                  radius={[4, 4, 0, 0]}
                  opacity={0.7}
                />
                <Bar 
                  dataKey="lowerBound" 
                  fill={gradientColors.green[0]} 
                  name="Minimum Days"
                  radius={[4, 4, 0, 0]}
                  opacity={0.7}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      
      default:
        return <div className="h-[400px] flex items-center justify-center">No chart available for this forecast type</div>;
    }
  };

  // Render forecast details table with enhanced styling
  const renderTable = (type, data) => {
    if (isLoading) {
      return (
        <div className="space-y-2">
          <div className="h-8 bg-slate-100 dark:bg-slate-800 rounded-md animate-pulse" />
          <div className="h-8 bg-slate-100 dark:bg-slate-800 rounded-md animate-pulse" />
          <div className="h-8 bg-slate-100 dark:bg-slate-800 rounded-md animate-pulse" />
          <div className="h-8 bg-slate-100 dark:bg-slate-800 rounded-md animate-pulse" />
          <div className="h-8 bg-slate-100 dark:bg-slate-800 rounded-md animate-pulse" />
        </div>
      );
    }
    
    if (!data || !data.data || !data.data.forecast) {
      return <div>No forecast data available</div>;
    }
    
    const forecast = data.data.forecast;
    const colorScheme = getColorScheme(type);
    
    if (type === 'carrier') {
      // Carrier data has different structure
      return (
        <div className="overflow-x-auto rounded-md border">
          <table className="w-full">
            <thead className="bg-slate-50 dark:bg-slate-800">
              <tr>
                <th className="py-3 px-4 text-left font-medium">Month</th>
                <th className="py-3 px-4 text-left font-medium">Top Carriers</th>
                <th className="py-3 px-4 text-right font-medium">Distribution</th>
              </tr>
            </thead>
            <tbody>
              {forecast.map((item, index) => (
                <tr 
                  key={index} 
                  className={`hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors ${
                    index !== forecast.length - 1 ? 'border-b dark:border-slate-700' : ''
                  }`}
                >
                  <td className="py-3 px-4 font-medium">{item.month}</td>
                  <td className="py-3 px-4">
                    {item.carriers.map((c, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: gradientColors[Object.keys(gradientColors)[i % 6]][0] }}></div>
                        {c.carrier}
                      </div>
                    )).slice(0, 3)}
                  </td>
                  <td className="py-3 px-4 text-right">
                    {item.carriers.map((c, i) => (
                      <div key={i} style={{ color: gradientColors[Object.keys(gradientColors)[i % 6]][0] }}>
                        {c.percentage.toFixed(1)}%
                      </div>
                    )).slice(0, 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    
    // Standard forecast table with enhanced styling
    return (
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full">
          <thead className="bg-slate-50 dark:bg-slate-800">
            <tr>
              <th className="py-3 px-4 text-left font-medium">Month</th>
              <th className="py-3 px-4 text-right font-medium">Forecast</th>
              <th className="py-3 px-4 text-right font-medium">Lower Bound</th>
              <th className="py-3 px-4 text-right font-medium">Upper Bound</th>
              <th className="py-3 px-4 text-right font-medium">Confidence Range</th>
            </tr>
          </thead>
          <tbody>
            {forecast.map((item, index) => (
              <tr 
                key={index} 
                className={`hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors ${
                  index !== forecast.length - 1 ? 'border-b dark:border-slate-700' : ''
                }`}
              >
                <td className="py-3 px-4 font-medium">{item.month}</td>
                <td className="py-3 px-4 text-right">
                  <span style={{ color: gradientColors[colorScheme][0] }}>
                    {item.forecast.toLocaleString()}
                  </span>
                </td>
                <td className="py-3 px-4 text-right">{item.lower_bound.toLocaleString()}</td>
                <td className="py-3 px-4 text-right">{item.upper_bound.toLocaleString()}</td>
                <td className="py-3 px-4 text-right">
                  <div className="flex items-center justify-end gap-2">
                    <span>±{(((item.upper_bound - item.lower_bound) / 2) / item.forecast * 100).toFixed(1)}%</span>
                    <div 
                      className="w-16 h-3 rounded-full" 
                      style={{ background: `linear-gradient(to right, ${gradientColors[colorScheme][1]}33, ${gradientColors[colorScheme][0]}aa)` }}
                    ></div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render model selection and metrics with enhanced styling
  const renderModelSelection = (type) => {
    const models = modelOptions[type] || [];
    const accuracies = modelAccuracies[type] || {};
    const colorScheme = getColorScheme(type);
    
    return (
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">Forecasting Model:</span>
            <div className="flex flex-wrap gap-2">
              {models.map(model => (
                <Button
                  key={model}
                  variant={selectedModels[type] === model ? "default" : "outline"}
                  size="sm"
                  onClick={() => handleModelChange(type, model)}
                  className="flex items-center gap-2"
                  style={{ 
                    background: selectedModels[type] === model ? gradientColors[colorScheme][0] : 'transparent',
                    borderColor: selectedModels[type] !== model ? gradientColors[colorScheme][0] : 'transparent'
                  }}
                >
                  {model === 'ml' ? 'Machine Learning' : 
                    model === 'arima' ? 'ARIMA' : 
                    model === 'historical' ? 'Historical' : 
                    model}
                  {accuracies[model] && (
                    <Badge variant="outline" className="ml-1 bg-white/10 border-0">
                      {(accuracies[model] * 100).toFixed(0)}%
                    </Badge>
                  )}
                </Button>
              ))}
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button 
              variant="outline" 
              size="sm" 
              className="flex items-center gap-1" 
              onClick={handleRefresh}
            >
              <RefreshCcw className="h-4 w-4" />
              Refresh
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              className="flex items-center gap-1"
              onClick={handleDownload}
            >
              <Download className="h-4 w-4" />
              Export
            </Button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4 bg-slate-50 dark:bg-slate-800/50 p-4 rounded-lg">
          <h4 className="text-sm font-medium col-span-full mb-2 flex items-center gap-2">
            <Activity className="h-4 w-4 text-blue-500" />
            Model Performance Metrics
          </h4>
          {models.length > 0 ? (
            <>
              {models.map(model => (
                <div key={model} className="bg-white dark:bg-slate-800 p-3 rounded-md border shadow-sm">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium text-sm">
                      {model === 'ml' ? 'Machine Learning' : 
                        model === 'arima' ? 'ARIMA' : 
                        model === 'historical' ? 'Historical' : 
                        model}
                    </span>
                    <Badge 
                      className={`${
                        accuracies[model] > 0.85 ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                        accuracies[model] > 0.75 ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400' :
                        'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400'
                      }`}
                    >
                      {accuracies[model] ? `${(accuracies[model] * 100).toFixed(1)}%` : 'N/A'}
                    </Badge>
                  </div>
                  <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full transition-all duration-500"
                      style={{
                        width: `${accuracies[model] ? accuracies[model] * 100 : 0}%`,
                        backgroundColor: accuracies[model] > 0.85 ? gradientColors.green[0] :
                                       accuracies[model] > 0.75 ? gradientColors.blue[0] :
                                       gradientColors.amber[0]
                      }}
                    />
                  </div>
                </div>
              ))}
            </>
          ) : (
            <div className="col-span-full text-sm text-gray-500 dark:text-gray-400 flex items-center justify-center p-4">
              <AlertTriangle className="h-4 w-4 mr-2" />
              No model performance data available
            </div>
          )}
        </div>
      </div>
    );
  };

  // Get title and description based on forecast type
  const getForecastInfo = (type) => {
    switch (type) {
      case 'demand':
        return {
          title: 'Shipment Volume Forecast',
          description: 'Predicted shipment volume for the next 6 months'
        };
      case 'weight':
        return {
          title: 'Weight Forecast',
          description: 'Projected total shipment weight for the next 6 months'
        };
      case 'value':
        return {
          title: 'Value Forecast',
          description: 'Forecasted monetary value of shipments for the next 6 months'
        };
      case 'carrier':
        return {
          title: 'Carrier Utilization',
          description: 'Projected carrier distribution for upcoming shipments'
        };
      case 'seasonal':
        return {
          title: 'Seasonal Analysis',
          description: 'Seasonal patterns and their impact on shipment volumes'
        };
      case 'processing':
        return {
          title: 'Processing Time Forecast',
          description: 'Projected time between document generation and actual shipping'
        };
      default:
        return {
          title: 'Forecast',
          description: 'Logistics forecast data'
        };
    }
  };

  // Main render
  return (
    <div className="space-y-6">
      {/* Header with animated background */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-to-r from-blue-900 to-indigo-900 p-6 mb-6">
        <div className="absolute inset-0 opacity-20">
          <ForecastAnimations.GridPattern size={24} className="text-white" strokeWidth={0.5} />
        </div>
        <ForecastAnimations.WaveAnimation className="text-white" />
        
        <div className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-xl font-bold text-white mb-2 flex items-center">
              <Activity className="mr-2 h-5 w-5" />
              Detailed Forecast Analytics
            </h2>
            <p className="text-blue-100 max-w-3xl">
              Explore detailed forecasts for different aspects of your logistics operations. 
              Compare different forecasting models and analyze confidence intervals.
            </p>
            
            <div className="mt-4 flex flex-wrap gap-3">
              <Badge className="bg-blue-100/20 text-blue-100 hover:bg-blue-100/30 border-0">
                <Zap className="h-3 w-3 mr-1" />
                AI-Powered
              </Badge>
              <Badge className="bg-blue-100/20 text-blue-100 hover:bg-blue-100/30 border-0">
                High Accuracy
              </Badge>
              <Badge className="bg-blue-100/20 text-blue-100 hover:bg-blue-100/30 border-0">
                Multi-Model
              </Badge>
            </div>
          </motion.div>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <div className="flex items-center space-x-4 pb-2 border-b">
          <TabsList className="flex p-1 bg-transparent">
            <TabsTrigger 
              value="demand" 
              className="flex gap-1 items-center rounded-md px-3 py-2 text-sm transition-all data-[state=active]:bg-blue-50 data-[state=active]:text-blue-700 dark:data-[state=active]:bg-blue-900/20 dark:data-[state=active]:text-blue-400"
            >
              <TrendingUp className="h-4 w-4" />
              <span>Demand</span>
            </TabsTrigger>
            <TabsTrigger 
              value="weight" 
              className="flex gap-1 items-center rounded-md px-3 py-2 text-sm transition-all data-[state=active]:bg-green-50 data-[state=active]:text-green-700 dark:data-[state=active]:bg-green-900/20 dark:data-[state=active]:text-green-400"
            >
              <Package className="h-4 w-4" />
              <span>Weight</span>
            </TabsTrigger>
            <TabsTrigger 
              value="value" 
              className="flex gap-1 items-center rounded-md px-3 py-2 text-sm transition-all data-[state=active]:bg-amber-50 data-[state=active]:text-amber-700 dark:data-[state=active]:bg-amber-900/20 dark:data-[state=active]:text-amber-400"
            >
              <DollarSign className="h-4 w-4" />
              <span>Value</span>
            </TabsTrigger>
            <TabsTrigger 
              value="carrier" 
              className="flex gap-1 items-center rounded-md px-3 py-2 text-sm transition-all data-[state=active]:bg-cyan-50 data-[state=active]:text-cyan-700 dark:data-[state=active]:bg-cyan-900/20 dark:data-[state=active]:text-cyan-400"
            >
              <TruckIcon className="h-4 w-4" />
              <span>Carriers</span>
            </TabsTrigger>
            <TabsTrigger 
              value="seasonal" 
              className="flex gap-1 items-center rounded-md px-3 py-2 text-sm transition-all data-[state=active]:bg-purple-50 data-[state=active]:text-purple-700 dark:data-[state=active]:bg-purple-900/20 dark:data-[state=active]:text-purple-400"
            >
              <Calendar className="h-4 w-4" />
              <span>Seasonal</span>
            </TabsTrigger>
            <TabsTrigger 
              value="processing" 
              className="flex gap-1 items-center rounded-md px-3 py-2 text-sm transition-all data-[state=active]:bg-pink-50 data-[state=active]:text-pink-700 dark:data-[state=active]:bg-pink-900/20 dark:data-[state=active]:text-pink-400"
            >
              <Clock className="h-4 w-4" />
              <span>Processing</span>
            </TabsTrigger>
          </TabsList>
        </div>
        
        {['demand', 'weight', 'value', 'carrier', 'seasonal', 'processing'].map(type => (
          <TabsContent key={type} value={type} className="space-y-4">
            <motion.div 
              className="space-y-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <Card 
                className="border-l-4 relative overflow-hidden dark:bg-slate-900" 
                style={{ borderLeftColor: gradientColors[getColorScheme(type)][0] }}
              >
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-full" style={{ 
                      backgroundColor: `rgba(${getColorScheme(type) === 'blue' ? '59, 130, 246, 0.1' : 
                        getColorScheme(type) === 'green' ? '16, 185, 129, 0.1' : 
                        getColorScheme(type) === 'amber' ? '245, 158, 11, 0.1' : 
                        getColorScheme(type) === 'purple' ? '139, 92, 246, 0.1' : 
                        getColorScheme(type) === 'pink' ? '236, 72, 153, 0.1' : 
                        '6, 182, 212, 0.1'})` 
                    }}>
                      {React.cloneElement(getIcon(type), { 
                        className: "h-5 w-5", 
                        style: { color: gradientColors[getColorScheme(type)][0] }
                      })}
                    </div>
                    <div>
                      <CardTitle>{getForecastInfo(type).title}</CardTitle>
                      <CardDescription>{getForecastInfo(type).description}</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {renderModelSelection(type)}
                </CardContent>
              </Card>

              <Card className="overflow-hidden">
                <CardHeader>
                  <CardTitle>Forecast Visualization</CardTitle>
                  <CardDescription>Interactive chart showing forecasted values with confidence intervals</CardDescription>
                </CardHeader>
                <CardContent className="bg-slate-900 p-0">
                  {renderChart(type, forecastData[type])}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Forecast Details</CardTitle>
                  <CardDescription>Monthly forecasts with confidence intervals</CardDescription>
                </CardHeader>
                <CardContent>
                  {renderTable(type, forecastData[type])}
                </CardContent>
                <CardFooter className="border-t pt-4 flex justify-between">
                  <Badge variant="outline" className="flex items-center gap-1">
                    <Zap className="h-3.5 w-3.5" />
                    AI-Powered Forecast
                  </Badge>
                  <div className="text-xs text-muted-foreground">
                    Last updated: {new Date().toLocaleDateString()}
                  </div>
                </CardFooter>
              </Card>
            </motion.div>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}