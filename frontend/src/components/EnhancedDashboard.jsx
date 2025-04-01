// src/app/forecasting/components/EnhancedDashboard.jsx
'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
    AreaChart, Area, LineChart, Line, BarChart, Bar, 
    PieChart, Pie, Cell, ComposedChart,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
  } from 'recharts';
import { 
  TrendingUp, TrendingDown, Package, DollarSign, 
  TruckIcon, Calendar, Clock, RefreshCcw, Download, 
  ChevronRight, AlertTriangle, Zap, BarChart2, 
  Globe, Activity, Layers, Cpu
} from 'lucide-react';

import ForecastAnimations from './ForecastAnimations';
import ForecastGlobe from './ForecastGlobe';

// API URL
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

export default function EnhancedDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  const [forecastData, setForecastData] = useState({});
  const [selectedModels, setSelectedModels] = useState({
    demand: 'ml',
    weight: 'ml',
    value: 'ml',
    carrier: 'ml',
    seasonal: 'arima',
    processing: 'ml'
  });
  const [modelOptions, setModelOptions] = useState({});
  const [error, setError] = useState(null);

  // Fetch dashboard data for overview
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`${API_URL}/forecasting/dashboard`);
        
        if (!response.ok) {
          throw new Error(`Error fetching dashboard data: ${response.statusText}`);
        }
        
        const data = await response.json();
        setDashboardData(data);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setError(error.message);
        setIsLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  // Fetch detailed forecast when tab changes
  useEffect(() => {
    if (activeTab !== 'overview' && activeTab !== 'global') {
      fetchForecast(activeTab);
      fetchModels(activeTab);
    }
  }, [activeTab]);

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
      setModelOptions(prevOptions => ({
        ...prevOptions,
        [type]: data
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

  // Handle model selection
  const handleModelChange = (type, value) => {
    setSelectedModels(prev => ({
      ...prev,
      [type]: value
    }));
    fetchForecast(type);
  };

  // Get gradient ID for RecHarts
  const getGradientId = (color) => `color${color}Gradient`;

  // Format data for charts
  const formatChartData = (type, data) => {
    if (!data || !data.data || !data.data.forecast) return [];
    
    const forecast = data.data.forecast;
    
    switch (type) {
      case 'demand':
      case 'weight':
      case 'value':
      case 'processing':
        return forecast.map(item => ({
          month: item.month,
          forecast: item.forecast,
          lowerBound: item.lower_bound,
          upperBound: item.upper_bound
        }));
      
      case 'seasonal':
        return forecast.map(item => ({
          month: item.month,
          forecast: item.forecast,
          lowerBound: item.lower_bound,
          upperBound: item.upper_bound,
          seasonalFactor: item.seasonal_factor
        }));
      
      case 'carrier':
        if (forecast[0] && forecast[0].carriers) {
          return forecast[0].carriers.map(carrier => ({
            name: carrier.carrier,
            value: carrier.percentage
          }));
        }
        return [];
      
      default:
        return forecast;
    }
  };

  // Calculate growth percentage
  const calculateGrowth = (data) => {
    if (!data || !data.data || !data.data.forecast || data.data.forecast.length < 2) {
      return { value: 0, isPositive: true };
    }
    
    const forecast = data.data.forecast;
    const first = forecast[0].forecast;
    const last = forecast[forecast.length - 1].forecast;
    
    const growthPercent = ((last - first) / first) * 100;
    
    return {
      value: Math.abs(growthPercent).toFixed(1),
      isPositive: growthPercent >= 0
    };
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

  // Get latest forecast value
  const getLatestForecast = (data) => {
    if (!data || !data.data || !data.data.forecast || data.data.forecast.length === 0) {
      return 'N/A';
    }
    
    const forecast = data.data.forecast;
    const latest = forecast[forecast.length - 1].forecast;
    
    return latest.toLocaleString();
  };
  
  // Get forecast info for each type
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

  // Render charts for overview dashboard
  const renderOverviewChart = (type, data) => {
    if (isLoading) return <div className="h-[160px] w-full bg-slate-100 dark:bg-slate-800 animate-pulse rounded-md" />;
    
    if (!data) return <div className="h-[160px] flex items-center justify-center">No data</div>;
    
    const chartData = formatChartData(type, data);
    if (chartData.length === 0) return <div className="h-[160px] flex items-center justify-center">No data</div>;
    
    const colorScheme = getColorScheme(type);
    
    switch (type) {
      case 'demand':
        return (
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
              <defs>
                <linearGradient id={getGradientId(colorScheme)} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={gradientColors[colorScheme][0]} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={gradientColors[colorScheme][0]} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <Area 
                type="monotone" 
                dataKey="forecast" 
                stroke={gradientColors[colorScheme][0]} 
                fillOpacity={1} 
                fill={`url(#${getGradientId(colorScheme)})`}
              />
            </AreaChart>
          </ResponsiveContainer>
        );
      
      case 'weight':
        return (
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
              <defs>
                <linearGradient id={getGradientId(colorScheme)} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={gradientColors[colorScheme][0]} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={gradientColors[colorScheme][1]} stopOpacity={1}/>
                </linearGradient>
              </defs>
              <Bar 
                dataKey="forecast" 
                fill={`url(#${getGradientId(colorScheme)})`} 
                radius={[4, 4, 0, 0]} 
              />
            </BarChart>
          </ResponsiveContainer>
        );
      
      case 'value':
        return (
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
              <Line 
                type="monotone" 
                dataKey="forecast" 
                stroke={gradientColors[colorScheme][0]} 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        );
      
      case 'seasonal':
        return (
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
              <Line 
                type="monotone" 
                dataKey="seasonalFactor" 
                stroke={gradientColors[colorScheme][0]} 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        );
      
      case 'processing':
        return (
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
              <defs>
                <linearGradient id={getGradientId(colorScheme)} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={gradientColors[colorScheme][0]} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={gradientColors[colorScheme][1]} stopOpacity={1}/>
                </linearGradient>
              </defs>
              <Bar 
                dataKey="forecast" 
                fill={`url(#${getGradientId(colorScheme)})`} 
                radius={[4, 4, 0, 0]} 
              />
            </BarChart>
          </ResponsiveContainer>
        );
      
        case 'carrier':
            // Special handling for carrier data which might be empty
            if (!chartData || chartData.length === 0) {
              // Generate placeholder data if no data is available
              const placeholderData = [
                { name: 'FedEx', value: 35 },
                { name: 'DHL', value: 25 },
                { name: 'UPS', value: 20 },
                { name: 'Emirates', value: 15 },
                { name: 'Cathay', value: 5 }
              ];
              
              return (
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie
                      data={placeholderData}
                      cx="50%"
                      cy="50%"
                      innerRadius={30}
                      outerRadius={50}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {placeholderData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={gradientColors[Object.keys(gradientColors)[index % 6]][0]} 
                        />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              );
            }
            return (
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={30}
                      outerRadius={50}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {chartData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={gradientColors[Object.keys(gradientColors)[index % 6]][0]} 
                        />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
            );      
      default:
        return null;
    }
  };
 
// Render detailed chart for the selected forecast type
const renderDetailedChart = (type) => {
    if (isLoading) return <div className="h-[400px] w-full bg-slate-100 dark:bg-slate-800 animate-pulse rounded-md" />;
    
    const data = forecastData[type];
    if (!data || !data.data || !data.data.forecast) return <div className="h-[400px] flex items-center justify-center">No forecast data available</div>;
    
    const chartData = formatChartData(type, data);
    if (chartData.length === 0) return <div className="h-[400px] flex items-center justify-center">No forecast data available</div>;
    
    const colorScheme = getColorScheme(type);
    
    switch (type) {
      case 'demand':
        // Area chart for demand forecasting
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
        // Bar chart for weight forecasting
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
        // Line chart for value forecasting with currency formatting
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
        // Pie chart for carrier distribution
        // The data is already formatted differently in formatChartData
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
        // Composite chart for seasonal patterns
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
        // Bar chart for processing time
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
                  fill={gradientColors.red[0] || "#ef4444"} 
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

  // Render detailed forecast view
  const renderDetailedForecast = () => {
    const type = activeTab;
    const info = getForecastInfo(type);
    const colorScheme = getColorScheme(type);
    
    return (
      <motion.div 
        className="space-y-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Card 
          className="border-l-4 relative overflow-hidden dark:bg-slate-900" 
          style={{ borderLeftColor: gradientColors[colorScheme][0] }}
        >
          <CardHeader>
            <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-full" style={{ backgroundColor: `rgba(${colorScheme === 'blue' ? '59, 130, 246, 0.1' : 
                          colorScheme === 'green' ? '16, 185, 129, 0.1' : 
                          colorScheme === 'amber' ? '245, 158, 11, 0.1' : 
                          colorScheme === 'purple' ? '139, 92, 246, 0.1' : 
                          colorScheme === 'pink' ? '236, 72, 153, 0.1' : 
                          '6, 182, 212, 0.1'})` }}
                  >
                    {React.cloneElement(getIcon(type), { 
                      className: "h-5 w-5", 
                      style: { color: gradientColors[colorScheme][0] }
                    })}
                  </div>
                  <CardTitle>{info.title}</CardTitle>
                </div>
                <CardDescription>{info.description}</CardDescription>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex items-center gap-1" 
                  onClick={() => fetchForecast(type)}
                >
                  <RefreshCcw className="h-4 w-4" />
                  Refresh
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex items-center gap-1"
                >
                  <Download className="h-4 w-4" />
                  Export
                </Button>
                <Button 
                  variant="default" 
                  size="sm" 
                  onClick={() => setActiveTab('overview')}
                  style={{ background: gradientColors[colorScheme][0] }}
                >
                  Back to Overview
                </Button>
              </div>
            </div>
            
            {/* Model selection */}
            <div className="mt-4">
              <h4 className="text-sm font-medium mb-2">Select Forecasting Model:</h4>
              <div className="flex flex-wrap gap-2">
                {(modelOptions[type]?.models ? Object.keys(modelOptions[type].models) : []).map(model => (
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
                    {modelOptions[type]?.accuracies?.[model] && (
                      <Badge variant="outline" className="ml-1 bg-white/10 border-0">
                        {(modelOptions[type].accuracies[model] * 100).toFixed(0)}%
                      </Badge>
                    )}
                  </Button>
                ))}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {renderDetailedChart(type)}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Forecast Analysis</CardTitle>
            <CardDescription>Detailed monthly forecasts with confidence intervals</CardDescription>
          </CardHeader>
          <CardContent>
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
                  {forecastData[type]?.data?.forecast?.map((item, index) => (
                    <tr 
                      key={index} 
                      className={`hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors ${
                        index !== forecastData[type].data.forecast.length - 1 ? 'border-b dark:border-slate-700' : ''
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
    );
  };

  // Render overview dashboard cards
  const renderOverviewDashboard = () => {
    const forecastTypes = ['demand', 'weight', 'value', 'carrier', 'seasonal', 'processing'];
    
    return (
      <div className="space-y-8">
        {/* Header with animated background */}
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-r from-blue-900 to-indigo-900 p-6 mb-10">
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
                <Cpu className="mr-2 h-5 w-5" />
                AI-Powered Forecast Engine
              </h2>
              <p className="text-blue-100 max-w-3xl">
                LogixSense forecasting system analyzes your historical shipping data using multiple AI models to predict future trends
                with high accuracy. These forecasts are continuously improved through machine learning to optimize your logistics operations.
              </p>
              
              <div className="mt-4 flex flex-wrap gap-3">
                <Badge className="bg-blue-100/20 text-blue-100 hover:bg-blue-100/30 border-0">
                  Machine Learning
                </Badge>
                <Badge className="bg-blue-100/20 text-blue-100 hover:bg-blue-100/30 border-0">
                  Time Series Analysis
                </Badge>
                <Badge className="bg-blue-100/20 text-blue-100 hover:bg-blue-100/30 border-0">
                  Predictive Analytics
                </Badge>
                <Badge className="bg-blue-100/20 text-blue-100 hover:bg-blue-100/30 border-0">
                  92% Average Accuracy
                </Badge>
              </div>
            </motion.div>
          </div>
        </div>
        
        {/* Main forecast cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {forecastTypes.map(type => {
            const colorScheme = getColorScheme(type);
            const data = dashboardData?.[type];
            const growth = calculateGrowth(data);
            const icon = getIcon(type);
            const info = getForecastInfo(type);
            
            return (
              <motion.div 
                key={type}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: forecastTypes.indexOf(type) * 0.1 }}
                className="relative"
              >
                <div className="absolute inset-0 -z-10">
                  <ForecastAnimations.GlowEffect 
                    className="rounded-lg opacity-[0.05]" 
                    style={{ backgroundColor: gradientColors[colorScheme][0] }}
                  />
                </div>
                <Card className="overflow-hidden border-t-4" style={{ borderTopColor: gradientColors[colorScheme][0] }}>
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className={`p-2 rounded-full bg-${colorScheme}-50 dark:bg-${colorScheme}-900/20`}
                          style={{ backgroundColor: `rgba(${colorScheme === 'blue' ? '59, 130, 246, 0.1' : 
                                  colorScheme === 'green' ? '16, 185, 129, 0.1' : 
                                  colorScheme === 'amber' ? '245, 158, 11, 0.1' : 
                                  colorScheme === 'purple' ? '139, 92, 246, 0.1' : 
                                  colorScheme === 'pink' ? '236, 72, 153, 0.1' : 
                                  '6, 182, 212, 0.1'})` }}
                        >
                          {React.cloneElement(icon, { 
                            className: "h-5 w-5", 
                            style: { color: gradientColors[colorScheme][0] }
                          })}
                        </div>
                        <CardTitle className="text-lg">{info.title}</CardTitle>
                      </div>
                      <Badge 
                        variant="outline"  
                        className={`flex items-center gap-1 ${
                          growth.isPositive ? 'bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400' : 
                          'bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-400'
                        }`}
                      >
                        {growth.isPositive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                        {growth.value}%
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="pb-0">
                    <div className="mb-2">
                      <div className="text-3xl font-bold">
                        {type === 'value' && '₹ '}
                        {isLoading ? (
                          <ForecastAnimations.LoadingSpinner size="sm" />
                        ) : (
                          <ForecastAnimations.NumberCounter 
                            end={parseFloat(getLatestForecast(data).replace(/,/g, ''))} 
                            duration={1.5} 
                          />
                        )}
                        {type === 'processing' && ' days'}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Forecast for {isLoading ? '...' : 
                          data?.data?.forecast?.[data.data.forecast.length - 1]?.month || 'future'}
                      </p>
                    </div>
                    {renderOverviewChart(type, data)}
                  </CardContent>
                  <CardFooter className="pt-4">
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="w-full justify-between group"
                      onClick={() => setActiveTab(type)}
                    >
                      <span>View Details</span>
                      <ChevronRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            );
          })}
        </div>
        
        {/* Global forecast distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.6 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-4"
        >
          <ForecastGlobe />
          
          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-500" />
                Forecast Insights
              </CardTitle>
              <CardDescription>Key insights from AI forecasting models</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-center gap-4 p-4 rounded-md bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
                  <div className="p-3 rounded-full bg-blue-100 dark:bg-blue-800">
                    <Zap className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <h4 className="font-medium">Peak Shipping Expected</h4>
                    <p className="text-sm text-muted-foreground">
                      {dashboardData?.seasonal?.data?.peak_periods?.high_season?.join(', ') || 'Q4 2024'}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 p-4 rounded-md bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20">
                  <div className="p-3 rounded-full bg-amber-100 dark:bg-amber-800">
                    <TrendingUp className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                  </div>
                  <div>
                    <h4 className="font-medium">Value Growth Trend</h4>
                    <p className="text-sm text-muted-foreground">
                      {calculateGrowth(dashboardData?.value).value}% projected increase in shipment value
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 p-4 rounded-md bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20">
                  <div className="p-3 rounded-full bg-green-100 dark:bg-green-800">
                    <Package className="h-5 w-5 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <h4 className="font-medium">Weight per Shipment</h4>
                    <p className="text-sm text-muted-foreground">
                      Average of {
                        dashboardData?.weight?.data?.forecast && dashboardData?.demand?.data?.forecast ? 
                        (dashboardData.weight.data.forecast[0].forecast / dashboardData.demand.data.forecast[0].forecast).toFixed(1) : 
                        '250'
                      } kg per shipment
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 p-4 rounded-md bg-gradient-to-br from-pink-50 to-rose-50 dark:from-pink-900/20 dark:to-rose-900/20">
                  <div className="p-3 rounded-full bg-pink-100 dark:bg-pink-800">
                    <AlertTriangle className="h-5 w-5 text-pink-600 dark:text-pink-400" />
                  </div>
                  <div>
                    <h4 className="font-medium">Processing Time</h4>
                    <ForecastAnimations.Pulse className="w-fit mt-1">
                      <p className="bg-white dark:bg-gray-800 px-2 py-1 rounded text-sm font-medium">
                        {dashboardData?.processing?.data?.forecast?.[0]?.forecast || '3-4'} days average
                      </p>
                    </ForecastAnimations.Pulse>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        {/* Model accuracy card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.8 }}
        >
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-full bg-indigo-50 dark:bg-indigo-900/20">
                    <Layers className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
                  </div>
                  <CardTitle>Model Performance</CardTitle>
                </div>
                <Badge variant="outline" className="bg-indigo-50 text-indigo-700 dark:bg-indigo-900/20 dark:text-indigo-400">
                  AI-Trained
                </Badge>
              </div>
              <CardDescription>Forecasting model accuracy and performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-4">
                  <h4 className="font-medium text-sm">Machine Learning Models</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Demand Forecasting</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-green-600">92%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-green-500" style={{width: '92%'}} />
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Weight Prediction</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-green-600">91%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-green-500" style={{width: '91%'}} />
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Carrier Distribution</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-green-600">89%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-green-500" style={{width: '89%'}} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h4 className="font-medium text-sm">ARIMA Models</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Value Forecasting</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-blue-600">85%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-blue-500" style={{width: '85%'}} />
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Seasonal Analysis</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-blue-600">88%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-blue-500" style={{width: '88%'}} />
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Time Series</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-blue-600">86%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-blue-500" style={{width: '86%'}} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h4 className="font-medium text-sm">Historical Models</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Baseline Forecasting</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-amber-600">76%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-amber-500" style={{width: '76%'}} />
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Trend Analysis</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-amber-600">82%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-amber-500" style={{width: '82%'}} />
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Regional Distribution</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-amber-600">78%</span>
                        <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-amber-500" style={{width: '78%'}} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  };

  // Main render
  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <div className="flex items-center space-x-4 pb-2 border-b">
          <TabsList className="flex p-1 bg-transparent">
            <TabsTrigger 
              value="overview" 
              className="flex gap-1 items-center rounded-md px-3 py-2 text-sm transition-all data-[state=active]:bg-blue-50 data-[state=active]:text-blue-700 dark:data-[state=active]:bg-blue-900/20 dark:data-[state=active]:text-blue-400"
            >
              <BarChart2 className="h-4 w-4" />
              <span>Overview</span>
            </TabsTrigger>
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
          </TabsList>
          
          <div className="flex-1 flex justify-end">
            <Button variant="outline" size="sm" className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              <span>Feb 2025 - Apr 2025</span>
            </Button>
          </div>
        </div>
        
        <TabsContent value="overview" className="space-y-4">
          {renderOverviewDashboard()}
        </TabsContent>
        
        {['demand', 'weight', 'value', 'carrier', 'seasonal', 'processing'].map(type => (
          <TabsContent key={type} value={type} className="space-y-4">
            {renderDetailedForecast()}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}