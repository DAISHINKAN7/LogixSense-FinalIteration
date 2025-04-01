// src/app/forecasting/components/ForecastingDashboard.jsx
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ArrowRight, TrendingUp, Package, DollarSign, Truck, Calendar, Clock } from 'lucide-react';

// API URL
const API_URL = 'http://localhost:8000/api';

export default function ForecastingDashboard() {
  const [isLoading, setIsLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  const [error, setError] = useState(null);

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

  // Chart colors
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  // Skeleton loader
  const ChartSkeleton = () => (
    <div className="w-full h-[200px] bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse" />
  );

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
          month: item.month.substring(0, 3), // First 3 chars of month
          forecast: item.forecast,
          lowerBound: item.lower_bound,
          upperBound: item.upper_bound
        }));
      
      case 'seasonal':
        return forecast.map(item => ({
          month: item.month.substring(0, 3),
          forecast: item.forecast,
          seasonalFactor: item.seasonal_factor
        }));
      
      default:
        return forecast;
    }
  };

  // Calculate growth trend
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

  // Render different charts based on forecast type
  const renderOverviewChart = (type, data) => {
    if (isLoading) return <ChartSkeleton />;
    if (!data) return <div className="h-[200px] flex items-center justify-center">No data available</div>;
    
    const chartData = formatChartData(type, data);
    
    if (chartData.length === 0) {
      return <div className="h-[200px] flex items-center justify-center">No forecast data available</div>;
    }
    
    switch (type) {
      case 'demand':
        return (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <XAxis dataKey="month" tickLine={false} axisLine={false} />
              <YAxis hide={true} />
              <Tooltip />
              <Line type="monotone" dataKey="forecast" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        );
      
      case 'weight':
        return (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <XAxis dataKey="month" tickLine={false} axisLine={false} />
              <YAxis hide={true} />
              <Tooltip />
              <Bar dataKey="forecast" fill="#00C49F" />
            </BarChart>
          </ResponsiveContainer>
        );
      
      case 'value':
        return (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <XAxis dataKey="month" tickLine={false} axisLine={false} />
              <YAxis hide={true} />
              <Tooltip formatter={(value) => value.toLocaleString()} />
              <Line type="monotone" dataKey="forecast" stroke="#FFBB28" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        );
      
      case 'seasonal':
        return (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <XAxis dataKey="month" tickLine={false} axisLine={false} />
              <YAxis hide={true} />
              <Tooltip />
              <Line type="monotone" dataKey="seasonalFactor" stroke="#FF8042" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        );
      
      case 'processing':
        return (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <XAxis dataKey="month" tickLine={false} axisLine={false} />
              <YAxis hide={true} />
              <Tooltip />
              <Bar dataKey="forecast" fill="#8884D8" />
            </BarChart>
          </ResponsiveContainer>
        );
      
      default:
        return <div>No chart available</div>;
    }
  };

  // Get growth trends
  const getDemandGrowth = () => calculateGrowth(dashboardData?.demand);
  const getWeightGrowth = () => calculateGrowth(dashboardData?.weight);
  const getValueGrowth = () => calculateGrowth(dashboardData?.value);
  const getProcessingGrowth = () => calculateGrowth(dashboardData?.processing);

  // Get card icon by type
  const getCardIcon = (type) => {
    switch (type) {
      case 'demand':
        return <TrendingUp className="h-6 w-6 text-blue-500" />;
      case 'weight':
        return <Package className="h-6 w-6 text-green-500" />;
      case 'value':
        return <DollarSign className="h-6 w-6 text-yellow-500" />;
      case 'seasonal':
        return <Calendar className="h-6 w-6 text-orange-500" />;
      case 'processing':
        return <Clock className="h-6 w-6 text-purple-500" />;
      default:
        return <TrendingUp className="h-6 w-6" />;
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

  // Get seasonal peak periods
  const getPeakPeriods = () => {
    if (!dashboardData?.seasonal?.data?.peak_periods) {
      return { high: [], low: [] };
    }
    
    return {
      high: dashboardData.seasonal.data.peak_periods.high_season || [],
      low: dashboardData.seasonal.data.peak_periods.low_season || []
    };
  };

  if (error) {
    return (
      <Alert variant="destructive" className="mb-6">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Forecasting Overview</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Demand Forecast Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Shipment Volume</CardTitle>
            {getCardIcon('demand')}
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? 'Loading...' : getLatestForecast(dashboardData?.demand)}
            </div>
            <p className="text-xs text-muted-foreground">
              Projected for {isLoading ? '...' : dashboardData?.demand?.data?.forecast?.[dashboardData.demand.data.forecast.length - 1]?.month || 'future'}
            </p>
            {!isLoading && dashboardData?.demand && (
              <div className="mt-1">
                <span className={`text-xs ${getDemandGrowth().isPositive ? 'text-green-500' : 'text-red-500'}`}>
                  {getDemandGrowth().isPositive ? '↑' : '↓'} {getDemandGrowth().value}%
                </span>
                <span className="text-xs text-muted-foreground ml-1">growth trend</span>
              </div>
            )}
            <div className="mt-4">
              {renderOverviewChart('demand', dashboardData?.demand)}
            </div>
          </CardContent>
        </Card>

        {/* Weight Forecast Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Weight (kg)</CardTitle>
            {getCardIcon('weight')}
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? 'Loading...' : getLatestForecast(dashboardData?.weight)}
            </div>
            <p className="text-xs text-muted-foreground">
              Projected for {isLoading ? '...' : dashboardData?.weight?.data?.forecast?.[dashboardData.weight.data.forecast.length - 1]?.month || 'future'}
            </p>
            {!isLoading && dashboardData?.weight && (
              <div className="mt-1">
                <span className={`text-xs ${getWeightGrowth().isPositive ? 'text-green-500' : 'text-red-500'}`}>
                  {getWeightGrowth().isPositive ? '↑' : '↓'} {getWeightGrowth().value}%
                </span>
                <span className="text-xs text-muted-foreground ml-1">growth trend</span>
              </div>
            )}
            <div className="mt-4">
              {renderOverviewChart('weight', dashboardData?.weight)}
            </div>
          </CardContent>
        </Card>

        {/* Value Forecast Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Shipment Value (₹)</CardTitle>
            {getCardIcon('value')}
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? 'Loading...' : getLatestForecast(dashboardData?.value)}
            </div>
            <p className="text-xs text-muted-foreground">
              Projected for {isLoading ? '...' : dashboardData?.value?.data?.forecast?.[dashboardData.value.data.forecast.length - 1]?.month || 'future'}
            </p>
            {!isLoading && dashboardData?.value && (
              <div className="mt-1">
                <span className={`text-xs ${getValueGrowth().isPositive ? 'text-green-500' : 'text-red-500'}`}>
                  {getValueGrowth().isPositive ? '↑' : '↓'} {getValueGrowth().value}%
                </span>
                <span className="text-xs text-muted-foreground ml-1">growth trend</span>
              </div>
            )}
            <div className="mt-4">
              {renderOverviewChart('value', dashboardData?.value)}
            </div>
          </CardContent>
        </Card>

        {/* Seasonal Patterns Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Seasonal Patterns</CardTitle>
            {getCardIcon('seasonal')}
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div>
                <span className="text-xs font-medium">Peak Seasons:</span>
                <span className="text-xs ml-2">
                  {isLoading ? 'Loading...' : getPeakPeriods().high.join(', ') || 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-xs font-medium">Low Seasons:</span>
                <span className="text-xs ml-2">
                  {isLoading ? 'Loading...' : getPeakPeriods().low.join(', ') || 'N/A'}
                </span>
              </div>
            </div>
            <div className="mt-4">
              {renderOverviewChart('seasonal', dashboardData?.seasonal)}
            </div>
          </CardContent>
        </Card>

        {/* Processing Time Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing Time (days)</CardTitle>
            {getCardIcon('processing')}
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? 'Loading...' : getLatestForecast(dashboardData?.processing)}
            </div>
            <p className="text-xs text-muted-foreground">
              Projected for {isLoading ? '...' : dashboardData?.processing?.data?.forecast?.[dashboardData.processing.data.forecast.length - 1]?.month || 'future'}
            </p>
            {!isLoading && dashboardData?.processing && (
              <div className="mt-1">
                <span className={`text-xs ${!getProcessingGrowth().isPositive ? 'text-green-500' : 'text-red-500'}`}>
                  {!getProcessingGrowth().isPositive ? '↓' : '↑'} {getProcessingGrowth().value}%
                </span>
                <span className="text-xs text-muted-foreground ml-1">
                  {!getProcessingGrowth().isPositive ? 'faster' : 'slower'} processing
                </span>
              </div>
            )}
            <div className="mt-4">
              {renderOverviewChart('processing', dashboardData?.processing)}
            </div>
          </CardContent>
        </Card>

        {/* View All Button */}
        <Card className="flex items-center justify-center">
          <CardContent className="pt-6">
            <Button variant="outline" size="lg" className="w-full" onClick={() => {}}>
              View Detailed Forecasts
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}