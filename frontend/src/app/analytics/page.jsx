// src/app/analytics/page.jsx
'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, ScatterChart, Scatter,
  Cell, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, 
  PolarRadiusAxis, Radar, ComposedChart, Area
} from 'recharts';
import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react';

// API base URL - change this for production
const API_BASE_URL = 'http://localhost:8000/api';

// Chart placeholder for loading or error state
const ChartPlaceholder = ({ isLoading, error, onRetry }) => (
  <div className="h-[300px] w-full flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-md">
    {isLoading ? (
      <div className="flex flex-col items-center text-gray-500">
        <Loader2 className="h-8 w-8 animate-spin mb-2" />
        <p>Loading data...</p>
      </div>
    ) : error ? (
      <div className="flex flex-col items-center text-gray-500 px-4">
        <AlertTriangle className="h-8 w-8 mb-2 text-amber-500" />
        <p className="text-center mb-2">{error}</p>
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    ) : null}
  </div>
);

export default function AnalyticsPage() {
  const [activeTab, setActiveTab] = useState('shipment');
  const [timeRange, setTimeRange] = useState('last30days');
  
  // Loading and error states for each data category
  const [loadingStates, setLoadingStates] = useState({
    temporal: true,
    geographic: true,
    valueWeight: true,
    carrier: true,
    clustering: true,
    predictive: true,
    correlation: true
  });
  
  const [errorStates, setErrorStates] = useState({
    temporal: null,
    geographic: null,
    valueWeight: null,
    carrier: null,
    clustering: null,
    predictive: null,
    correlation: null
  });
  
  // Data state
  const [data, setData] = useState({
    temporal: {
      monthlyShipments: [],
      seasonalValues: [],
      processingTimeDistribution: [],
      dailyTrends: []
    },
    geographic: {
      topRoutes: [],
      originDestinationMatrix: []
    },
    valueWeight: {
      valueDistribution: [],
      weightValueRelationship: [],
      valuePerWeightByAirline: [],
      weightDistribution: [],
      seasonalAnalysis: []
    },
    carrier: {
      carrierMetrics: []
    },
    clustering: {
      clusters: []
    },
    predictive: {
      featureImportance: [],
      modelPerformance: []
    },
    correlation: {
      correlationMatrix: []
    }
  });

  // Colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FF6B6B', '#6A0572', '#AB83A1'];

  // Function to fetch data from a specific endpoint
  const fetchData = async (endpoint, dataType) => {
    // Update loading state
    setLoadingStates(prev => ({ ...prev, [dataType]: true }));
    // Clear previous error
    setErrorStates(prev => ({ ...prev, [dataType]: null }));
    
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/${endpoint}`);
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Update data state
      setData(prev => ({ ...prev, [dataType]: result }));
    } catch (error) {
      console.error(`Error fetching ${dataType} data:`, error);
      setErrorStates(prev => ({ 
        ...prev, 
        [dataType]: `Failed to load ${dataType} data: ${error.message}` 
      }));
    } finally {
      setLoadingStates(prev => ({ ...prev, [dataType]: false }));
    }
  };

  // Fetch all data on initial load and when time range changes
  useEffect(() => {
    fetchData('temporal', 'temporal');
    fetchData('geographic', 'geographic');
    fetchData('value-weight', 'valueWeight');
    fetchData('carrier', 'carrier');
    fetchData('clustering', 'clustering');
    fetchData('predictive', 'predictive');
    fetchData('correlation', 'correlation');
  }, [timeRange]);

  // Handle retry for a specific data type
  const handleRetry = (dataType, endpoint) => {
    fetchData(endpoint, dataType);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Logistics Analytics</h1>
        <div className="flex items-center space-x-2">
          <select 
            className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md text-sm p-2"
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <option value="last30days">Last 30 days</option>
            <option value="last3months">Last 3 months</option>
            <option value="last6months">Last 6 months</option>
            <option value="lastyear">Last year</option>
            <option value="custom">Custom range</option>
          </select>
          <Button>
            Export
          </Button>
        </div>
      </div>

      <Tabs defaultValue={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-4 lg:grid-cols-8 mb-4">
          <TabsTrigger value="shipment">Shipment</TabsTrigger>
          <TabsTrigger value="geographic">Geographic</TabsTrigger>
          <TabsTrigger value="value">Value Analysis</TabsTrigger>
          <TabsTrigger value="carrier">Carrier</TabsTrigger>
          <TabsTrigger value="time">Time Analysis</TabsTrigger>
          <TabsTrigger value="clustering">Clusters</TabsTrigger>
          <TabsTrigger value="predictive">Predictive</TabsTrigger>
          <TabsTrigger value="correlation">Correlation</TabsTrigger>
        </TabsList>

        {/* Shipment Analysis Tab */}
        <TabsContent value="shipment" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Monthly Shipment Volume */}
            <Card>
              <CardHeader>
                <CardTitle>Monthly Shipment Volume</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.temporal || errorStates.temporal ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.temporal} 
                    error={errorStates.temporal}
                    onRetry={() => handleRetry('temporal', 'temporal')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.temporal.monthlyShipments}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" name="Shipments" fill="#0088FE" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Weight Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Shipment Weight Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.valueWeight || errorStates.valueWeight ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.valueWeight} 
                    error={errorStates.valueWeight}
                    onRetry={() => handleRetry('valueWeight', 'value-weight')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={data.valueWeight.weightDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={true}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="count"
                        nameKey="range"
                        label={({ range, percent }) => `${range}: ${(percent * 100).toFixed(0)}%`}
                      >
                        {data.valueWeight.weightDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value, name) => [`${value} shipments`, name]} />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Processing Time Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Processing Time Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.temporal || errorStates.temporal ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.temporal} 
                    error={errorStates.temporal}
                    onRetry={() => handleRetry('temporal', 'temporal')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.temporal.processingTimeDistribution}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="processingTime" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" name="Shipments" fill="#00C49F" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Seasonal Analysis */}
            <Card>
              <CardHeader>
                <CardTitle>Seasonal Value Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.valueWeight || errorStates.valueWeight ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.valueWeight} 
                    error={errorStates.valueWeight}
                    onRetry={() => handleRetry('valueWeight', 'value-weight')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.valueWeight.seasonalAnalysis}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="season" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="avgValue" name="Avg Value" fill="#FFBB28" />
                      <Bar dataKey="avgWeight" name="Avg Weight" fill="#FF8042" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Geographic Analysis Tab */}
        <TabsContent value="geographic" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Top Routes */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Top Routes by Volume</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.geographic || errorStates.geographic ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.geographic} 
                    error={errorStates.geographic}
                    onRetry={() => handleRetry('geographic', 'geographic')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart 
                      data={data.geographic.topRoutes}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="route" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" name="Shipments" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Origin-Destination Matrix */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Origin-Destination Connections</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.geographic || errorStates.geographic ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.geographic} 
                    error={errorStates.geographic}
                    onRetry={() => handleRetry('geographic', 'geographic')}
                  />
                ) : data.geographic.originDestinationMatrix.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                      <thead>
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Origin</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Destination</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Shipments</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {data.geographic.originDestinationMatrix
                          .sort((a, b) => b.value - a.value)
                          .slice(0, 20)
                          .map((item, index) => (
                            <tr key={index}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm">{item.origin}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm">{item.destination}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm">{item.value}</td>
                            </tr>
                          ))
                        }
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-[300px] text-gray-500">
                    No origin-destination data available
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Value Analysis Tab */}
        <TabsContent value="value" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Value Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Value Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.valueWeight || errorStates.valueWeight ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.valueWeight} 
                    error={errorStates.valueWeight}
                    onRetry={() => handleRetry('valueWeight', 'value-weight')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.valueWeight.valueDistribution}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" name="Shipments" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Weight-Value Relationship */}
            <Card>
              <CardHeader>
                <CardTitle>Weight vs. Value</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.valueWeight || errorStates.valueWeight ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.valueWeight} 
                    error={errorStates.valueWeight}
                    onRetry={() => handleRetry('valueWeight', 'value-weight')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="weight" name="Weight" label={{ value: 'Weight (kg)', position: 'insideBottomRight', offset: -5 }} />
                      <YAxis type="number" dataKey="value" name="Value" label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Weight-Value" data={data.valueWeight.weightValueRelationship} fill="#8884d8" />
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Value per Weight by Airline */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Value per Weight by Airline</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.valueWeight || errorStates.valueWeight ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.valueWeight} 
                    error={errorStates.valueWeight}
                    onRetry={() => handleRetry('valueWeight', 'value-weight')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart 
                      data={data.valueWeight.valuePerWeightByAirline}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="airline" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="valuePerWeight" name="Value per Weight" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Carrier Performance Tab */}
        <TabsContent value="carrier" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Processing Time by Carrier */}
            <Card>
              <CardHeader>
                <CardTitle>Average Processing Time by Carrier</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.carrier || errorStates.carrier ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.carrier} 
                    error={errorStates.carrier}
                    onRetry={() => handleRetry('carrier', 'carrier')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.carrier.carrierMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="carrier" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="avgProcessingTime" name="Processing Time (days)" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Total Value by Carrier */}
            <Card>
              <CardHeader>
                <CardTitle>Total Value by Carrier</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.carrier || errorStates.carrier ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.carrier} 
                    error={errorStates.carrier}
                    onRetry={() => handleRetry('carrier', 'carrier')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.carrier.carrierMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="carrier" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="totalValue" name="Total Value" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Carrier Performance Radar */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Carrier Performance Radar</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.carrier || errorStates.carrier ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.carrier} 
                    error={errorStates.carrier}
                    onRetry={() => handleRetry('carrier', 'carrier')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={400}>
                    <RadarChart outerRadius={150} data={data.carrier.carrierMetrics.slice(0, 5)}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="carrier" />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} />
                      <Radar name="Value Rank" dataKey="valueRank" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                      <Radar name="Efficiency Rank" dataKey="efficiencyRank" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                      <Radar name="Volume Rank" dataKey="volumeRank" stroke="#ffc658" fill="#ffc658" fillOpacity={0.6} />
                      <Radar name="Network Rank" dataKey="networkRank" stroke="#ff8042" fill="#ff8042" fillOpacity={0.6} />
                      <Legend />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Time Analysis Tab */}
        <TabsContent value="time" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Daily Trends */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Daily Shipment Trends</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.temporal || errorStates.temporal ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.temporal} 
                    error={errorStates.temporal}
                    onRetry={() => handleRetry('temporal', 'temporal')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={data.temporal.dailyTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Bar yAxisId="left" dataKey="shipments" barSize={20} fill="#413ea0" name="Shipments" />
                      <Line yAxisId="right" type="monotone" dataKey="avgValue" stroke="#ff7300" name="Avg Value" />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Processing Time Trends */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Processing Time Trends</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.temporal || errorStates.temporal ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.temporal} 
                    error={errorStates.temporal}
                    onRetry={() => handleRetry('temporal', 'temporal')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={data.temporal.dailyTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 'dataMax + 0.5']} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="avgProcessingTime" name="Avg Processing Time (days)" stroke="#8884d8" activeDot={{ r: 8 }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Seasonal Box Plot (visualized as a bar chart) */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Seasonal Value Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.temporal || errorStates.temporal ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.temporal} 
                    error={errorStates.temporal}
                    onRetry={() => handleRetry('temporal', 'temporal')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={data.temporal.seasonalValues}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="season" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="mean" name="Mean Value" fill="#8884d8" />
                      <Bar dataKey="median" name="Median Value" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Clustering Tab */}
        <TabsContent value="clustering" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Shipment Clusters Visualization */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Shipment Clusters</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.clustering || errorStates.clustering ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.clustering} 
                    error={errorStates.clustering}
                    onRetry={() => handleRetry('clustering', 'clustering')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={
                      [0, 1, 2, 3, 4].map(cluster => ({
                        cluster: `Cluster ${cluster}`,
                        avgValue: data.clustering.clusters
                          .filter(item => item.cluster === cluster && !item.isCenter)
                          .reduce((sum, item) => sum + item.fobValue, 0) / 
                          data.clustering.clusters.filter(item => item.cluster === cluster && !item.isCenter).length || 0
                      }))
                    }>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="cluster" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="avgValue" name="Average Value" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Cluster Processing Time Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Processing Time by Cluster</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.clustering || errorStates.clustering ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.clustering} 
                    error={errorStates.clustering}
                    onRetry={() => handleRetry('clustering', 'clustering')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={
                      [0, 1, 2, 3, 4].map(cluster => ({
                        cluster: `Cluster ${cluster}`,
                        avgProcessingTime: data.clustering.clusters
                          .filter(item => item.cluster === cluster && !item.isCenter)
                          .reduce((sum, item) => sum + item.processingTime, 0) / 
                          data.clustering.clusters.filter(item => item.cluster === cluster && !item.isCenter).length || 0
                      }))
                    }>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="cluster" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="avgProcessingTime" name="Avg Processing Time" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Predictive Analytics Tab */}
        <TabsContent value="predictive" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Feature Importance */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Feature Importance for Value Prediction</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.predictive || errorStates.predictive ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.predictive} 
                    error={errorStates.predictive}
                    onRetry={() => handleRetry('predictive', 'predictive')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart
                      data={data.predictive.featureImportance.sort((a, b) => b.importance - a.importance)}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="importance" name="Importance Score" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Model Performance Comparison */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Model Performance Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.predictive || errorStates.predictive ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.predictive} 
                    error={errorStates.predictive}
                    onRetry={() => handleRetry('predictive', 'predictive')}
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.predictive.modelPerformance}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="model" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="r2" name="R² Score" fill="#8884d8" />
                      <Bar dataKey="rmse" name="RMSE (scaled)" fill="#82ca9d" />
                      <Bar dataKey="mae" name="MAE (scaled)" fill="#ffc658" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Correlation Tab */}
        <TabsContent value="correlation" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            {/* Top Feature Correlations */}
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>Top Feature Correlations</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.correlation || errorStates.correlation ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.correlation} 
                    error={errorStates.correlation}
                    onRetry={() => handleRetry('correlation', 'correlation')}
                  />
                ) : (
                  <div className="h-[300px]">
                    {data.correlation.correlationMatrix.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart 
                          data={
                            // Filter correlations between different features and sort by absolute value
                            data.correlation.correlationMatrix
                              .filter(item => item.feature1 !== item.feature2)
                              .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
                              .slice(0, 10)
                              .map(item => ({
                                pair: `${item.feature1} - ${item.feature2}`,
                                correlation: item.correlation
                              }))
                          }
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" domain={[-1, 1]} />
                          <YAxis type="category" dataKey="pair" width={150} />
                          <Tooltip formatter={(value) => value.toFixed(2)} />
                          <Bar 
                            dataKey="correlation" 
                            name="Correlation" 
                            fill={(data) => data.correlation >= 0 ? "#0088FE" : "#FF0000"}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        No correlation data available
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
            
            {/* Correlation Table */}
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>Feature Correlation Matrix</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStates.correlation || errorStates.correlation ? (
                  <ChartPlaceholder 
                    isLoading={loadingStates.correlation} 
                    error={errorStates.correlation}
                    onRetry={() => handleRetry('correlation', 'correlation')}
                  />
                ) : (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                      <thead>
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                            Feature 1
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                            Feature 2
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                            Correlation
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200 dark:divide-gray-700">
                        {data.correlation.correlationMatrix
                          .filter(item => item.feature1 !== item.feature2)
                          .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
                          .slice(0, 20)
                          .map((item, index) => (
                            <tr key={index}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                {item.feature1}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm">
                                {item.feature2}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm" 
                                  style={{ color: item.correlation >= 0 ? 'blue' : 'red' }}>
                                {item.correlation.toFixed(2)}
                              </td>
                            </tr>
                          ))
                        }
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}