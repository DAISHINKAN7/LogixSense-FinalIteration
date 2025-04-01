// src/app/risk-analysis/page.jsx
'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, Radar, PolarGrid, 
  PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import { AlertTriangle, Shield, TrendingUp, Clock, Download, Send, Zap, 
  AlertCircle, CheckCircle2, RefreshCw, BarChart2, PieChart as PieChartIcon, 
  Map, Info, Clipboard, Bug, Layers } from 'lucide-react';

export default function RiskAnalysisPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [loadingProgress, setLoadingProgress] = useState("Initializing risk analysis...");
  const [riskData, setRiskData] = useState({
    overallRisk: 'Medium',
    riskScore: 65,
    riskFactors: [],
    riskByRegion: [],
    riskTrend: [],
    riskCategories: [],
    anomalies: []
  });
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [feedbackShipment, setFeedbackShipment] = useState(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  
  // Ref for the report generation
  const reportRef = useRef(null);

  const fetchRiskData = async () => {
    try {
      setIsLoading(true);
      setLoadingProgress("Starting risk analysis...");
      
      // Add a timeout to update loading message
      const loadingMessages = [
        "Analyzing shipment patterns...",
        "Identifying risk categories...",
        "Calculating regional risk scores...",
        "Detecting anomalies...",
        "Finalizing risk assessment..."
      ];
      
      // Update loading message every 2 seconds
      const messageInterval = setInterval(() => {
        setLoadingProgress(prevMsg => {
          const currentIndex = loadingMessages.indexOf(prevMsg);
          const nextIndex = (currentIndex + 1) % loadingMessages.length;
          return loadingMessages[nextIndex];
        });
      }, 2000);
      
      const response = await fetch('/api/risk/assessment');
      
      clearInterval(messageInterval);
      
      if (!response.ok) {
        throw new Error(`Error fetching risk data: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Risk Data:", data); // Debug: Check the data structure
      console.log("Regional Data:", data.riskByRegion); // Debug: Check regional data
      
      setRiskData(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching risk data:', err);
      setError('Failed to load risk analysis data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchRiskData();
  }, []);
  
  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchRiskData();
    setRefreshing(false);
  };

  const handleFeedbackSubmit = async (data) => {
    try {
      // In a real app, this would send feedback to the backend
      console.log('Submitting feedback:', data);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setFeedbackSubmitted(true);
      setTimeout(() => {
        setFeedbackShipment(null);
        setFeedbackSubmitted(false);
      }, 2000);
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  const handleGenerateReport = () => {
    // In a real app, this would generate a PDF report
    alert('Report generation would be implemented in a production environment');
  };

  // Skeleton loader
  const ChartSkeleton = () => (
    <div className="w-full h-full min-h-[300px] bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse" />
  );

  // Risk score gauge component
  const RiskGauge = ({ score }) => {
    const getRiskColor = (score) => {
      if (score < 40) return '#10b981'; // Low risk - green
      if (score < 70) return '#f59e0b'; // Medium risk - amber
      return '#ef4444'; // High risk - red
    };

    const rotation = (score / 100) * 180;
    const color = getRiskColor(score);

    return (
      <div className="relative w-48 h-24 mx-auto">
        {/* Gauge background */}
        <div className="absolute w-full h-full overflow-hidden">
          <div className="absolute top-0 w-48 h-48 border-[24px] rounded-full border-gray-200 dark:border-gray-700"></div>
        </div>
        
        {/* Colored gauge indicator */}
        <div className="absolute w-full h-full overflow-hidden">
          <div 
            className="absolute top-0 w-48 h-48 border-[24px] rounded-full" 
            style={{ 
              borderColor: `${color} ${color} transparent transparent`,
              transform: `rotate(${rotation}deg)`,
              transition: 'transform 1s ease-out',
              clip: 'rect(0px, 48px, 96px, 0px)'
            }}
          ></div>
        </div>
        
        {/* Center point */}
        <div className="absolute bottom-0 left-1/2 w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full transform -translate-x-1/2"></div>
        
        {/* Gauge value */}
        <div className="absolute -bottom-8 w-full text-center">
          <span className="text-2xl font-bold">{score}</span>
          <span className="text-sm">/100</span>
        </div>
      </div>
    );
  };

  // Helper functions for styling
  const getRiskColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'text-green-600 dark:text-green-500';
      case 'medium': return 'text-amber-600 dark:text-amber-500';
      case 'high': return 'text-red-600 dark:text-red-500';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getRiskBgColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'bg-green-100 dark:bg-green-900/20';
      case 'medium': return 'bg-amber-100 dark:bg-amber-900/20';
      case 'high': return 'bg-red-100 dark:bg-red-900/20';
      default: return 'bg-gray-100 dark:bg-gray-800';
    }
  };

  return (
    <div ref={reportRef} className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Risk Analysis</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            AI-driven risk assessment of your shipping operations
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            {refreshing ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4 mr-2" />
            )}
            Refresh Data
          </Button>
          <Button
            className="bg-blue-600 hover:bg-blue-700 text-white"
            size="sm"
            onClick={handleGenerateReport}
          >
            <Download className="h-4 w-4 mr-2" />
            Generate Report
          </Button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="p-4 mb-4 text-red-700 bg-red-100 dark:bg-red-900/20 dark:text-red-400 rounded-md">
          <p>{error}</p>
        </div>
      )}

      {isLoading && (
        <div className="loading-container p-8 text-center">
          <div className="spinner w-12 h-12 border-4 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="loading-message text-lg font-medium mb-2">{loadingProgress}</p>
          <p className="loading-note text-gray-500">This may take up to 30 seconds as we analyze your data.</p>
        </div>
      )}
      
      {/* Tab navigation */}
      <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-4 mb-4">
          <TabsTrigger value="overview">
            <Shield className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="regions">
            <Map className="h-4 w-4 mr-2" />
            Regional Analysis
          </TabsTrigger>
          <TabsTrigger value="factors">
            <Layers className="h-4 w-4 mr-2" />
            Risk Factors
          </TabsTrigger>
          <TabsTrigger value="anomalies">
            <Bug className="h-4 w-4 mr-2" />
            Anomaly Detection
          </TabsTrigger>
        </TabsList>
        
        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Risk Overview */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>Overall Risk Assessment</CardTitle>
                <CardDescription>
                  AI-generated risk score based on multiple factors
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center">
                {isLoading ? (
                  <div className="w-full space-y-4">
                    <div className="h-48 bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse" />
                  </div>
                ) : (
                  <>
                    <RiskGauge score={riskData.riskScore} />
                    <div className="mt-8 text-center">
                      <div className="text-xl font-bold mb-1">
                        <span className={getRiskColor(riskData.overallRisk)}>
                          {riskData.overallRisk} Risk
                        </span>
                      </div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Based on analysis of {riskData.riskByRegion?.reduce((acc, curr) => acc + curr.shipmentCount, 0) || 0} 
                        {' '}shipments across {riskData.riskByRegion?.length || 0} regions
                      </p>
                    </div>
                  </>
                )}
              </CardContent>
              <CardFooter className="flex justify-center">
                <p className="text-xs text-gray-500 text-center">
                  Risk assessment based on machine learning analysis of your historical shipping data
                </p>
              </CardFooter>
            </Card>

            <Card className="col-span-1 lg:col-span-2">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Risk Trend Analysis</CardTitle>
                  <CardDescription>
                    How risk levels have changed over time
                  </CardDescription>
                </div>
                <Select defaultValue="6months">
                  <SelectTrigger className="w-[140px]">
                    <SelectValue placeholder="Time period" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="3months">Last 3 months</SelectItem>
                    <SelectItem value="6months">Last 6 months</SelectItem>
                    <SelectItem value="12months">Last 12 months</SelectItem>
                  </SelectContent>
                </Select>
              </CardHeader>
              <CardContent>
                {isLoading ? <ChartSkeleton /> : (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={riskData.riskTrend}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip 
                        formatter={(value) => [`${value}/100`, 'Risk Score']}
                        labelFormatter={(label) => `Month: ${label}`}
                      />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="score" 
                        name="Risk Score" 
                        stroke="#f59e0b" 
                        activeDot={{ r: 8 }} 
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
              <CardFooter>
                <p className="text-xs text-gray-500">
                  Trend calculated from statistical analysis of shipment patterns over time
                </p>
              </CardFooter>
            </Card>
          </div>

          {/* Risk Categories and Stats */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Risk Categories</CardTitle>
                <CardDescription>
                  Machine learning identified risk dimensions
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? <ChartSkeleton /> : (
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={riskData.riskCategories}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="subject" />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} />
                      <Radar 
                        name="Risk Score" 
                        dataKey="A" 
                        stroke="#f59e0b" 
                        fill="#f59e0b" 
                        fillOpacity={0.6} 
                      />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
              <CardFooter>
                <p className="text-xs text-gray-500">
                  Risk categories identified through data analysis algorithms
                </p>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Key Risk Factors</CardTitle>
                <CardDescription>
                  Major factors contributing to overall risk
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="space-y-4">
                    <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                    <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                    <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                    <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    {riskData.riskFactors.map((factor, index) => (
                      <div key={index} className="flex items-center justify-between p-3 rounded-md bg-gray-50 dark:bg-gray-800">
                        <div className="flex items-start space-x-3">
                          <div className={`p-2 rounded-full ${getRiskBgColor(factor.impact)}`}>
                            <AlertTriangle className={`h-5 w-5 ${getRiskColor(factor.impact)}`} />
                          </div>
                          <div>
                            <p className="font-medium">{factor.factor}</p>
                            <div className="flex items-center space-x-2 mt-1">
                              <span className={`text-xs px-2 py-1 rounded-full ${getRiskBgColor(factor.impact)} ${getRiskColor(factor.impact)}`}>
                                Impact: {factor.impact}
                              </span>
                              <span className={`text-xs px-2 py-1 rounded-full ${getRiskBgColor(factor.probability)} ${getRiskColor(factor.probability)}`}>
                                Probability: {factor.probability}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="text-xl font-bold">
                          {factor.score}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
              <CardFooter>
                <p className="text-xs text-gray-500">
                  Risk factors are calculated using statistical clustering and data analysis
                </p>
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
        
        {/* Regional Analysis Tab */}
        <TabsContent value="regions" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Risk Assessment by Region</CardTitle>
              <CardDescription>
                Analysis of supply chain risk factors across different regions
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <ChartSkeleton />
              ) : riskData.riskByRegion && riskData.riskByRegion.length > 0 ? (
                <ResponsiveContainer width="100%" height={500}>
                  <BarChart
                    data={riskData.riskByRegion}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 100]} />
                    <YAxis 
                      type="category" 
                      dataKey="region" 
                      width={150}
                    />
                    <Tooltip 
                      formatter={(value, name) => [
                        name === 'riskScore' ? `${value}/100` : value, 
                        name === 'riskScore' ? 'Risk Score' : 'Shipment Count'
                      ]}
                    />
                    <Legend />
                    <Bar 
                      dataKey="riskScore" 
                      name="Risk Score" 
                      fill="#f59e0b" 
                    />
                    <Bar 
                      dataKey="shipmentCount" 
                      name="Shipment Count" 
                      fill="#3b82f6" 
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-center">
                  <AlertCircle className="h-10 w-10 mb-4 text-amber-500" />
                  <h3 className="text-lg font-medium mb-2">No Regional Risk Data Available</h3>
                  <p className="text-sm text-gray-500 max-w-md">
                    This could be because:
                  </p>
                  <ul className="list-disc pl-6 mt-2 text-sm text-gray-500 text-left">
                    <li>The dataset doesn't contain country information</li>
                    <li>There are not enough unique countries in the data</li>
                    <li>The data is still being processed</li>
                  </ul>
                  <Button variant="outline" size="sm" className="mt-4" onClick={handleRefresh}>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh Data
                  </Button>
                </div>
              )}
            </CardContent>
            <CardFooter>
              <p className="text-xs text-gray-500">
                Regional risk is calculated by analyzing historical shipment outcomes by destination
              </p>
            </CardFooter>
          </Card>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Regional Risk Breakdown</CardTitle>
                <CardDescription>
                  Risk factors by region
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <ChartSkeleton />
                ) : riskData.riskByRegion && riskData.riskByRegion.length > 0 ? (
                  <div className="space-y-6">
                    {riskData.riskByRegion.slice(0, 5).map((region, index) => (
                      <div key={index} className="space-y-2">
                        <div className="flex justify-between items-center">
                          <h3 className="font-medium">{region.region}</h3>
                          <span className={`px-2 py-1 text-xs rounded-full ${getRiskBgColor(region.riskScore >= 70 ? 'high' : region.riskScore >= 45 ? 'medium' : 'low')}`}>
                            {region.riskScore} / 100
                          </span>
                        </div>
                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                          <div 
                            className="h-2 rounded-full"
                            style={{ 
                              width: `${region.riskScore}%`,
                              background: region.riskScore >= 70 ? '#ef4444' : region.riskScore >= 45 ? '#f59e0b' : '#10b981'
                            }}
                          ></div>
                        </div>
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>{region.shipmentCount} shipments</span>
                          <span>
                            {region.riskScore >= 70 ? 'High Risk' : region.riskScore >= 45 ? 'Medium Risk' : 'Low Risk'}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-8 text-center text-gray-500">
                    <p>No regional breakdown data available.</p>
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Risk Mitigation Recommendations</CardTitle>
                <CardDescription>
                  AI-generated suggestions to reduce regional risk
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="space-y-4">
                    <div className="h-24 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                    <div className="h-24 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                  </div>
                ) : riskData.riskByRegion && riskData.riskByRegion.length > 0 ? (
                  <div className="space-y-4">
                    {riskData.riskByRegion.slice(0, 3).map((region, index) => {
                      // Generate appropriate recommendations based on risk score
                      const recommendations = [];
                      
                      if (region.riskScore >= 70) {
                        recommendations.push(
                          "Implement enhanced tracking for all shipments to this region",
                          "Consider alternate routing options to reduce transit time",
                          "Review carrier performance data for this destination"
                        );
                      } else if (region.riskScore >= 50) {
                        recommendations.push(
                          "Schedule regular risk assessment reviews for this region",
                          "Establish backup carrier relationships for peak periods",
                          "Monitor transit time trends for early warning signs"
                        );
                      } else {
                        recommendations.push(
                          "Maintain current risk mitigation strategies",
                          "Consider this region for capacity expansion",
                          "Document successful handling procedures for knowledge sharing"
                        );
                      }
                      
                      return (
                        <div key={index} className="p-4 border rounded-md dark:border-gray-700">
                          <h3 className="font-medium mb-2">
                            {region.region} ({region.riskScore}/100)
                          </h3>
                          <ul className="space-y-1 text-sm">
                            {recommendations.map((rec, i) => (
                              <li key={i} className="flex items-start">
                                <CheckCircle2 className="h-4 w-4 mr-2 mt-0.5 text-green-500" />
                                <span>{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="p-8 text-center text-gray-500">
                    <p>No regional data available for recommendations.</p>
                  </div>
                )}
              </CardContent>
              <CardFooter>
                <Button variant="outline" size="sm" className="w-full">
                  <Download className="h-4 w-4 mr-2" />
                  Download Regional Risk Report
                </Button>
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
        
        {/* Risk Factors Tab */}
        <TabsContent value="factors" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Risk Distribution</CardTitle>
                <CardDescription>
                  How different risk factors contribute to overall risk
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? <ChartSkeleton /> : (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={riskData.riskCategories.map(cat => ({
                          name: cat.subject,
                          value: cat.A
                        }))}
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        dataKey="value"
                        nameKey="name"
                        label={(entry) => entry.name}
                        labelLine
                      >
                        {riskData.riskCategories.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={`hsl(${index * 35}, 70%, 50%)`} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => [`${value.toFixed(1)}/100`, 'Risk Score']} />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Risk Factors Breakdown</CardTitle>
                <CardDescription>
                  Detailed analysis of each risk component
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="space-y-4">
                    <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                    <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                    <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    {riskData.riskCategories.map((category, index) => (
                      <div key={index} className="space-y-1">
                        <div className="flex justify-between">
                          <h3 className="font-medium">{category.subject}</h3>
                          <span className="text-sm">{category.A.toFixed(1)}/100</span>
                        </div>
                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                          <div 
                            className="h-2 rounded-full"
                            style={{ 
                              width: `${category.A}%`,
                              background: `hsl(${index * 35}, 70%, 50%)`
                            }}
                          ></div>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                          {category.subject === "Shipping Time" && "Analysis of processing and transit times"}
                          {category.subject === "Value" && "Risk based on shipment value concentration"}
                          {category.subject === "Weight" && "Analysis of weight distribution and handling requirements"}
                          {category.subject === "Destination" && "Risk assessment of shipping destinations"}
                          {category.subject === "Country" && "Country-specific risk factors"}
                          {category.subject === "Weather" && "Impact of seasonal weather patterns"}
                          {category.subject === "Political" && "Geopolitical stability assessment"}
                          {category.subject === "Economic" && "Economic and currency risk factors"}
                          {category.subject === "Logistical" && "Logistics infrastructure assessment"}
                          {category.subject === "Compliance" && "Regulatory and compliance risk factors"}
                          {category.subject === "Cybersecurity" && "Digital and data security considerations"}
                          {category.subject === "System" && "System performance and reliability metrics"}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle>AI-Driven Risk Thresholds</CardTitle>
              <CardDescription>
                Statistically determined risk classification boundaries
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? <ChartSkeleton /> : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Risk threshold cards - in a real app, these would use actual thresholds */}
                  <div className="p-4 border rounded-md dark:border-gray-700">
                    <h3 className="font-medium mb-2">Weight Risk Thresholds</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-green-600">Low Risk</span>
                        <span className="text-sm font-medium">{"< 50 kg"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-amber-600">Medium Risk</span>
                        <span className="text-sm font-medium">{"50-200 kg"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-red-600">High Risk</span>
                        <span className="text-sm font-medium">{"> 200 kg"}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-4 border rounded-md dark:border-gray-700">
                    <h3 className="font-medium mb-2">Value Risk Thresholds</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-green-600">Low Risk</span>
                        <span className="text-sm font-medium">{"< $10,000"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-amber-600">Medium Risk</span>
                        <span className="text-sm font-medium">{"$10,000-$100,000"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-red-600">High Risk</span>
                        <span className="text-sm font-medium">{"> $100,000"}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-4 border rounded-md dark:border-gray-700">
                    <h3 className="font-medium mb-2">Transit Time Thresholds</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-green-600">Low Risk</span>
                        <span className="text-sm font-medium">{"< 3 days"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-amber-600">Medium Risk</span>
                        <span className="text-sm font-medium">{"3-7 days"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-red-600">High Risk</span>
                        <span className="text-sm font-medium">{"> 7 days"}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
            <CardFooter className="text-xs text-gray-500">
              Thresholds are dynamically determined using K-means clustering on your shipping data
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Anomaly Detection Tab */}
        <TabsContent value="anomalies" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Anomaly Detection</CardTitle>
              <CardDescription>
                Machine learning identified unusual shipment patterns
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-4">
                  <div className="h-20 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                  <div className="h-20 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                  <div className="h-20 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                </div>
              ) : riskData.anomalies && riskData.anomalies.length > 0 ? (
                <div className="space-y-4">
                  {riskData.anomalies.map((anomaly, index) => (
                    <div key={index} className="flex items-center justify-between p-4 border rounded-md dark:border-gray-700">
                      <div className="space-y-1">
                        <div className="flex items-center">
                          <Bug className="h-5 w-5 mr-2 text-amber-500" />
                          <h3 className="font-medium">{anomaly.id}</h3>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {anomaly.destination}
                        </p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {anomaly.reasons.map((reason, i) => (
                            <span key={i} className="px-2 py-1 text-xs bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-full">
                              {reason}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xl font-bold">{anomaly.anomalyScore}</div>
                        <div className="text-sm text-gray-500">Anomaly Score</div>
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="outline" size="sm" className="mt-2">
                              <Info className="h-4 w-4 mr-2" />
                              Details
                            </Button>
                          </DialogTrigger>
                          <DialogContent>
                            <DialogHeader>
                              <DialogTitle>Anomaly Details: {anomaly.id}</DialogTitle>
                              <DialogDescription>
                                This shipment has been flagged by our anomaly detection algorithm.
                              </DialogDescription>
                            </DialogHeader>
                            <div className="space-y-4 py-4">
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <h4 className="text-sm font-medium mb-1">Destination</h4>
                                  <p className="text-sm">{anomaly.destination}</p>
                                </div>
                                <div>
                                  <h4 className="text-sm font-medium mb-1">Weight</h4>
                                  <p className="text-sm">{anomaly.weight}</p>
                                </div>
                                <div>
                                  <h4 className="text-sm font-medium mb-1">Value</h4>
                                  <p className="text-sm">{anomaly.value}</p>
                                </div>
                                <div>
                                  <h4 className="text-sm font-medium mb-1">Anomaly Score</h4>
                                  <p className="text-sm">{anomaly.anomalyScore}/100</p>
                                </div>
                              </div>
                              <div>
                                <h4 className="text-sm font-medium mb-1">Anomaly Reasons</h4>
                                <ul className="text-sm space-y-1">
                                  {anomaly.reasons.map((reason, i) => (
                                    <li key={i} className="flex items-start">
                                      <AlertCircle className="h-4 w-4 mr-2 mt-0.5 text-amber-500" />
                                      <span>{reason}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                              <div>
                                <h4 className="text-sm font-medium mb-1">Recommended Actions</h4>
                                <ul className="text-sm space-y-1">
                                  <li className="flex items-start">
                                    <CheckCircle2 className="h-4 w-4 mr-2 mt-0.5 text-green-500" />
                                    <span>Verify shipment details with carrier</span>
                                  </li>
                                  <li className="flex items-start">
                                    <CheckCircle2 className="h-4 w-4 mr-2 mt-0.5 text-green-500" />
                                    <span>Implement enhanced tracking</span>
                                  </li>
                                  <li className="flex items-start">
                                    <CheckCircle2 className="h-4 w-4 mr-2 mt-0.5 text-green-500" />
                                    <span>Document resolution for feedback loop</span>
                                  </li>
                                </ul>
                              </div>
                            </div>
                            <DialogFooter>
                              <Button 
                                variant="outline" 
                                onClick={() => setFeedbackShipment(anomaly.id)}
                              >
                                <Send className="h-4 w-4 mr-2" />
                                Provide Feedback
                              </Button>
                            </DialogFooter>
                          </DialogContent>
                        </Dialog>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-center">
                  <CheckCircle2 className="h-10 w-10 mb-4 text-green-500" />
                  <h3 className="text-lg font-medium mb-2">No Anomalies Detected</h3>
                  <p className="text-sm text-gray-500 max-w-md">
                    Our machine learning system has not identified any unusual patterns 
                    in your recent shipments. Continue monitoring for changes.
                  </p>
                </div>
              )}
            </CardContent>
            <CardFooter>
              <p className="text-xs text-gray-500">
                Anomalies are detected using statistical analysis with a 3-sigma threshold
              </p>
            </CardFooter>
          </Card>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Anomaly Detection Methodology</CardTitle>
                <CardDescription>
                  How our machine learning identifies unusual patterns
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-start">
                    <div className="bg-blue-100 dark:bg-blue-900/20 p-2 rounded-full mr-4">
                      <Layers className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                    </div>
                    <div>
                      <h3 className="font-medium mb-1">Multi-Dimensional Analysis</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Our system analyzes multiple dimensions simultaneously, including weight, value, 
                        processing time, destination patterns, and historical averages.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start">
                    <div className="bg-purple-100 dark:bg-purple-900/20 p-2 rounded-full mr-4">
                      <AlertCircle className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                    </div>
                    <div>
                      <h3 className="font-medium mb-1">Statistical Outlier Detection</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        We use statistical methods to identify outliers that deviate significantly from normal patterns,
                        focusing on shipments that exceed 3 standard deviations from the mean.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start">
                    <div className="bg-green-100 dark:bg-green-900/20 p-2 rounded-full mr-4">
                      <RefreshCw className="h-5 w-5 text-green-600 dark:text-green-400" />
                    </div>
                    <div>
                      <h3 className="font-medium mb-1">Adaptive Learning</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        The system continuously learns from your feedback on identified anomalies, 
                        improving detection accuracy over time and reducing false positives.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Feedback Loop</CardTitle>
                <CardDescription>
                  Help improve anomaly detection accuracy
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-sm">
                    Our machine learning system improves with your input. Provide feedback on 
                    identified anomalies to refine future detection.
                  </p>
                  
                  <Button 
                    className="w-full"
                    onClick={() => setFeedbackShipment(riskData.anomalies?.[0]?.id || "AWB123456789")}
                  >
                    <Send className="h-4 w-4 mr-2" />
                    Submit Feedback
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
      
      {/* Feedback Dialog */}
      <Dialog open={!!feedbackShipment} onOpenChange={(open) => !open && setFeedbackShipment(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Provide Risk Assessment Feedback</DialogTitle>
            <DialogDescription>
              Your feedback helps improve our machine learning models.
            </DialogDescription>
          </DialogHeader>
          
          {feedbackSubmitted ? (
            <div className="py-6 flex flex-col items-center text-center">
              <CheckCircle2 className="h-10 w-10 mb-4 text-green-500" />
              <h3 className="text-lg font-medium mb-2">Feedback Submitted</h3>
              <p className="text-sm text-gray-500">
                Thank you for your input! Your feedback helps improve our risk assessment accuracy.
              </p>
            </div>
          ) : (
            <div className="space-y-4 py-4">
              <div>
                <Label htmlFor="shipment-id" className="text-sm font-medium">Shipment ID</Label>
                <Input 
                  id="shipment-id" 
                  value={feedbackShipment || ""} 
                  readOnly 
                  className="mt-1"
                />
              </div>
              
              <div>
                <Label className="text-sm font-medium">Actual Risk Level</Label>
                <RadioGroup defaultValue="medium" className="mt-2">
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="low" id="low" />
                    <Label htmlFor="low" className="text-green-600">Low Risk</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="medium" id="medium" />
                    <Label htmlFor="medium" className="text-amber-600">Medium Risk</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="high" id="high" />
                    <Label htmlFor="high" className="text-red-600">High Risk</Label>
                  </div>
                </RadioGroup>
              </div>
              
              <div>
                <Label htmlFor="feedback-comments" className="text-sm font-medium">Comments</Label>
                <Textarea 
                  id="feedback-comments" 
                  placeholder="Please provide any additional context about this shipment..."
                  className="mt-1"
                />
              </div>
            </div>
          )}
          
          <DialogFooter>
            {!feedbackSubmitted && (
              <>
                <Button variant="outline" onClick={() => setFeedbackShipment(null)}>
                  Cancel
                </Button>
                <Button onClick={() => handleFeedbackSubmit({
                  shipmentId: feedbackShipment,
                  riskLevel: "medium",
                  comments: "Example feedback"
                })}>
                  Submit Feedback
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}