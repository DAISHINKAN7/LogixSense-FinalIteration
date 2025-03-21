// src/app/risk-analysis/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, Radar, PolarGrid, 
  PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import { AlertTriangle, Shield, TrendingUp, Clock } from 'lucide-react';

export default function RiskAnalysisPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [riskData, setRiskData] = useState({
    overallRisk: 'Medium',
    riskScore: 65,
    riskFactors: [],
    riskByRegion: []
  });

  useEffect(() => {
    // Simulate API call to fetch risk data
    const fetchData = async () => {
      try {
        // In a real app, this would be an API call
        // For the prototype, we'll use mock data
        setTimeout(() => {
          setRiskData({
            overallRisk: 'Medium',
            riskScore: 65,
            riskFactors: [
              { factor: 'Weather Disruptions', impact: 'High', probability: 'Medium', score: 75 },
              { factor: 'Customs Delays', impact: 'Medium', probability: 'High', score: 70 },
              { factor: 'Transportation Strikes', impact: 'High', probability: 'Low', score: 55 },
              { factor: 'Capacity Constraints', impact: 'Medium', probability: 'Medium', score: 60 },
              { factor: 'Political Instability', impact: 'High', probability: 'Low', score: 50 },
              { factor: 'Currency Fluctuations', impact: 'Low', probability: 'Medium', score: 40 }
            ],
            riskByRegion: [
              { region: 'Dubai, UAE', riskScore: 78, shipmentCount: 245 },
              { region: 'Singapore', riskScore: 42, shipmentCount: 183 },
              { region: 'Tokyo, Japan', riskScore: 38, shipmentCount: 127 },
              { region: 'New York, USA', riskScore: 56, shipmentCount: 312 },
              { region: 'London, UK', riskScore: 52, shipmentCount: 205 },
              { region: 'Shanghai, China', riskScore: 67, shipmentCount: 178 },
              { region: 'Hong Kong', riskScore: 61, shipmentCount: 143 },
              { region: 'Sydney, Australia', riskScore: 39, shipmentCount: 97 }
            ],
            riskTrend: [
              { month: 'Sep', score: 72 },
              { month: 'Oct', score: 69 },
              { month: 'Nov', score: 75 },
              { month: 'Dec', score: 71 },
              { month: 'Jan', score: 68 },
              { month: 'Feb', score: 65 }
            ],
            riskCategories: [
              { subject: 'Weather', A: 80, fullMark: 100 },
              { subject: 'Political', A: 50, fullMark: 100 },
              { subject: 'Economic', A: 65, fullMark: 100 },
              { subject: 'Logistical', A: 70, fullMark: 100 },
              { subject: 'Cybersecurity', A: 55, fullMark: 100 },
              { subject: 'Compliance', A: 60, fullMark: 100 }
            ]
          });
          setIsLoading(false);
        }, 1500);
      } catch (error) {
        console.error('Error fetching risk data:', error);
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  // Skeleton loader
  const ChartSkeleton = () => (
    <div className="w-full h-full min-h-[300px] bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse" />
  );

  // Risk score gauge
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

  const getRiskColor = (level) => {
    switch (level.toLowerCase()) {
      case 'low': return 'text-green-600 dark:text-green-500';
      case 'medium': return 'text-amber-600 dark:text-amber-500';
      case 'high': return 'text-red-600 dark:text-red-500';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getRiskBgColor = (level) => {
    switch (level.toLowerCase()) {
      case 'low': return 'bg-green-100 dark:bg-green-900/20';
      case 'medium': return 'bg-amber-100 dark:bg-amber-900/20';
      case 'high': return 'bg-red-100 dark:bg-red-900/20';
      default: return 'bg-gray-100 dark:bg-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Risk Analysis</h1>
        <div className="flex items-center space-x-2">
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm">
            Generate Risk Report
          </button>
        </div>
      </div>

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
                    Based on analysis of {riskData.riskByRegion.reduce((acc, curr) => acc + curr.shipmentCount, 0)} 
                    {' '}shipments across {riskData.riskByRegion.length} regions
                  </p>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="col-span-1 lg:col-span-2">
          <CardHeader>
            <CardTitle>Risk Trend Analysis</CardTitle>
            <CardDescription>
              How risk levels have changed over time
            </CardDescription>
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
        </Card>
      </div>

      {/* Risk by Region */}
      <Card>
        <CardHeader>
          <CardTitle>Risk Assessment by Region</CardTitle>
          <CardDescription>
            Analysis of supply chain risk factors across different regions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? <ChartSkeleton /> : (
            <ResponsiveContainer width="100%" height={400}>
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
          )}
        </CardContent>
      </Card>

      {/* Risk Categories and Factors */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Risk Categories</CardTitle>
            <CardDescription>
              Risk assessment across different categories
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
        </Card>
      </div>
    </div>
  );
}