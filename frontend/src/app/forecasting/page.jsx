// src/app/forecasting/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { AreaChart, Area, LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function ForecastingPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [forecastData, setForecastData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('ml');

  useEffect(() => {
    // Simulate API call to fetch forecast data
    const fetchData = async () => {
      try {
        // In a real app, this would be an API call: fetch('/api/forecasting/demand')
        // For the prototype, we'll use mock data
        setTimeout(() => {
          // Generate synthetic forecast data
          const today = new Date();
          const forecastData = [];
          
          for (let i = 0; i < 6; i++) {
            const month = new Date(today);
            month.setMonth(today.getMonth() + i);
            const monthName = month.toLocaleString('default', { month: 'short' });
            const year = month.getFullYear();
            
            // Base value with growth trend
            const baseValue = 1800 * (1 + 0.05 * i);
            
            // Add different forecasts based on selected model
            forecastData.push({
              month: `${monthName} ${year}`,
              'ML Forecast': Math.round(baseValue * (1 + 0.1 * Math.sin(i))),
              'ARIMA Forecast': Math.round(baseValue * (1 + 0.15 * Math.cos(i))),
              'Historical Avg': Math.round(baseValue * 0.9),
              'Lower Bound': Math.round(baseValue * 0.8),
              'Upper Bound': Math.round(baseValue * 1.2)
            });
          }
          
          setForecastData(forecastData);
          setIsLoading(false);
        }, 1500);
      } catch (error) {
        console.error('Error fetching forecast data:', error);
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  const modelOptions = [
    { id: 'ml', name: 'Machine Learning Model' },
    { id: 'arima', name: 'ARIMA Model' },
    { id: 'historical', name: 'Historical Average' }
  ];

  // Get the data key based on selected model
  const getDataKey = (model) => {
    switch (model) {
      case 'ml': return 'ML Forecast';
      case 'arima': return 'ARIMA Forecast';
      case 'historical': return 'Historical Avg';
      default: return 'ML Forecast';
    }
  };

  // Skeleton loader
  const ChartSkeleton = () => (
    <div className="w-full h-[400px] bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse" />
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Demand Forecasting</h1>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            <span className="text-sm text-gray-500 dark:text-gray-400">Model:</span>
            <select 
              className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md text-sm p-2"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {modelOptions.map(option => (
                <option key={option.id} value={option.id}>{option.name}</option>
              ))}
            </select>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm">
            Generate Report
          </button>
        </div>
      </div>

      {/* Main Forecast Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Shipment Volume Forecast</CardTitle>
          <CardDescription>
            Predicted shipment volume for the next 6 months using {
              modelOptions.find(opt => opt.id === selectedModel)?.name
            }
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? <ChartSkeleton /> : (
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart
                data={forecastData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey={getDataKey(selectedModel)} 
                  stroke="#3b82f6" 
                  fill="#3b82f6" 
                  fillOpacity={0.3} 
                />
                <Area 
                  type="monotone" 
                  dataKey="Upper Bound" 
                  stroke="#8884d8" 
                  fill="#8884d8" 
                  fillOpacity={0.1} 
                />
                <Area 
                  type="monotone" 
                  dataKey="Lower Bound" 
                  stroke="#82ca9d" 
                  fill="#82ca9d" 
                  fillOpacity={0.1} 
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Forecast Details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Model Performance</CardTitle>
            <CardDescription>
              Accuracy metrics for different forecasting models
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Machine Learning Model</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-600 dark:text-green-500">92.3%</span>
                    <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div className="bg-green-600 dark:bg-green-500 h-full" style={{width: '92.3%'}}></div>
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-medium">ARIMA Model</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-blue-600 dark:text-blue-500">87.8%</span>
                    <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div className="bg-blue-600 dark:bg-blue-500 h-full" style={{width: '87.8%'}}></div>
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-medium">Historical Average</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-amber-600 dark:text-amber-500">76.5%</span>
                    <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div className="bg-amber-600 dark:bg-amber-500 h-full" style={{width: '76.5%'}}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="col-span-1 lg:col-span-2">
          <CardHeader>
            <CardTitle>Forecast Table</CardTitle>
            <CardDescription>
              Monthly forecasts with confidence intervals
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-2">
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b dark:border-gray-700">
                      <th className="py-3 px-4 text-left">Month</th>
                      <th className="py-3 px-4 text-right">Forecast</th>
                      <th className="py-3 px-4 text-right">Lower Bound</th>
                      <th className="py-3 px-4 text-right">Upper Bound</th>
                      <th className="py-3 px-4 text-right">Range</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forecastData.map((item, index) => (
                      <tr 
                        key={index} 
                        className={`hover:bg-gray-50 dark:hover:bg-gray-800 ${
                          index !== forecastData.length - 1 ? 'border-b dark:border-gray-700' : ''
                        }`}
                      >
                        <td className="py-3 px-4 font-medium">{item.month}</td>
                        <td className="py-3 px-4 text-right">{item[getDataKey(selectedModel)].toLocaleString()}</td>
                        <td className="py-3 px-4 text-right">{item['Lower Bound'].toLocaleString()}</td>
                        <td className="py-3 px-4 text-right">{item['Upper Bound'].toLocaleString()}</td>
                        <td className="py-3 px-4 text-right">
                          {(item['Upper Bound'] - item['Lower Bound']).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}