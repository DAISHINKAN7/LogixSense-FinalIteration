// src/app/analytics/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function AnalyticsPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [data, setData] = useState({
    volumeByDestination: [],
    shipmentsByMonth: [],
    weightDistribution: [],
    commodityBreakdown: []
  });

  useEffect(() => {
    // Simulate API call to fetch data
    const fetchData = async () => {
      try {
        // In a real app, this would be an API call
        // For the prototype, we'll use mock data
        setTimeout(() => {
          setData({
            volumeByDestination: [
              { name: 'USA', value: 4200 },
              { name: 'UAE', value: 3100 },
              { name: 'Singapore', value: 2800 },
              { name: 'UK', value: 2400 },
              { name: 'Japan', value: 1900 },
              { name: 'Others', value: 5600 }
            ],
            shipmentsByMonth: [
              { name: 'Jan', shipments: 1200, weight: 42000 },
              { name: 'Feb', shipments: 1350, weight: 45000 },
              { name: 'Mar', shipments: 1500, weight: 48000 },
              { name: 'Apr', shipments: 1600, weight: 52000 },
              { name: 'May', shipments: 1450, weight: 49000 },
              { name: 'Jun', shipments: 1800, weight: 56000 },
            ],
            weightDistribution: [
              { name: '0-50 kg', value: 35 },
              { name: '51-200 kg', value: 40 },
              { name: '201-500 kg', value: 15 },
              { name: '501+ kg', value: 10 }
            ],
            commodityBreakdown: [
              { name: 'Electronics', value: 28 },
              { name: 'Textiles', value: 22 },
              { name: 'Pharmaceuticals', value: 15 },
              { name: 'Machinery', value: 18 },
              { name: 'Food Products', value: 12 },
              { name: 'Others', value: 5 }
            ]
          });
          setIsLoading(false);
        }, 1500);
      } catch (error) {
        console.error('Error fetching analytics data:', error);
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  // Skeleton loader for charts
  const ChartSkeleton = () => (
    <div className="w-full h-full min-h-[300px] bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse" />
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Analytics</h1>
        <div className="flex items-center space-x-2">
          <select className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md text-sm p-2">
            <option value="last30days">Last 30 days</option>
            <option value="last3months">Last 3 months</option>
            <option value="last6months">Last 6 months</option>
            <option value="lastyear">Last year</option>
            <option value="custom">Custom range</option>
          </select>
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm">
            Export
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Shipment Volume by Destination */}
        <Card className="col-span-1">
          <CardHeader className="pb-2">
            <CardTitle>Shipment Volume by Destination</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <ChartSkeleton /> : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={data.volumeByDestination}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" name="Shipments" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Shipments and Weight by Month */}
        <Card className="col-span-1">
          <CardHeader className="pb-2">
            <CardTitle>Shipments and Weight by Month</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <ChartSkeleton /> : (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={data.shipmentsByMonth}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="shipments" 
                    name="Shipments"
                    stroke="#3b82f6" 
                    activeDot={{ r: 8 }} 
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="weight" 
                    name="Weight (kg)" 
                    stroke="#10b981" 
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Shipment Weight Distribution */}
        <Card className="col-span-1">
          <CardHeader className="pb-2">
            <CardTitle>Shipment Weight Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <ChartSkeleton /> : (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={data.weightDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {data.weightDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Commodity Breakdown */}
        <Card className="col-span-1">
          <CardHeader className="pb-2">
            <CardTitle>Commodity Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <ChartSkeleton /> : (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={data.commodityBreakdown}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {data.commodityBreakdown.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}