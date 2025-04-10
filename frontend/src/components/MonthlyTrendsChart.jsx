import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { RefreshCw } from 'lucide-react';

const MonthlyTrendsChart = ({ data = [] }) => {
  const loading = !data || data.length === 0;

  return (
    <Card className="h-full bg-gradient-to-br from-gray-900 to-gray-800 border-none text-white shadow-xl">
      <CardHeader className="pb-2 border-b border-gray-700">
        <CardTitle className="text-lg font-medium flex items-center">
          <span className="inline-block w-2 h-6 bg-purple-500 mr-2 rounded-sm"></span>
          Monthly Shipment Trends
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <RefreshCw className="h-8 w-8 text-purple-400 animate-spin" />
          </div>
        ) : (
          <div className="h-full">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data}
                margin={{
                  top: 20,
                  right: 30,
                  left: 5,
                  bottom: 40,
                }}
              >
                <defs>
                  <linearGradient id="shipmentGradient" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.8}/>
                    <stop offset="100%" stopColor="#8B5CF6" stopOpacity={0.8}/>
                  </linearGradient>
                  <linearGradient id="weightGradient" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#10B981" stopOpacity={0.8}/>
                    <stop offset="100%" stopColor="#06B6D4" stopOpacity={0.8}/>
                  </linearGradient>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                    <feMerge>
                      <feMergeNode in="coloredBlur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
                <CartesianGrid 
                  strokeDasharray="3 3" 
                  vertical={false} 
                  horizontal={true} 
                  stroke="rgba(255,255,255,0.1)" 
                />
                <XAxis 
                  dataKey="name" 
                  tick={{ 
                    fill: '#cbd5e1',
                    fontSize: 12,
                  }}
                  tickLine={false}
                  axisLine={{ stroke: 'rgba(255,255,255,0.2)' }}
                  tickMargin={12}
                />
                <YAxis 
                  yAxisId="left"
                  tick={{ 
                    fill: '#cbd5e1',
                    fontSize: 12,
                  }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(value) => value.toLocaleString()}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  tick={{ 
                    fill: '#cbd5e1',
                    fontSize: 12,
                  }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(value) => `${(value/1000).toFixed(1)}k`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(30, 41, 59, 0.9)',
                    borderRadius: '8px', 
                    border: 'none',
                    boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
                    color: '#fff' 
                  }}
                  itemStyle={{ color: '#fff' }}
                  labelStyle={{ fontWeight: 'bold', color: '#cbd5e1' }}
                />
                <Legend 
                  verticalAlign="bottom" 
                  height={36} 
                  wrapperStyle={{ 
                    paddingTop: '20px',
                    color: '#cbd5e1',
                  }}
                  formatter={(value) => <span className="text-gray-300">{value}</span>}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="shipments"
                  stroke="url(#shipmentGradient)"
                  strokeWidth={3}
                  dot={{ r: 6, fill: '#3B82F6', strokeWidth: 1, stroke: '#fff' }}
                  activeDot={{ r: 8, fill: '#3B82F6', filter: 'url(#glow)' }}
                  name="Shipments"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="weight"
                  stroke="url(#weightGradient)"
                  strokeWidth={3}
                  dot={{ r: 6, fill: '#10B981', strokeWidth: 1, stroke: '#fff' }}
                  activeDot={{ r: 8, fill: '#10B981', filter: 'url(#glow)' }}
                  name="Weight (kg)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default MonthlyTrendsChart;