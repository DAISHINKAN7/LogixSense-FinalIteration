import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { RefreshCw } from 'lucide-react';

const WeightDistributionChart = ({ data = [] }) => {
  // Ensure we have valid data to display
  const chartData = Array.isArray(data) ? data.filter(item => 
    item && 
    typeof item === 'object' && 
    item.name && 
    (typeof item.value === 'number' || !isNaN(Number(item.value)))
  ) : [];
  
  // Convert string values to numbers if needed
  const processedData = chartData.map(item => ({
    ...item,
    value: typeof item.value === 'number' ? item.value : Number(item.value)
  }));
  
  const loading = processedData.length === 0;

  const formatTooltip = (value) => {
    if (typeof value === 'number') {
      return `${value.toFixed(1)}%`;
    }
    return `${value}%`;
  };

  return (
    <Card className="h-full bg-gradient-to-br from-gray-900 to-gray-800 border-none text-white shadow-xl">
      <CardHeader className="pb-2 border-b border-gray-700">
        <CardTitle className="text-lg font-medium flex items-center">
          <span className="inline-block w-2 h-6 bg-blue-500 mr-2 rounded-sm"></span>
          Shipment Weight Distribution
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <RefreshCw className="h-8 w-8 text-blue-400 animate-spin" />
          </div>
        ) : (
          <div className="h-full flex flex-col">
            {/* Chart area */}
            <div className="flex-1">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={processedData}
                  margin={{
                    top: 20,
                    right: 20,
                    left: 0,
                    bottom: 40,
                  }}
                >
                  <defs>
                    <linearGradient id="weightGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.9}/>
                      <stop offset="100%" stopColor="#10B981" stopOpacity={0.7}/>
                    </linearGradient>
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
                    tickFormatter={(value) => `${value}%`}
                    tick={{ 
                      fill: '#cbd5e1',
                      fontSize: 12,
                    }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, 'dataMax + 10']}
                  />
                  <Tooltip 
                    formatter={formatTooltip}
                    labelStyle={{ fontWeight: 'bold', color: '#cbd5e1' }}
                    contentStyle={{ 
                      backgroundColor: 'rgba(30, 41, 59, 0.9)',
                      borderRadius: '8px', 
                      border: 'none',
                      boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
                      color: '#fff' 
                    }}
                  />
                  <Bar 
                    dataKey="value" 
                    fill="url(#weightGradient)"
                    radius={[4, 4, 0, 0]}
                    barSize={50}
                    animationDuration={1500}
                    name="Percentage"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Bottom legend with weight ranges and percentages */}
            <div className="grid grid-cols-4 gap-4 mt-4">
              {processedData.map((item, index) => (
                <div key={`weight-category-${index}`} className="flex flex-col items-center">
                  <div className="text-lg font-bold">
                    {typeof item.value === 'number' ? item.value.toFixed(1) : item.value}%
                  </div>
                  <div className="text-sm text-gray-400">
                    {item.name}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default WeightDistributionChart;