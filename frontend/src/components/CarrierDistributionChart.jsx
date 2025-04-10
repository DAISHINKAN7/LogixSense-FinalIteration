import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { RefreshCw } from 'lucide-react';

// Modern color palette
const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#14B8A6', '#F97316'];

const CarrierDistributionChart = ({ data, loading: externalLoading }) => {
  const [carrierData, setCarrierData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchCarrierData = async () => {
      if (data) {
        setCarrierData(data);
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const response = await fetch('/api/dashboard/carriers');
        if (!response.ok) {
          throw new Error(`Error fetching carrier data: ${response.status}`);
        }
        const data = await response.json();
        setCarrierData(data);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch carrier data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchCarrierData();
  }, [data]);

  const isLoading = loading || externalLoading || !carrierData || carrierData.length === 0;

  const handleMouseEnter = (_, index) => {
    setActiveIndex(index);
  };

  const handleMouseLeave = () => {
    setActiveIndex(null);
  };

  return (
    <Card className="h-full bg-gradient-to-br from-gray-900 to-gray-800 border-none text-white shadow-xl">
      <CardHeader className="pb-2 border-b border-gray-700">
        <CardTitle className="text-lg font-medium flex items-center">
          <span className="inline-block w-2 h-6 bg-cyan-500 mr-2 rounded-sm"></span>
          Shipments by Carrier
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <RefreshCw className="h-8 w-8 text-cyan-400 animate-spin" />
          </div>
        ) : error ? (
          <div className="flex justify-center items-center h-64">
            <p className="text-red-400">Error loading carrier data</p>
          </div>
        ) : (
          <div className="h-full">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                layout="vertical"
                data={carrierData}
                margin={{
                  top: 20,
                  right: 40,
                  left: 40,
                  bottom: 20,
                }}
                onMouseLeave={handleMouseLeave}
              >
                <defs>
                  {COLORS.map((color, index) => (
                    <linearGradient key={`gradient-${index}`} id={`barGradient-${index}`} x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor={color} stopOpacity={0.9}/>
                      <stop offset="100%" stopColor={color} stopOpacity={0.7}/>
                    </linearGradient>
                  ))}
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
                  horizontal={false} 
                  vertical={true} 
                  stroke="rgba(255,255,255,0.1)" 
                />
                <XAxis 
                  type="number" 
                  tick={{ 
                    fill: '#cbd5e1',
                    fontSize: 12,
                  }}
                  tickLine={false}
                  axisLine={{ stroke: 'rgba(255,255,255,0.2)' }}
                  tickFormatter={(value) => value.toLocaleString()}
                />
                <YAxis 
                  type="category"
                  dataKey="name" 
                  tick={{ 
                    fill: '#cbd5e1',
                    fontSize: 12,
                  }}
                  axisLine={false}
                  tickLine={false}
                  width={90}
                />
                <Tooltip 
                  formatter={(value) => [value.toLocaleString(), "Shipments"]}
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
                <Bar 
                  dataKey="value" 
                  radius={[0, 4, 4, 0]}
                  onMouseEnter={handleMouseEnter}
                  onMouseLeave={handleMouseLeave}
                  animationDuration={1500}
                >
                  {carrierData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={`url(#barGradient-${index % COLORS.length})`}
                      filter={activeIndex === index ? 'url(#glow)' : 'none'}
                      strokeWidth={activeIndex === index ? 1 : 0}
                      stroke="#fff"
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default CarrierDistributionChart;