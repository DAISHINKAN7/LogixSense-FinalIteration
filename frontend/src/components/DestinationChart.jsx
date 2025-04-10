import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Sector } from 'recharts';
import { RefreshCw } from 'lucide-react';

// Modern gradient color palette for a futuristic look
const COLORS = [
  '#3B82F6', // blue
  '#10B981', // emerald
  '#F59E0B', // amber
  '#8B5CF6', // violet
  '#EC4899', // pink
  '#06B6D4', // cyan
  '#14B8A6', // teal
  '#F97316'  // orange
];

const DestinationChart = ({ data = [] }) => {
  const [activeIndex, setActiveIndex] = useState(null);
  
  // Ensure we have valid data to display
  const chartData = Array.isArray(data) ? data.filter(item => 
    item && typeof item === 'object' && 
    item.name && 
    (typeof item.value === 'number' || !isNaN(Number(item.value)))
  ) : [];
  
  // Convert string values to numbers if needed
  const processedData = chartData.map(item => ({
    ...item,
    value: typeof item.value === 'number' ? item.value : Number(item.value)
  }));
  
  const loading = processedData.length === 0;

  const onPieEnter = (_, index) => {
    setActiveIndex(index);
  };

  const onPieLeave = () => {
    setActiveIndex(null);
  };

  // Customize the appearance of the active sector
  const renderActiveShape = (props) => {
    const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props;
    
    return (
      <g>
        <Sector
          cx={cx}
          cy={cy}
          innerRadius={innerRadius}
          outerRadius={outerRadius + 12}
          startAngle={startAngle}
          endAngle={endAngle}
          fill={fill}
          opacity={0.9}
          stroke="#fff"
          strokeWidth={2}
          style={{ filter: 'drop-shadow(0px 0px 8px rgba(0, 0, 0, 0.3))' }}
        />
      </g>
    );
  };

  const formatTooltip = (value, name) => {
    if (typeof value === 'number') {
      return [`${value.toLocaleString()} shipments`, name];
    }
    return [`${value} shipments`, name];
  };

  return (
    <Card className="h-full bg-gradient-to-br from-gray-900 to-gray-800 border-none text-white shadow-xl">
      <CardHeader className="pb-2 border-b border-gray-700">
        <CardTitle className="text-lg font-medium flex items-center">
          <span className="inline-block w-2 h-6 bg-blue-500 mr-2 rounded-sm"></span>
          Top Shipping Destinations
        </CardTitle>
      </CardHeader>
      <CardContent className="h-[calc(100%-3rem)] pt-4">
        {loading ? (
          <div className="flex justify-center items-center h-full">
            <RefreshCw className="h-8 w-8 text-blue-400 animate-spin" />
          </div>
        ) : (
          <div className="flex h-full">
            {/* Chart area */}
            <div className="flex-1" style={{ minHeight: "230px" }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <defs>
                    {COLORS.map((color, index) => (
                      <linearGradient key={`gradient-${index}`} id={`gradient-${index}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={color} stopOpacity={0.9}/>
                        <stop offset="100%" stopColor={color} stopOpacity={0.7}/>
                      </linearGradient>
                    ))}
                  </defs>
                  <Pie
                    activeIndex={activeIndex}
                    activeShape={renderActiveShape}
                    data={processedData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={90}
                    innerRadius={52}
                    fill="#8884d8"
                    dataKey="value"
                    onMouseEnter={onPieEnter}
                    onMouseLeave={onPieLeave}
                    paddingAngle={2}
                    stroke="#1b1e23"
                    strokeWidth={1}
                  >
                    {processedData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={`url(#gradient-${index % COLORS.length})`}
                      />
                    ))}
                    <Label
                      position="center"
                      value="Destinations"
                      fill="#fff"
                      style={{ fontSize: '16px', fontWeight: 'bold' }}
                    />
                  </Pie>
                  <Tooltip 
                    formatter={formatTooltip}
                    contentStyle={{ 
                      backgroundColor: 'rgba(30, 41, 59, 0.9)',
                      borderRadius: '8px',
                      border: 'none',
                      boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
                      color: '#fff'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            {/* Legend on the right side - vertical list */}
            <div className="w-48 ml-2 flex flex-col space-y-1 overflow-y-auto pr-2" style={{ scrollbarWidth: 'thin' }}>
              {processedData.map((entry, index) => (
                <div 
                  key={`legend-${index}`} 
                  className={`flex flex-col p-3 rounded-md transition-all ${
                    activeIndex === index ? 'bg-gray-800' : ''
                  }`}
                  onMouseEnter={() => setActiveIndex(index)}
                  onMouseLeave={() => setActiveIndex(null)}
                >
                  <div className="flex items-center">
                    <div 
                      className="w-3 h-10 mr-2 rounded-sm"
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    ></div>
                    <div className="flex flex-col">
                      <div className="text-sm font-medium">{entry.name}</div>
                      <div className="text-xs text-gray-400">{entry.value.toLocaleString()} shipments</div>
                    </div>
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

export default DestinationChart;

// Don't forget to add this if you need it
const Label = ({ value, position, fill, style, ...props }) => {
  return (
    <text 
      x={props.cx} 
      y={props.cy} 
      fill={fill} 
      textAnchor="middle" 
      dominantBaseline="middle"
      style={style}
    >
      {value}
    </text>
  );
};