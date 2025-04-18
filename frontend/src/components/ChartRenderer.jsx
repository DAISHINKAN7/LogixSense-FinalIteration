import React from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';

// Define a set of colors to use for chart elements
const COLORS = [
  '#7C3AED', '#C026D3', '#06B6D4', '#10B981', '#F59E0B', 
  '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316'
];

/**
 * ChartRenderer component for rendering different types of charts
 * 
 * @param {Object} props - Component props
 * @param {string} props.type - Type of chart to render (bar, line, pie, scatter)
 * @param {Array} props.data - Data for the chart
 * @param {Object} props.config - Configuration for the chart
 * @param {string} props.description - Description of the chart
 */
const ChartRenderer = ({ type, data, config, description }) => {
  if (!data || data.length === 0) {
    return <div className="text-[#A0A0A0] text-center py-4">No data available for visualization</div>;
  }

  // Extract config values with defaults
  const { 
    xKey = 'name', 
    yKey = 'value', 
    nameKey = 'name', 
    valueKey = 'value',
    xLabel = 'X Axis',
    yLabel = 'Y Axis'
  } = config || {};

  // Render different chart types based on the type prop
  switch (type) {
    case 'bar':
      return (
        <div className="w-full h-full flex flex-col">
          {description && <div className="text-[#A0A0A0] text-sm mb-2">{description}</div>}
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey={xKey} 
                label={{ value: xLabel, position: 'bottom', offset: 0, fill: '#A0A0A0' }}
                tick={{ fill: '#A0A0A0' }}
              />
              <YAxis 
                label={{ value: yLabel, angle: -90, position: 'left', offset: -5, fill: '#A0A0A0' }}
                tick={{ fill: '#A0A0A0' }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1A1A1A', border: '1px solid #333' }}
                itemStyle={{ color: '#FFFFFF' }}
                labelStyle={{ color: '#A0A0A0' }}
              />
              <Legend wrapperStyle={{ color: '#A0A0A0' }} />
              <Bar dataKey={yKey} fill="#7C3AED" radius={[4, 4, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      );

    case 'line':
      return (
        <div className="w-full h-full flex flex-col">
          {description && <div className="text-[#A0A0A0] text-sm mb-2">{description}</div>}
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey={xKey} 
                label={{ value: xLabel, position: 'bottom', offset: 0, fill: '#A0A0A0' }}
                tick={{ fill: '#A0A0A0' }}
              />
              <YAxis 
                label={{ value: yLabel, angle: -90, position: 'left', offset: -5, fill: '#A0A0A0' }}
                tick={{ fill: '#A0A0A0' }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1A1A1A', border: '1px solid #333' }}
                itemStyle={{ color: '#FFFFFF' }}
                labelStyle={{ color: '#A0A0A0' }}
              />
              <Legend wrapperStyle={{ color: '#A0A0A0' }} />
              <Line 
                type="monotone" 
                dataKey={yKey} 
                stroke="#06B6D4" 
                strokeWidth={2}
                dot={{ fill: '#06B6D4', r: 4 }}
                activeDot={{ r: 6, stroke: '#06B6D4', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      );

    case 'pie':
      return (
        <div className="w-full h-full flex flex-col">
          {description && <div className="text-[#A0A0A0] text-sm mb-2">{description}</div>}
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                outerRadius={100}
                innerRadius={40}
                labelLine={false}
                dataKey={valueKey}
                nameKey={nameKey}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: '#1A1A1A', border: '1px solid #333' }}
                itemStyle={{ color: '#FFFFFF' }}
                labelStyle={{ color: '#A0A0A0' }}
              />
              <Legend wrapperStyle={{ color: '#A0A0A0' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      );

    case 'scatter':
      return (
        <div className="w-full h-full flex flex-col">
          {description && <div className="text-[#A0A0A0] text-sm mb-2">{description}</div>}
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 5, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey={xKey} 
                type="number" 
                name={xLabel}
                label={{ value: xLabel, position: 'bottom', offset: 0, fill: '#A0A0A0' }}
                tick={{ fill: '#A0A0A0' }}
              />
              <YAxis 
                dataKey={yKey} 
                type="number" 
                name={yLabel}
                label={{ value: yLabel, angle: -90, position: 'left', offset: -5, fill: '#A0A0A0' }}
                tick={{ fill: '#A0A0A0' }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1A1A1A', border: '1px solid #333' }}
                itemStyle={{ color: '#FFFFFF' }}
                labelStyle={{ color: '#A0A0A0' }}
                cursor={{ strokeDasharray: '3 3' }}
              />
              <Legend wrapperStyle={{ color: '#A0A0A0' }} />
              <Scatter name="Data Points" data={data} fill="#8884d8">
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      );

    default:
      return <div className="text-[#A0A0A0] text-center py-4">Unsupported visualization type: {type}</div>;
  }
};

export default ChartRenderer;