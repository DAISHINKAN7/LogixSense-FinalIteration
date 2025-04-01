import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

const worldRegions = [
  { name: 'North America', value: 35, color: '#3b82f6' },
  { name: 'Europe', value: 30, color: '#10b981' },
  { name: 'Asia', value: 25, color: '#f59e0b' },
  { name: 'South America', value: 5, color: '#8b5cf6' },
  { name: 'Africa', value: 3, color: '#ec4899' },
  { name: 'Oceania', value: 2, color: '#06b6d4' }
];

const Globe = () => {
  const canvasRef = useRef(null);
  const [rotation, setRotation] = useState(0);
  const [isHovering, setIsHovering] = useState(false);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setRotation(prev => (prev + 0.5) % 360);
    }, 50);
    
    return () => clearInterval(interval);
  }, []);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 10;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw globe
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#060b18';
    ctx.fill();
    
    // Draw grid lines (meridians)
    const numMeridians = 12;
    for (let i = 0; i < numMeridians; i++) {
      const angle = (i / numMeridians) * Math.PI * 2 + (rotation * Math.PI / 180);
      ctx.beginPath();
      ctx.ellipse(
        centerX, 
        centerY, 
        radius, 
        radius * 0.2, 
        0, 
        0, 
        Math.PI * 2
      );
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.stroke();
      
      // Draw latitude lines
      ctx.beginPath();
      ctx.ellipse(
        centerX, 
        centerY, 
        radius * Math.abs(Math.cos(angle)), // Use absolute value to prevent negative radius
        radius, 
        Math.PI/2, 
        0, 
        Math.PI * 2
      );
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.stroke();
    }
    
    // Draw highlighted regions
    const regionColors = {
      'North America': { x: -0.5, y: 0.2, r: 0.25, color: '#3b82f680' },
      'Europe': { x: 0.1, y: 0.15, r: 0.15, color: '#10b98180' },
      'Asia': { x: 0.4, y: 0.1, r: 0.3, color: '#f59e0b80' },
      'South America': { x: -0.3, y: 0.5, r: 0.2, color: '#8b5cf680' },
      'Africa': { x: 0.1, y: 0.4, r: 0.22, color: '#ec489980' },
      'Oceania': { x: 0.6, y: 0.5, r: 0.18, color: '#06b6d480' }
    };
    
    Object.entries(regionColors).forEach(([region, data]) => {
      // Apply rotation to x-coordinate
      const rotatedX = data.x * Math.cos(rotation * Math.PI / 180) - data.y * Math.sin(rotation * Math.PI / 180);
      
      // Only draw if the region is on the "visible" side of the globe
      if (rotatedX <= 0.8) {
        const x = centerX + rotatedX * radius;
        const y = centerY + data.y * radius;
        
        ctx.beginPath();
        ctx.arc(x, y, data.r * radius, 0, Math.PI * 2);
        ctx.fillStyle = data.color;
        ctx.fill();
        
        // Add glow effect
        ctx.shadowColor = data.color;
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.arc(x, y, data.r * radius * 0.5, 0, Math.PI * 2);
        ctx.fillStyle = data.color.replace('80', 'ff');
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    });
    
    // Draw outer glow
    const gradient = ctx.createRadialGradient(centerX, centerY, radius * 0.8, centerX, centerY, radius * 1.2);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0)');
    gradient.addColorStop(0.5, 'rgba(59, 130, 246, 0.05)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 1.2, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();
    
  }, [rotation]);
  
  return (
    <canvas 
      ref={canvasRef} 
      width={280} 
      height={280} 
      className="w-full h-full"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    />
  );
};

const ForecastGlobe = () => {
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);
  
  const handlePieClick = (data) => {
    setSelectedRegion(data.name);
  };
  
  // Top carriers by region
  const regionCarriers = {
    'North America': [
      { name: 'FedEx', value: 45 },
      { name: 'UPS', value: 35 },
      { name: 'DHL', value: 20 }
    ],
    'Europe': [
      { name: 'DHL', value: 40 },
      { name: 'UPS', value: 30 },
      { name: 'TNT', value: 30 }
    ],
    'Asia': [
      { name: 'Singapore Airlines', value: 35 },
      { name: 'Cathay Pacific', value: 35 },
      { name: 'Emirates', value: 30 }
    ],
    'South America': [
      { name: 'LATAM Cargo', value: 50 },
      { name: 'Azul Cargo', value: 30 },
      { name: 'DHL', value: 20 }
    ],
    'Africa': [
      { name: 'Ethiopian Airlines', value: 40 },
      { name: 'Kenya Airways', value: 30 },
      { name: 'Emirates', value: 30 }
    ],
    'Oceania': [
      { name: 'Qantas Freight', value: 50 },
      { name: 'Air New Zealand', value: 30 },
      { name: 'DHL', value: 20 }
    ]
  };
  
  // Regional growth predictions
  const regionGrowth = {
    'North America': '+4.5%',
    'Europe': '+3.8%',
    'Asia': '+7.2%',
    'South America': '+5.1%',
    'Africa': '+6.5%',
    'Oceania': '+3.2%'
  };
  
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-2 rounded-md shadow-md border border-gray-200 dark:border-gray-700">
          <p className="font-medium">{payload[0].name}</p>
          <p className="text-sm">{`${payload[0].value}% of shipments`}</p>
          <p className="text-xs text-green-600">{regionGrowth[payload[0].name]}</p>
        </div>
      );
    }
    return null;
  };
  
  return (
    <Card className={`transition-all duration-500 ${isExpanded ? 'col-span-2' : ''}`}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle>Global Forecast Distribution</CardTitle>
          <Button 
            variant="ghost" 
            size="sm" 
            className="h-8 w-8 p-0" 
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? '−' : '+'}
          </Button>
        </div>
        <CardDescription>
          Projected distribution of shipments by region
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex justify-center items-center">
            <Globe />
          </div>
          <div>
            <ResponsiveContainer width="100%" height={isExpanded ? 300 : 230}>
              <PieChart>
                <Pie
                  data={worldRegions}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                  onClick={handlePieClick}
                >
                  {worldRegions.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.color} 
                      stroke={selectedRegion === entry.name ? "#ffffff" : "transparent"}
                      strokeWidth={2}
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            
            {isExpanded && selectedRegion && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="mt-4"
              >
                <h4 className="font-medium text-center mb-2">{selectedRegion} Top Carriers</h4>
                <div className="space-y-2">
                  {regionCarriers[selectedRegion].map((carrier, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm">{carrier.name}</span>
                      <div className="flex items-center">
                        <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className="h-full"
                            style={{ 
                              width: `${carrier.value}%`,
                              backgroundColor: worldRegions.find(r => r.name === selectedRegion)?.color || '#3b82f6'
                            }}
                          />
                        </div>
                        <span className="text-xs ml-2">{carrier.value}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </CardContent>
      <CardFooter className="pt-0">
        <div className="w-full text-xs text-muted-foreground">
          <Badge variant="outline" className="mr-1">
            AI Model: Geographic Distribution Predictor
          </Badge>
          <Badge variant="outline">
            Confidence: 92%
          </Badge>
        </div>
      </CardFooter>
    </Card>
  );
};

export default ForecastGlobe;