// src/app/global-shipping/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Loader2, Filter, Download, RefreshCw } from 'lucide-react';

export default function GlobalShippingPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [shippingData, setShippingData] = useState({
    topRoutes: [],
    regionVolumes: {},
    recentShipments: []
  });

  useEffect(() => {
    // Simulate API call to fetch shipping data
    const fetchData = async () => {
      try {
        // In a real app, this would be an API call
        // For the prototype, we'll use mock data
        setTimeout(() => {
          setShippingData({
            topRoutes: [
              { id: 1, origin: 'Delhi, India', destination: 'New York, USA', volume: 312, growth: 8.4 },
              { id: 2, origin: 'Mumbai, India', destination: 'Dubai, UAE', volume: 245, growth: 15.2 },
              { id: 3, origin: 'Bangalore, India', destination: 'Singapore', volume: 183, growth: 6.7 },
              { id: 4, origin: 'Chennai, India', destination: 'London, UK', volume: 205, growth: -3.2 },
              { id: 5, origin: 'Hyderabad, India', destination: 'Tokyo, Japan', volume: 127, growth: 12.5 },
            ],
            regionVolumes: {
              'North America': 584,
              'Europe': 453,
              'Middle East': 368,
              'Asia Pacific': 724,
              'Africa': 125,
              'South America': 87
            },
            recentShipments: [
              { id: 'AWB10983762', origin: 'Delhi', destination: 'New York', status: 'In Transit', estimatedArrival: '2025-03-17' },
              { id: 'AWB10983571', origin: 'Mumbai', destination: 'Dubai', status: 'Delivered', estimatedArrival: '2025-03-15' },
              { id: 'AWB10983445', origin: 'Bangalore', destination: 'Singapore', status: 'Customs Clearance', estimatedArrival: '2025-03-16' },
              { id: 'AWB10983390', origin: 'Chennai', destination: 'London', status: 'Processing', estimatedArrival: '2025-03-19' },
              { id: 'AWB10983255', origin: 'Hyderabad', destination: 'Tokyo', status: 'Delivered', estimatedArrival: '2025-03-14' },
            ]
          });
          setIsLoading(false);
        }, 1500);
      } catch (error) {
        console.error('Error fetching shipping data:', error);
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Global Shipping Map</h1>
        <div className="flex items-center space-x-2">
          <button className="flex items-center space-x-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md px-3 py-1.5 text-sm">
            <Filter className="w-4 h-4" />
            <span>Filter</span>
          </button>
          <button className="flex items-center space-x-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md px-3 py-1.5 text-sm">
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
          <button className="flex items-center space-x-1 bg-blue-600 hover:bg-blue-700 text-white rounded-md px-3 py-1.5 text-sm">
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* World Map with Shipping Routes */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>Global Shipping Routes</CardTitle>
          <CardDescription>
            Visual representation of active shipping routes and volumes
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse flex items-center justify-center">
              <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
            </div>
          ) : (
            <div className="relative aspect-video bg-sky-50 dark:bg-sky-950 rounded-lg overflow-hidden">
              {/* This would be a real map component in a production implementation */}
              <WorldMapSVG />
              
              {/* Shipping routes - these would be dynamically generated in production */}
              <svg 
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 1000 500"
              >
                {/* Route 1: India to USA */}
                <path 
                  d="M700 250 Q 500 150, 250 200" 
                  stroke="#3b82f6" 
                  strokeWidth="2" 
                  fill="none" 
                  strokeDasharray="5,5"
                />
                <circle cx="700" cy="250" r="5" fill="#3b82f6" />
                <circle cx="250" cy="200" r="5" fill="#3b82f6" />
                
                {/* Route 2: India to UAE */}
                <path 
                  d="M700 250 Q 650 220, 600 230" 
                  stroke="#f59e0b" 
                  strokeWidth="2" 
                  fill="none" 
                  strokeDasharray="5,5"
                />
                <circle cx="700" cy="250" r="5" fill="#f59e0b" />
                <circle cx="600" cy="230" r="5" fill="#f59e0b" />
                
                {/* Route 3: India to Singapore */}
                <path 
                  d="M700 250 Q 720 280, 740 300" 
                  stroke="#10b981" 
                  strokeWidth="2" 
                  fill="none" 
                  strokeDasharray="5,5"
                />
                <circle cx="700" cy="250" r="5" fill="#10b981" />
                <circle cx="740" cy="300" r="5" fill="#10b981" />
                
                {/* Route 4: India to UK */}
                <path 
                  d="M700 250 Q 600 180, 450 160" 
                  stroke="#8b5cf6" 
                  strokeWidth="2" 
                  fill="none" 
                  strokeDasharray="5,5"
                />
                <circle cx="700" cy="250" r="5" fill="#8b5cf6" />
                <circle cx="450" cy="160" r="5" fill="#8b5cf6" />
                
                {/* Route 5: India to Japan */}
                <path 
                  d="M700 250 Q 750 200, 800 180" 
                  stroke="#ec4899" 
                  strokeWidth="2" 
                  fill="none" 
                  strokeDasharray="5,5"
                />
                <circle cx="700" cy="250" r="5" fill="#ec4899" />
                <circle cx="800" cy="180" r="5" fill="#ec4899" />
              </svg>
              
              {/* Legend */}
              <div className="absolute bottom-4 left-4 p-2 bg-white dark:bg-gray-800 rounded-md shadow-md text-xs">
                <div className="font-medium mb-1">Shipping Volume</div>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span>High</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                    <span>Medium</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-gray-500"></div>
                    <span>Low</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Dashboard Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Top Routes */}
        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle>Top Shipping Routes</CardTitle>
            <CardDescription>
              Most active shipping corridors by volume
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
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b dark:border-gray-700">
                      <th className="py-3 px-4 text-left">Origin</th>
                      <th className="py-3 px-4 text-left">Destination</th>
                      <th className="py-3 px-4 text-right">Volume</th>
                      <th className="py-3 px-4 text-right">Growth</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shippingData.topRoutes.map((route) => (
                      <tr 
                        key={route.id} 
                        className="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800"
                      >
                        <td className="py-3 px-4 font-medium">{route.origin}</td>
                        <td className="py-3 px-4">{route.destination}</td>
                        <td className="py-3 px-4 text-right">{route.volume}</td>
                        <td className="py-3 px-4 text-right">
                          <span className={route.growth >= 0 ? 'text-green-600 dark:text-green-500' : 'text-red-600 dark:text-red-500'}>
                            {route.growth >= 0 ? '+' : ''}{route.growth}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Region Volumes */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Volume by Region</CardTitle>
            <CardDescription>
              Shipment distribution across global regions
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                <div className="h-6 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-6 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-6 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-6 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-6 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                <div className="h-6 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              </div>
            ) : (
              <div className="space-y-4">
                {Object.entries(shippingData.regionVolumes).map(([region, volume]) => (
                  <div key={region} className="flex items-center">
                    <div className="w-32 min-w-[8rem] text-sm">{region}</div>
                    <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-600" 
                        style={{ width: `${(volume / 800) * 100}%` }}
                      ></div>
                    </div>
                    <div className="ml-3 text-sm font-medium w-12 text-right">{volume}</div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Shipments */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>Recent Global Shipments</CardTitle>
          <CardDescription>
            Latest shipments across international routes
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              <div className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {shippingData.recentShipments.map((shipment) => (
                <div 
                  key={shipment.id} 
                  className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-medium">{shipment.id}</span>
                    <span className={`
                      inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                      ${shipment.status === 'Delivered' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-500'
                        : shipment.status === 'In Transit'
                        ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-500'
                        : shipment.status === 'Customs Clearance'
                        ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/20 dark:text-amber-500'
                        : 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-500'
                      }
                    `}>
                      {shipment.status}
                    </span>
                  </div>
                  <div className="flex items-center mb-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                    <div className="mx-2 text-gray-500 dark:text-gray-400 text-sm">
                      {shipment.origin}
                    </div>
                    <div className="flex-1 h-px bg-gray-300 dark:bg-gray-600"></div>
                    <div className="mx-2 text-gray-500 dark:text-gray-400 text-sm">
                      {shipment.destination}
                    </div>
                    <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    Estimated arrival: {new Date(shipment.estimatedArrival).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// Simple world map SVG component
const WorldMapSVG = () => (
  <svg 
    viewBox="0 0 1000 500" 
    className="w-full h-full"
    style={{ filter: 'drop-shadow(0px 1px 2px rgba(0,0,0,0.1))' }}
  >
    <g fill="#e5e7eb" stroke="#9ca3af" strokeWidth="1">
      {/* Simplified world map paths */}
      <path d="M250,90 L340,90 L370,180 L230,200 Z" /> {/* North America */}
      <path d="M300,210 L350,240 L320,300 L270,260 Z" /> {/* South America */}
      <path d="M450,130 L510,110 L530,170 L440,180 Z" /> {/* Europe */}
      <path d="M530,180 L600,190 L590,230 L510,240 Z" /> {/* Middle East */}
      <path d="M500,250 L550,280 L520,320 L470,300 Z" /> {/* Africa */}
      <path d="M700,230 L750,210 L770,300 L680,290 Z" /> {/* India */}
      <path d="M790,170 L850,160 L860,220 L800,230 Z" /> {/* East Asia */}
      <path d="M750,310 L820,300 L850,350 L770,370 Z" /> {/* Australia */}
    </g>
    
    {/* Major shipping hubs */}
    <circle cx="700" cy="250" r="8" fill="#3b82f6" opacity="0.8" /> {/* India */}
    <circle cx="250" cy="180" r="6" fill="#3b82f6" opacity="0.8" /> {/* USA */}
    <circle cx="450" cy="160" r="5" fill="#3b82f6" opacity="0.8" /> {/* UK */}
    <circle cx="600" cy="230" r="5" fill="#3b82f6" opacity="0.8" /> {/* Dubai */}
    <circle cx="740" cy="300" r="5" fill="#3b82f6" opacity="0.8" /> {/* Singapore */}
    <circle cx="800" cy="180" r="5" fill="#3b82f6" opacity="0.8" /> {/* Japan */}
  </svg>
);