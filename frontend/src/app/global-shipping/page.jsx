'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import StatusBadge from '@/components/ui/status_badge';
import { Loader2, Filter, Download, RefreshCw, AlertTriangle, X } from 'lucide-react';
import WorldMapSVG from '@/components/WorldMapSVG';

export default function GlobalShippingPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [shippingData, setShippingData] = useState({
    topRoutes: [],
    regionVolumes: {},
    recentShipments: []
  });
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [routeDetails, setRouteDetails] = useState(null);
  const [isRouteDetailsLoading, setIsRouteDetailsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  useEffect(() => {
    fetchShippingData();
  }, []);

  const fetchShippingData = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const response = await fetch('/api/global-shipping/overview');
      if (!response.ok) {
        throw new Error('Failed to fetch shipping data');
      }
      const data = await response.json();
      setShippingData(data);
    } catch (error) {
      console.error('Error fetching shipping data:', error);
      setErrorMessage('Failed to load shipping data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  const fetchRouteDetails = async (originCode, destinationCode) => {
    if (!originCode || !destinationCode) return;
    
    setIsRouteDetailsLoading(true);
    setErrorMessage(null);
    try {
      const response = await fetch(`/api/global-shipping/route-details/${originCode}/${destinationCode}`);
      if (!response.ok) {
        throw new Error('Failed to fetch route details');
      }
      const data = await response.json();
      setRouteDetails(data);
    } catch (error) {
      console.error('Error fetching route details:', error);
      setErrorMessage('Failed to load route details. Please try again.');
    } finally {
      setIsRouteDetailsLoading(false);
    }
  };
  
  const handleRouteClick = (route) => {
    setSelectedRoute(route);
    fetchRouteDetails(route.originCode, route.destinationCode);
  };
  
  const handleRefresh = () => {
    fetchShippingData();
    if (selectedRoute) {
      fetchRouteDetails(selectedRoute.originCode, selectedRoute.destinationCode);
    }
  };
  
  return (
    <div className="space-y-6">
      {/* Header */}
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
          <button 
            className="flex items-center space-x-1 bg-blue-600 hover:bg-blue-700 text-white rounded-md px-3 py-1.5 text-sm"
            onClick={handleRefresh}
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">{errorMessage}</span>
        </div>
      )}

      {/* World Map with Shipping Routes */}
      <Card className="overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle>Global Shipping Routes</CardTitle>
          <CardDescription>
            Visual representation of active shipping routes and volumes
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse flex items-center justify-center">
              <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
            </div>
          ) : (
            <div className="relative aspect-video bg-transparent rounded-lg overflow-hidden">
              <WorldMapSVG />
              
              {/* Shipping routes - generate dynamically based on top routes */}
              <svg 
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 1000 500"
              >
                {shippingData.topRoutes && shippingData.topRoutes.slice(0, 5).map((route, index) => {
                  const colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FDCB6E", "#6C5CE7"];
                  const color = colors[index % colors.length];
                  
                  // Generate route coordinates based on origin/destination codes
                  const coordinates = getRouteCoordinates(route.originCode, route.destinationCode);
                  
                  return (
                    <g key={route.id}>
                      <path 
                        d={`M${coordinates.startX} ${coordinates.startY} Q ${coordinates.controlX} ${coordinates.controlY}, ${coordinates.endX} ${coordinates.endY}`}
                        stroke={color} 
                        strokeWidth="2" 
                        fill="none" 
                        strokeDasharray="5,5"
                        opacity="0.7"
                      />
                    </g>
                  );
                })}
              </svg>
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
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-8 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                ))}
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
                    {shippingData.topRoutes && shippingData.topRoutes.map((route) => (
                      <tr 
                        key={route.id} 
                        className={`border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer ${
                          selectedRoute && selectedRoute.id === route.id ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                        }`}
                        onClick={() => handleRouteClick(route)}
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
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="h-6 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                {Object.entries(shippingData.regionVolumes || {}).map(([region, volume]) => (
                  <div key={region} className="flex items-center">
                    <div className="w-32 min-w-[8rem] text-sm">{region}</div>
                    <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-600" 
                        style={{ width: `${(volume / getMaxRegionVolume(shippingData.regionVolumes)) * 100}%` }}
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

      {/* Route Details (Shown when a route is selected) */}
      {selectedRoute && (
        <Card>
          <CardHeader className="pb-2 flex flex-row items-start justify-between">
            <div>
              <CardTitle>Route Details: {selectedRoute.origin} to {selectedRoute.destination}</CardTitle>
              <CardDescription>
                Detailed analytics for selected shipping corridor
              </CardDescription>
            </div>
            <button 
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
              onClick={() => setSelectedRoute(null)}
            >
              <span className="sr-only">Close</span>
              <X className="w-5 h-5" />
            </button>
          </CardHeader>
          <CardContent>
            {isRouteDetailsLoading ? (
              <div className="space-y-4">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="h-24 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                ))}
              </div>
            ) : routeDetails ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Route Overview */}
                <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
                  <h3 className="font-medium text-sm text-gray-500 dark:text-gray-400 mb-3">Route Overview</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">Distance</span>
                      <span className="font-medium">{routeDetails.routeInfo.distanceKm.toLocaleString()} km</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">Transit Time</span>
                      <span className="font-medium">{routeDetails.routeInfo.avgTransitDays} days</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">Total Shipments</span>
                      <span className="font-medium">{routeDetails.routeInfo.totalShipments.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">On-Time Rate</span>
                      <span className="font-medium">{routeDetails.routeInfo.onTimePerformance}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-300">Risk Score</span>
                      <span className={`font-medium ${getRiskColorClass(routeDetails.routeInfo.riskScore)}`}>
                        {routeDetails.routeInfo.riskScore}
                      </span>
                      <span className={`font-medium ${getRiskColorClass(routeDetails.routeInfo.riskScore)}`}>
                        {routeDetails.routeInfo.riskScore}
                      </span>
                    </div>
                  </div>
                </div>
                
                {/* Airline Services */}
                <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
                  <h3 className="font-medium text-sm text-gray-500 dark:text-gray-400 mb-3">Airline Services</h3>
                  <div className="space-y-3">
                    {routeDetails.serviceDetails.airlines.map((airline, idx) => (
                      <div key={idx} className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-300 truncate max-w-[70%]">
                          {airline.name || "Unknown Carrier"}
                        </span>
                        <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-full">
                          {airline.shipmentCount} shipments
                        </span>
                      </div>
                    ))}
                    {routeDetails.serviceDetails.airlines.length === 0 && (
                      <div className="text-sm text-gray-500 italic">No airline data available</div>
                    )}
                  </div>
                </div>
                
                {/* Commodity Types */}
                <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
                  <h3 className="font-medium text-sm text-gray-500 dark:text-gray-400 mb-3">Top Commodities</h3>
                  <div className="space-y-3">
                    {routeDetails.serviceDetails.topCommodities.map((commodity, idx) => (
                      <div key={idx} className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-300 truncate max-w-[70%]">
                          {commodity.name || "Unknown"}
                        </span>
                        <span className="text-xs px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 rounded-full">
                          {commodity.percentage}%
                        </span>
                      </div>
                    ))}
                    {routeDetails.serviceDetails.topCommodities.length === 0 && (
                      <div className="text-sm text-gray-500 italic">No commodity data available</div>
                    )}
                  </div>
                </div>
                
                {/* Volume Statistics */}
                <div className="md:col-span-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
                  <h3 className="font-medium text-sm text-gray-500 dark:text-gray-400 mb-3">Monthly Volume Trend</h3>
                  <div className="h-64">
                    <MonthlyTrendChart data={routeDetails.volumeStats.monthlyTrend} />
                  </div>
                </div>
                
                {/* Flight Frequency */}
                <div className="md:col-span-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
                  <h3 className="font-medium text-sm text-gray-500 dark:text-gray-400 mb-3">Weekly Flight Schedule</h3>
                  <div className="grid grid-cols-7 gap-2">
                    {routeDetails.serviceDetails.frequencies.map((day) => (
                      <div key={day.day} className="flex flex-col items-center">
                        <div className="text-sm font-medium mb-2">{day.day.substring(0, 3)}</div>
                        <div className="flex flex-col items-center space-y-1">
                          {[...Array(day.flights)].map((_, i) => (
                            <div key={i} className="w-6 h-1 bg-blue-500 rounded-full"></div>
                          ))}
                          {day.flights === 0 && (
                            <div className="text-xs text-gray-400">-</div>
                          )}
                        </div>
                        <div className="mt-2 text-xs">{day.flights} flight{day.flights !== 1 ? 's' : ''}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center p-8 text-gray-500">
                <AlertTriangle className="w-5 h-5 mr-2" />
                <span>Route details unavailable</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}
      {/* Recent Shipments */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>Recent Global Shipments</CardTitle>
          <CardDescription>
            Latest actual shipments from dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-12 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {shippingData.recentShipments && shippingData.recentShipments.length > 0 ? (
                shippingData.recentShipments.map((shipment) => (
                  <div 
                    key={shipment.id} 
                    className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium">{shipment.id}</span>
                      <StatusBadge status={shipment.status} />
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
                      Date: {new Date(shipment.actualDate || shipment.estimatedArrival).toLocaleDateString()}
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-2 p-6 text-center text-gray-500 dark:text-gray-400">
                  <AlertTriangle className="w-5 h-5 mx-auto mb-2" />
                  <p>No shipment data available in the dataset.</p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// Monthly trend chart component
const MonthlyTrendChart = ({ data }) => {
  // If no data, return an empty box
  if (!data || data.length === 0) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-gray-100 dark:bg-gray-800 rounded">
        <span className="text-gray-500 dark:text-gray-400">No trend data available</span>
      </div>
    );
  }

  // Find the maximum shipment count to scale the chart
  const maxShipments = Math.max(...data.map(item => item.shipments));
  
  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 relative">
        <div className="absolute inset-0 flex items-end">
          {data.map((item, index) => {
            const height = `${(item.shipments / maxShipments) * 100}%`;
            return (
              <div key={index} className="flex-1 flex flex-col items-center justify-end h-full">
                <div 
                  style={{ height }} 
                  className="w-full max-w-[30px] mx-auto bg-blue-500 dark:bg-blue-600 rounded-t"
                ></div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="h-6 flex">
        {data.map((item, index) => (
          <div key={index} className="flex-1 text-center text-xs mt-1">{item.month}</div>
        ))}
      </div>
    </div>
  );
};
const getRouteCoordinates = (originCode, destinationCode) => {
  const locations = {
    // Updated hub locations to match new map design
    'MEM': { x: 250, y: 150 },    // Memphis
    'DXB': { x: 570, y: 200 },    // Dubai
    'LHR': { x: 480, y: 120 },    // London
    'DEL': { x: 700, y: 220 },    // Delhi
    'SIN': { x: 740, y: 250 },    // Singapore
    'NRT': { x: 780, y: 150 }     // Tokyo (Narita)
  };
  
  // Default coordinates if code not found
  const defaultOrigin = { x: 500, y: 200 };
  const defaultDestination = { x: 600, y: 150 };
  
  // Get coordinates or use defaults
  const origin = locations[originCode] || defaultOrigin;
  const destination = locations[destinationCode] || defaultDestination;
  
  // Calculate a control point for the curve
  const midX = (origin.x + destination.x) / 2;
  const midY = (origin.y + destination.y) / 2;
  
  return {
    startX: origin.x,
    startY: origin.y,
    endX: destination.x,
    endY: destination.y,
    controlX: midX,
    controlY: midY - 50  // Adjusted for more subtle curvature
  };
};

// Helper function to get the maximum value in region volumes
function getMaxRegionVolume(regionVolumes) {
  if (!regionVolumes || Object.keys(regionVolumes).length === 0) {
    return 1; // Avoid division by zero
  }
  return Math.max(...Object.values(regionVolumes));
}

// Helper function to get color class based on risk score
function getRiskColorClass(score) {
  if (score >= 70) return 'text-red-600 dark:text-red-500';
  if (score >= 50) return 'text-amber-600 dark:text-amber-500';
  return 'text-green-600 dark:text-green-500';
}