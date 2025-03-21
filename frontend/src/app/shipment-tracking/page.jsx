// src/app/shipment-tracking/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Loader2, Search, Package, Truck, Globe, AlertTriangle, CheckCircle, MapPin } from 'lucide-react';

export default function ShipmentTrackingPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [shipmentDetails, setShipmentDetails] = useState(null);
  const [recentSearches, setRecentSearches] = useState([
    'AWB10983762',
    'AWB10983571',
    'AWB10983445'
  ]);

  // Handle search
  const handleSearch = (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    
    // Simulate API call with 1 second delay
    setTimeout(() => {
      // Mock data for demonstration
      const mockShipment = {
        id: searchQuery.trim(),
        status: 'In Transit',
        origin: {
          location: 'Delhi, India',
          departureTime: '2025-03-15T08:30:00Z',
          facility: 'Delhi Air Cargo Terminal'
        },
        destination: {
          location: 'New York, USA',
          estimatedArrival: '2025-03-18T14:45:00Z',
          facility: 'JFK Airport Cargo Terminal'
        },
        currentLocation: {
          location: 'Dubai, UAE',
          timestamp: '2025-03-16T19:15:00Z',
          status: 'In Transit - Connection'
        },
        details: {
          weight: '328.5 kg',
          packages: 4,
          dimensions: '120x80x75 cm',
          type: 'Commercial Goods',
          service: 'Express Air Freight'
        },
        timeline: [
          { status: 'Order Created', location: 'Delhi, India', timestamp: '2025-03-14T15:20:00Z', isCompleted: true },
          { status: 'Package Received', location: 'Delhi, India', timestamp: '2025-03-15T06:45:00Z', isCompleted: true },
          { status: 'Departed Origin', location: 'Delhi, India', timestamp: '2025-03-15T08:30:00Z', isCompleted: true },
          { status: 'Arrived at Connection', location: 'Dubai, UAE', timestamp: '2025-03-16T19:15:00Z', isCompleted: true },
          { status: 'Departed Connection', location: 'Dubai, UAE', timestamp: '2025-03-17T01:30:00Z', isCompleted: false },
          { status: 'Customs Clearance', location: 'New York, USA', timestamp: null, isCompleted: false },
          { status: 'Out for Delivery', location: 'New York, USA', timestamp: null, isCompleted: false },
          { status: 'Delivered', location: 'New York, USA', timestamp: null, isCompleted: false }
        ],
        updates: [
          { type: 'info', message: 'Shipment has departed from Delhi', timestamp: '2025-03-15T08:35:00Z' },
          { type: 'warning', message: 'Slight delay at Dubai due to weather conditions', timestamp: '2025-03-16T20:00:00Z' },
          { type: 'info', message: 'Shipment is scheduled to depart Dubai in the next 2 hours', timestamp: '2025-03-16T23:15:00Z' }
        ]
      };
      
      setShipmentDetails(mockShipment);
      
      // Add to recent searches if not already there
      if (!recentSearches.includes(searchQuery.trim())) {
        setRecentSearches(prev => [searchQuery.trim(), ...prev].slice(0, 5));
      }
      
      setIsLoading(false);
    }, 1000);
  };
  
  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return 'Pending';
    
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };
  
  // Get status color
  const getStatusColor = (status) => {
    const statusMap = {
      'Delivered': 'text-green-600 dark:text-green-500',
      'In Transit': 'text-blue-600 dark:text-blue-500',
      'Delayed': 'text-amber-600 dark:text-amber-500',
      'Exception': 'text-red-600 dark:text-red-500',
      'Pending': 'text-gray-600 dark:text-gray-400'
    };
    
    return statusMap[status] || 'text-gray-600 dark:text-gray-400';
  };
  
  // Get status background color
  const getStatusBgColor = (status) => {
    const statusMap = {
      'Delivered': 'bg-green-100 dark:bg-green-900/20',
      'In Transit': 'bg-blue-100 dark:bg-blue-900/20',
      'Delayed': 'bg-amber-100 dark:bg-amber-900/20',
      'Exception': 'bg-red-100 dark:bg-red-900/20',
      'Pending': 'bg-gray-100 dark:bg-gray-800'
    };
    
    return statusMap[status] || 'bg-gray-100 dark:bg-gray-800';
  };
  
  // Get update icon
  const getUpdateIcon = (type) => {
    switch (type) {
      case 'info':
        return <Globe className="w-5 h-5 text-blue-600 dark:text-blue-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-500" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-500" />;
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-500" />;
      default:
        return <Globe className="w-5 h-5 text-gray-600 dark:text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Shipment Tracking</h1>
      </div>
      
      {/* Search Form */}
      <Card>
        <CardContent className="pt-6">
          <form onSubmit={handleSearch} className="flex flex-col space-y-4">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                <Search className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              </div>
              <input
                type="text"
                className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white"
                placeholder="Enter AWB number, Order ID, or Reference number"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                required
              />
              <button
                type="submit"
                disabled={isLoading}
                className="absolute right-2.5 bottom-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg text-sm px-4 py-1 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Track'}
              </button>
            </div>
            
            {/* Recent Searches */}
            {recentSearches.length > 0 && (
              <div className="flex flex-wrap gap-2 items-center text-sm">
                <span className="text-gray-500 dark:text-gray-400">Recent:</span>
                {recentSearches.map((search, index) => (
                  <button 
                    key={index}
                    type="button"
                    onClick={() => {
                      setSearchQuery(search);
                      // Automatically submit the form
                      setTimeout(() => handleSearch({ preventDefault: () => {} }), 0);
                    }}
                    className="px-3 py-1 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded-full text-sm"
                  >
                    {search}
                  </button>
                ))}
              </div>
            )}
          </form>
        </CardContent>
      </Card>
      
      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600 mr-2" />
          <span className="text-lg font-medium">Tracking shipment...</span>
        </div>
      )}
      
      {/* Shipment Details */}
      {!isLoading && shipmentDetails && (
        <div className="space-y-6">
          {/* Overview Card */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">Shipment {shipmentDetails.id}</CardTitle>
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusBgColor(shipmentDetails.status)} ${getStatusColor(shipmentDetails.status)}`}>
                  {shipmentDetails.status}
                </span>
              </div>
              <CardDescription>
                From {shipmentDetails.origin.location} to {shipmentDetails.destination.location}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Origin</p>
                  <p className="font-medium">{shipmentDetails.origin.location}</p>
                  <p className="text-sm">{shipmentDetails.origin.facility}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Departure: {formatDate(shipmentDetails.origin.departureTime)}
                  </p>
                </div>
                
                <div className="space-y-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Destination</p>
                  <p className="font-medium">{shipmentDetails.destination.location}</p>
                  <p className="text-sm">{shipmentDetails.destination.facility}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Estimated arrival: {formatDate(shipmentDetails.destination.estimatedArrival)}
                  </p>
                </div>
                
                <div className="space-y-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Current Location</p>
                  <p className="font-medium">{shipmentDetails.currentLocation.location}</p>
                  <p className="text-sm">{shipmentDetails.currentLocation.status}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Updated: {formatDate(shipmentDetails.currentLocation.timestamp)}
                  </p>
                </div>
                
                <div className="space-y-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Shipment Details</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                    <p className="text-sm">Weight:</p>
                    <p className="text-sm font-medium">{shipmentDetails.details.weight}</p>
                    
                    <p className="text-sm">Packages:</p>
                    <p className="text-sm font-medium">{shipmentDetails.details.packages}</p>
                    
                    <p className="text-sm">Service:</p>
                    <p className="text-sm font-medium">{shipmentDetails.details.service}</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Timeline Card */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Shipping Timeline</CardTitle>
              <CardDescription>
                Track the journey of your shipment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative">
                {/* Timeline line */}
                <div className="absolute left-3.5 top-0 h-full w-0.5 bg-gray-200 dark:bg-gray-700"></div>
                
                <div className="space-y-6">
                  {shipmentDetails.timeline.map((item, index) => (
                    <div key={index} className="relative flex items-start">
                      <div className={`absolute left-0 rounded-full mt-1.5 w-7 h-7 flex items-center justify-center ${
                        item.isCompleted 
                          ? 'bg-green-100 dark:bg-green-900/20' 
                          : 'bg-gray-100 dark:bg-gray-800'
                      }`}>
                        <CheckCircle className={`w-5 h-5 ${
                          item.isCompleted 
                            ? 'text-green-600 dark:text-green-500' 
                            : 'text-gray-400 dark:text-gray-600'
                        }`} />
                      </div>
                      
                      <div className="ml-10">
                        <h3 className={`text-base font-medium ${
                          item.isCompleted 
                            ? 'text-gray-900 dark:text-white' 
                            : 'text-gray-500 dark:text-gray-400'
                        }`}>
                          {item.status}
                        </h3>
                        
                        <div className="flex items-center mt-1">
                          <MapPin className="w-4 h-4 text-gray-500 dark:text-gray-400 mr-1" />
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {item.location}
                          </span>
                        </div>
                        
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                          {item.timestamp ? formatDate(item.timestamp) : 'Scheduled'}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Updates Card */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Shipment Updates</CardTitle>
              <CardDescription>
                Latest updates and notifications
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {shipmentDetails.updates.map((update, index) => (
                  <div 
                    key={index} 
                    className={`p-4 rounded-lg ${
                      update.type === 'warning' 
                        ? 'bg-amber-50 dark:bg-amber-900/10' 
                        : update.type === 'error'
                        ? 'bg-red-50 dark:bg-red-900/10'
                        : update.type === 'success'
                        ? 'bg-green-50 dark:bg-green-900/10'
                        : 'bg-blue-50 dark:bg-blue-900/10'
                    }`}
                  >
                    <div className="flex items-start">
                      <div className="mr-3 mt-0.5">
                        {getUpdateIcon(update.type)}
                      </div>
                      <div>
                        <p className="font-medium mb-1">{update.message}</p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          {formatDate(update.timestamp)}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          
          {/* Actions */}
          <div className="flex flex-wrap gap-3">
            <button className="inline-flex items-center bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg text-sm px-5 py-2.5">
              <Package className="w-4 h-4 mr-2" />
              Download POD
            </button>
            <button className="inline-flex items-center bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-white font-medium rounded-lg text-sm px-5 py-2.5 border border-gray-200 dark:border-gray-700">
              <Truck className="w-4 h-4 mr-2" />
              Request Delivery Change
            </button>
            <button className="inline-flex items-center bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-white font-medium rounded-lg text-sm px-5 py-2.5 border border-gray-200 dark:border-gray-700">
              <AlertTriangle className="w-4 h-4 mr-2" />
              Report Issue
            </button>
          </div>
        </div>
      )}
      
      {/* No Results State */}
      {!isLoading && !shipmentDetails && searchQuery && (
        <div className="py-12 text-center">
          <div className="mx-auto w-16 h-16 mb-4 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center">
            <Package className="w-8 h-8 text-gray-500 dark:text-gray-400" />
          </div>
          <h3 className="text-lg font-medium mb-2">No shipment found</h3>
          <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            We couldn't find any shipment with the tracking number "{searchQuery}". 
            Please check the number and try again.
          </p>
        </div>
      )}
      
      {/* Empty State */}
      {!isLoading && !shipmentDetails && !searchQuery && (
        <div className="py-12 text-center">
          <div className="mx-auto w-16 h-16 mb-4 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center">
            <Globe className="w-8 h-8 text-gray-500 dark:text-gray-400" />
          </div>
          <h3 className="text-lg font-medium mb-2">Track your shipment</h3>
          <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            Enter your tracking number, order ID, or reference number above to track your shipment's status and location.
          </p>
          
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <button className="inline-flex items-center bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-white font-medium rounded-lg text-sm px-5 py-2.5 border border-gray-200 dark:border-gray-700">
              <Truck className="w-4 h-4 mr-2" />
              My Shipments
            </button>
            <button className="inline-flex items-center bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-white font-medium rounded-lg text-sm px-5 py-2.5 border border-gray-200 dark:border-gray-700">
              <Package className="w-4 h-4 mr-2" />
              Create Shipment
            </button>
          </div>
        </div>
      )}
    </div>
  );
}