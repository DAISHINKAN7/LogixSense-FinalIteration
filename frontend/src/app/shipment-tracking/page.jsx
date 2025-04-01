// src/app/shipment-tracking/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Loader2, Search, Package, Truck, Globe, AlertTriangle, CheckCircle, MapPin, Clock, RefreshCw } from 'lucide-react';

export default function ShipmentTrackingPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [shipmentDetails, setShipmentDetails] = useState(null);
  const [recentSearches, setRecentSearches] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [sampleAWBs, setSampleAWBs] = useState(["23258167594", "23258167605"]);
  const [isFetchingSamples, setIsFetchingSamples] = useState(false);

  // Fetch some sample AWBs for easy testing
  useEffect(() => {
    const fetchSampleAWBs = async () => {
      setIsFetchingSamples(true);
      try {
        const response = await fetch('/api/tracking/sample-awbs?count=5');
        if (response.ok) {
          const data = await response.json();
          if (Array.isArray(data) && data.length > 0) {
            setSampleAWBs(data);
          }
        }
      } catch (error) {
        console.error('Error fetching sample AWBs:', error);
      } finally {
        setIsFetchingSamples(false);
      }
    };

    fetchSampleAWBs();
  }, []);

  // Load recent searches from localStorage
  useEffect(() => {
    const savedSearches = localStorage.getItem('recentTrackingSearches');
    if (savedSearches) {
      try {
        setRecentSearches(JSON.parse(savedSearches));
      } catch (e) {
        console.error('Error parsing saved searches', e);
      }
    }
  }, []);

  // Save recent searches to localStorage
  const saveRecentSearch = (query) => {
    const updatedSearches = [query, ...recentSearches.filter(s => s !== query)].slice(0, 5);
    setRecentSearches(updatedSearches);
    localStorage.setItem('recentTrackingSearches', JSON.stringify(updatedSearches));
  };

  // Handle search
  const handleSearch = async (e) => {
    if (e && e.preventDefault) {
      e.preventDefault();
    }
    
    const query = searchQuery.trim();
    if (!query) return;
    
    setIsLoading(true);
    setErrorMessage('');
    
    try {
      const response = await fetch(`/api/tracking/${query}`);
      
      if (response.ok) {
        const data = await response.json();
        setShipmentDetails(data);
        saveRecentSearch(query);
      } else {
        if (response.status === 404) {
          setErrorMessage(`No shipment found with tracking ID "${query}"`);
        } else {
          try {
            const errorData = await response.json();
            setErrorMessage(errorData.detail || 'An error occurred while retrieving tracking information');
          } catch (e) {
            setErrorMessage('An error occurred while retrieving tracking information');
          }
        }
        setShipmentDetails(null);
      }
    } catch (error) {
      console.error('Error fetching tracking data:', error);
      setErrorMessage('An error occurred while retrieving tracking information. Please try again later.');
      setShipmentDetails(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Load a specific tracking ID
  const loadTrackingID = (id) => {
    setSearchQuery(id);
    setTimeout(() => {
      handleSearch();
    }, 0);
  };
  
  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return 'Pending';
    
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return 'Invalid Date';
      
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      console.error('Error formatting date', e);
      return 'Invalid Date';
    }
  };
  
  // Get status color
  const getStatusColor = (status) => {
    if (!status) return 'text-gray-600 dark:text-gray-400';
    
    const statusMap = {
      'Delivered': 'text-green-600 dark:text-green-500',
      'In Transit': 'text-blue-600 dark:text-blue-500',
      'Customs Clearance': 'text-amber-600 dark:text-amber-500',
      'Delayed': 'text-amber-600 dark:text-amber-500',
      'Exception': 'text-red-600 dark:text-red-500',
      'Processing': 'text-purple-600 dark:text-purple-500',
      'Pending': 'text-gray-600 dark:text-gray-400'
    };
    
    return statusMap[status] || 'text-gray-600 dark:text-gray-400';
  };
  
  // Get status background color
  const getStatusBgColor = (status) => {
    if (!status) return 'bg-gray-100 dark:bg-gray-800';
    
    const statusMap = {
      'Delivered': 'bg-green-100 dark:bg-green-900/20',
      'In Transit': 'bg-blue-100 dark:bg-blue-900/20',
      'Customs Clearance': 'bg-amber-100 dark:bg-amber-900/20',
      'Delayed': 'bg-amber-100 dark:bg-amber-900/20',
      'Exception': 'bg-red-100 dark:bg-red-900/20',
      'Processing': 'bg-purple-100 dark:bg-purple-900/20',
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

  // Refresh tracking data
  const handleRefresh = () => {
    if (shipmentDetails && shipmentDetails.id) {
      const trackingId = shipmentDetails.id.replace('AWB', '');
      loadTrackingID(trackingId);
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
            
            {/* Recent Searches and Sample AWBs */}
            <div className="flex flex-wrap items-center gap-2 text-sm">
              {recentSearches.length > 0 && (
                <>
                  <span className="text-gray-500 dark:text-gray-400">Recent:</span>
                  {recentSearches.map((search, index) => (
                    <button 
                      key={`recent-${index}`}
                      type="button"
                      onClick={() => loadTrackingID(search)}
                      className="px-3 py-1 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded-full text-sm"
                    >
                      {search}
                    </button>
                  ))}
                </>
              )}
              
              {sampleAWBs.length > 0 && (
                <>
                  <span className="text-gray-500 dark:text-gray-400 ml-auto">Sample AWBs:</span>
                  {sampleAWBs.map((awb, index) => (
                    <button 
                      key={`sample-${index}`}
                      type="button"
                      onClick={() => loadTrackingID(awb)}
                      className="px-3 py-1 bg-blue-50 hover:bg-blue-100 dark:bg-blue-900/20 dark:hover:bg-blue-800/20 rounded-full text-sm text-blue-700 dark:text-blue-400"
                    >
                      {awb}
                    </button>
                  ))}
                </>
              )}
              
              {isFetchingSamples && (
                <span className="flex items-center text-sm text-gray-500 dark:text-gray-400 ml-2">
                  <Loader2 className="w-3 h-3 animate-spin mr-1" />
                  Loading samples...
                </span>
              )}
            </div>
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
      
      {/* Error State */}
      {!isLoading && errorMessage && (
        <div className="py-12 text-center">
          <div className="mx-auto w-16 h-16 mb-4 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
            <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-500" />
          </div>
          <h3 className="text-lg font-medium mb-2">Error</h3>
          <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            {errorMessage}
          </p>
        </div>
      )}
      
      {/* Shipment Details */}
      {!isLoading && !errorMessage && shipmentDetails && (
        <div className="space-y-6">
          {/* Overview Card */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">Shipment {shipmentDetails.id}</CardTitle>
                <div className="flex items-center space-x-2">
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusBgColor(shipmentDetails.status)} ${getStatusColor(shipmentDetails.status)}`}>
                    {shipmentDetails.status}
                  </span>
                  <button 
                    onClick={handleRefresh} 
                    className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800"
                    title="Refresh tracking data"
                  >
                    <RefreshCw className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  </button>
                </div>
              </div>
              <CardDescription>
                From {shipmentDetails.origin?.location || 'Unknown'} to {shipmentDetails.destination?.location || 'Unknown'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Origin</p>
                  <p className="font-medium">{shipmentDetails.origin?.location || 'Unknown'}</p>
                  <p className="text-sm">{shipmentDetails.origin?.facility || 'Unknown'}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Departure: {formatDate(shipmentDetails.origin?.departureTime)}
                  </p>
                </div>
                
                <div className="space-y-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Destination</p>
                  <p className="font-medium">{shipmentDetails.destination?.location || 'Unknown'}</p>
                  <p className="text-sm">{shipmentDetails.destination?.facility || 'Unknown'}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Estimated arrival: {formatDate(shipmentDetails.destination?.estimatedArrival)}
                  </p>
                </div>
                
                <div className="space-y-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Current Location</p>
                  <p className="font-medium">{shipmentDetails.currentLocation?.location || 'Unknown'}</p>
                  <p className="text-sm">{shipmentDetails.currentLocation?.status || 'Unknown'}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Updated: {formatDate(shipmentDetails.currentLocation?.timestamp)}
                  </p>
                </div>
                
                <div className="space-y-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Shipment Details</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                    <p className="text-sm">Weight:</p>
                    <p className="text-sm font-medium">{shipmentDetails.details?.weight || 'Unknown'}</p>
                    
                    <p className="text-sm">Packages:</p>
                    <p className="text-sm font-medium">{shipmentDetails.details?.packages || 'Unknown'}</p>
                    
                    <p className="text-sm">Type:</p>
                    <p className="text-sm font-medium">{shipmentDetails.details?.type || 'Unknown'}</p>
                    
                    <p className="text-sm">Service:</p>
                    <p className="text-sm font-medium">{shipmentDetails.details?.service || 'Unknown'}</p>
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
                  {shipmentDetails.timeline && shipmentDetails.timeline.map((item, index) => (
                    <div key={index} className="relative flex items-start">
                      <div className={`absolute left-0 rounded-full mt-1.5 w-7 h-7 flex items-center justify-center ${
                        item.isCompleted 
                          ? 'bg-green-100 dark:bg-green-900/20' 
                          : 'bg-gray-100 dark:bg-gray-800'
                      }`}>
                        {item.isCompleted ? (
                          <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-500" />
                        ) : (
                          <Clock className="w-5 h-5 text-gray-400 dark:text-gray-600" />
                        )}
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
          {shipmentDetails.updates && shipmentDetails.updates.length > 0 && (
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
          )}
          
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
      
      {/* Empty State */}
      {!isLoading && !errorMessage && !shipmentDetails && !searchQuery && (
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