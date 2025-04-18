'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { 
  Loader2, Search, Package, Truck, Globe, AlertTriangle, CheckCircle, 
  MapPin, Clock, RefreshCw, TrendingUp, Clipboard, Download, ArrowRight, Info 
} from 'lucide-react';

export default function ShipmentTrackingPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [shipmentDetails, setShipmentDetails] = useState(null);
  const [recentSearches, setRecentSearches] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [sampleAWBs, setSampleAWBs] = useState(["23258167594", "23258167605"]);
  const [isFetchingSamples, setIsFetchingSamples] = useState(false);
  const [activeTab, setActiveTab] = useState('timeline');

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
        return <Info className="w-5 h-5 text-blue-600 dark:text-blue-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-500" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-500" />;
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-500" />;
      default:
        return <Info className="w-5 h-5 text-gray-600 dark:text-gray-400" />;
    }
  };

  // Refresh tracking data
  const handleRefresh = () => {
    if (shipmentDetails && shipmentDetails.id) {
      const trackingId = shipmentDetails.id.replace('AWB', '');
      loadTrackingID(trackingId);
    }
  };
  
  // Calculate estimated delivery time remaining
  const getDeliveryTimeRemaining = () => {
    if (!shipmentDetails || !shipmentDetails.destination || !shipmentDetails.destination.estimatedArrival) {
      return null;
    }
    
    const now = new Date();
    const estimatedArrival = new Date(shipmentDetails.destination.estimatedArrival);
    
    if (isNaN(estimatedArrival.getTime()) || estimatedArrival < now) {
      return null;
    }
    
    const diffMs = estimatedArrival - now;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor((diffMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    
    if (diffDays > 0) {
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ${diffHours} hour${diffHours > 1 ? 's' : ''}`;
    } else {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''}`;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Shipment Tracking</h1>
      </div>
      
      {/* Search Form */}
      <Card className="overflow-hidden border-2 border-blue-100 dark:border-blue-900/30">
        <CardContent className="p-6 bg-gradient-to-br from-blue-50 to-white dark:from-blue-950/30 dark:to-gray-900">
          <form onSubmit={handleSearch} className="flex flex-col space-y-4">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 flex items-center pl-4 pointer-events-none">
                <Search className="w-5 h-5 text-blue-500 dark:text-blue-400" />
              </div>
              <input
                type="text"
                className="bg-white border border-blue-200 text-gray-900 text-sm rounded-full focus:ring-blue-500 focus:border-blue-500 block w-full pl-12 p-3.5 dark:bg-gray-800 dark:border-blue-800 dark:placeholder-gray-400 dark:text-white shadow-sm"
                placeholder="Enter AWB number, Order ID, or Reference number"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                required
              />
              <button
                type="submit"
                disabled={isLoading}
                className="absolute right-2 top-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-full text-sm px-5 py-1.5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
              >
                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Track'}
              </button>
            </div>
            
            {/* Recent Searches and Sample AWBs */}
            <div className="flex flex-wrap items-center gap-2 text-sm mt-4">
              {recentSearches.length > 0 && (
                <>
                  <span className="text-gray-500 dark:text-gray-400 font-medium">Recent:</span>
                  {recentSearches.map((search, index) => (
                    <button 
                      key={`recent-${index}`}
                      type="button"
                      onClick={() => loadTrackingID(search)}
                      className="px-3 py-1 bg-white hover:bg-gray-100 dark:bg-gray-800 dark:hover:bg-gray-700 rounded-full text-sm border border-gray-200 dark:border-gray-700 shadow-sm transition-colors"
                    >
                      {search}
                    </button>
                  ))}
                </>
              )}
              
              {sampleAWBs.length > 0 && (
                <>
                  <span className="text-gray-500 dark:text-gray-400 font-medium ml-auto">Sample AWBs:</span>
                  {sampleAWBs.map((awb, index) => (
                    <button 
                      key={`sample-${index}`}
                      type="button"
                      onClick={() => loadTrackingID(awb)}
                      className="px-3 py-1 bg-blue-50 hover:bg-blue-100 dark:bg-blue-900/20 dark:hover:bg-blue-800/30 rounded-full text-sm text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-800/50 shadow-sm transition-colors"
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
        <div className="flex flex-col items-center justify-center py-16 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 shadow-sm">
          <div className="relative w-16 h-16 mb-4">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-12 h-12 border-4 border-blue-200 dark:border-blue-900 rounded-full"></div>
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-12 h-12 border-4 border-transparent border-t-blue-600 dark:border-t-blue-500 rounded-full animate-spin"></div>
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <Package className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
          <span className="text-lg font-medium text-gray-900 dark:text-white">Tracking shipment...</span>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-2">Please wait while we retrieve the latest information</p>
        </div>
      )}
      
      {/* Error State */}
      {!isLoading && errorMessage && (
        <div className="py-16 text-center bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 shadow-sm">
          <div className="mx-auto w-16 h-16 mb-4 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
            <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-500" />
          </div>
          <h3 className="text-lg font-medium mb-2 text-gray-900 dark:text-white">Error</h3>
          <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            {errorMessage}
          </p>
          <button
            onClick={() => setErrorMessage('')}
            className="mt-6 px-4 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded-md text-sm font-medium transition-colors"
          >
            Back to Search
          </button>
        </div>
      )}
      
      {/* Shipment Details */}
      {!isLoading && !errorMessage && shipmentDetails && (
        <div className="space-y-6">
          {/* Overview Card */}
          <Card className="border-t-4 overflow-hidden" style={{ borderTopColor: shipmentDetails.status === 'Delivered' ? '#10b981' : 
                                                                         shipmentDetails.status === 'In Transit' ? '#3b82f6' :
                                                                         shipmentDetails.status === 'Customs Clearance' ? '#f59e0b' :
                                                                         shipmentDetails.status === 'Exception' ? '#ef4444' : '#8b5cf6' }}>
            <CardHeader className="pb-0">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <div className="flex items-center space-x-2">
                    <CardTitle className="text-xl font-bold">Shipment {shipmentDetails.id}</CardTitle>
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusBgColor(shipmentDetails.status)} ${getStatusColor(shipmentDetails.status)}`}>
                      {shipmentDetails.status}
                    </span>
                  </div>
                  <CardDescription className="mt-1">
                    From {shipmentDetails.origin?.location || 'Unknown'} to {shipmentDetails.destination?.location || 'Unknown'}
                  </CardDescription>
                </div>
                
                <div className="flex items-center space-x-3">
                  <button 
                    onClick={handleRefresh} 
                    className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                    title="Refresh tracking data"
                  >
                    <RefreshCw className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  </button>
                  <button className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" title="Copy tracking number">
                    <Clipboard className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  </button>
                  <button className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" title="Share">
                    <TrendingUp className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  </button>
                </div>
              </div>
              
              {getDeliveryTimeRemaining() && (
                <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="flex items-center">
                    <Clock className="w-5 h-5 text-blue-600 dark:text-blue-400 mr-2" />
                    <span className="text-sm font-medium text-blue-800 dark:text-blue-300">
                      Estimated delivery in {getDeliveryTimeRemaining()}
                    </span>
                  </div>
                </div>
              )}
            </CardHeader>
            
            <CardContent className="pt-6">
              <div className="flex flex-wrap -mx-2">
                <div className="w-full md:w-1/3 px-2 mb-4 md:mb-0">
                  <div className="p-4 h-full bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-100 dark:border-gray-800">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Origin</h3>
                    <p className="font-medium text-gray-900 dark:text-white">{shipmentDetails.origin?.location || 'Unknown'}</p>
                    <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">{shipmentDetails.origin?.facility || 'Unknown'}</p>
                    <div className="flex items-center mt-3 text-sm">
                      <Clock className="w-4 h-4 text-gray-400 mr-1.5" />
                      <span className="text-gray-500 dark:text-gray-400">
                        Departure: {formatDate(shipmentDetails.origin?.departureTime)}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="w-full md:w-1/3 px-2 mb-4 md:mb-0">
                  <div className="p-4 h-full bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-100 dark:border-gray-800">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Current Location</h3>
                    <p className="font-medium text-gray-900 dark:text-white">{shipmentDetails.currentLocation?.location || 'Unknown'}</p>
                    <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">{shipmentDetails.currentLocation?.status || 'Unknown'}</p>
                    <div className="flex items-center mt-3 text-sm">
                      <Clock className="w-4 h-4 text-gray-400 mr-1.5" />
                      <span className="text-gray-500 dark:text-gray-400">
                        Updated: {formatDate(shipmentDetails.currentLocation?.timestamp)}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="w-full md:w-1/3 px-2">
                  <div className="p-4 h-full bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-100 dark:border-gray-800">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Destination</h3>
                    <p className="font-medium text-gray-900 dark:text-white">{shipmentDetails.destination?.location || 'Unknown'}</p>
                    <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">{shipmentDetails.destination?.facility || 'Unknown'}</p>
                    <div className="flex items-center mt-3 text-sm">
                      <Clock className="w-4 h-4 text-gray-400 mr-1.5" />
                      <span className="text-gray-500 dark:text-gray-400">
                        Estimated arrival: {formatDate(shipmentDetails.destination?.estimatedArrival)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-100 dark:border-gray-800">
                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">Shipment Details</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Weight</p>
                    <p className="font-medium text-gray-900 dark:text-white">{shipmentDetails.details?.weight || 'Unknown'}</p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Packages</p>
                    <p className="font-medium text-gray-900 dark:text-white">{shipmentDetails.details?.packages || 'Unknown'}</p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Type</p>
                    <p className="font-medium text-gray-900 dark:text-white">{shipmentDetails.details?.type || 'Unknown'}</p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Service</p>
                    <p className="font-medium text-gray-900 dark:text-white">{shipmentDetails.details?.service || 'Unknown'}</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Tabs for Timeline and Updates */}
          <div className="border-b border-gray-200 dark:border-gray-800">
            <div className="flex space-x-8">
              <button
                onClick={() => setActiveTab('timeline')}
                className={`py-3 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'timeline'
                    ? 'border-blue-600 text-blue-600 dark:border-blue-500 dark:text-blue-500'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                Shipping Timeline
              </button>
              
              <button
                onClick={() => setActiveTab('updates')}
                className={`py-3 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'updates'
                    ? 'border-blue-600 text-blue-600 dark:border-blue-500 dark:text-blue-500'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                Updates & Notifications
                {shipmentDetails.updates && shipmentDetails.updates.length > 0 && (
                  <span className="ml-2 px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-xs">
                    {shipmentDetails.updates.length}
                  </span>
                )}
              </button>
            </div>
          </div>
          
          {/* Timeline Content */}
          {activeTab === 'timeline' && (
            <Card className="border border-gray-200 dark:border-gray-800 shadow-sm">
              <CardContent className="p-6">
                <div className="relative">
                  {/* Timeline line */}
                  <div className="absolute left-4 top-0 h-full w-0.5 bg-gray-200 dark:bg-gray-700"></div>
                  
                  <div className="space-y-8">
                    {shipmentDetails.timeline && shipmentDetails.timeline.map((item, index) => (
                      <div key={index} className="relative flex items-start group">
                        <div className={`absolute left-0 rounded-full mt-1 w-8 h-8 flex items-center justify-center ${
                          item.isCompleted 
                            ? 'bg-green-100 dark:bg-green-900/20 shadow-sm' 
                            : 'bg-gray-100 dark:bg-gray-800 shadow-sm'
                        } transition-colors`}>
                          {item.isCompleted ? (
                            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-500" />
                          ) : (
                            <Clock className="w-5 h-5 text-gray-400 dark:text-gray-600" />
                          )}
                        </div>
                        
                        <div className="ml-12 bg-white dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800 shadow-sm group-hover:border-gray-300 dark:group-hover:border-gray-700 transition-colors w-full">
                          <h3 className={`text-base font-medium ${
                            item.isCompleted 
                              ? 'text-gray-900 dark:text-white' 
                              : 'text-gray-500 dark:text-gray-400'
                          }`}>
                            {item.status}
                          </h3>
                          
                          <div className="flex items-center mt-2">
                            <MapPin className="w-4 h-4 text-gray-500 dark:text-gray-400 mr-1.5" />
                            <span className="text-sm text-gray-600 dark:text-gray-300">
                              {item.location}
                            </span>
                          </div>
                          
                          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                            {item.timestamp ? formatDate(item.timestamp) : 'Scheduled'}
                          </p>
                          
                          {item.details && (
                            <p className="mt-2 text-sm text-gray-600 dark:text-gray-300 border-t border-gray-100 dark:border-gray-800 pt-2">
                              {item.details}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Updates Content */}
          {activeTab === 'updates' && (
            <Card className="border border-gray-200 dark:border-gray-800 shadow-sm">
              <CardContent className="p-6">
                {shipmentDetails.updates && shipmentDetails.updates.length > 0 ? (
                  <div className="space-y-4">
                    {shipmentDetails.updates.map((update, index) => (
                      <div 
                        key={index} 
                        className={`p-4 rounded-lg border ${
                          update.type === 'warning' 
                            ? 'bg-amber-50 border-amber-200 dark:bg-amber-900/10 dark:border-amber-800/30' 
                            : update.type === 'error'
                            ? 'bg-red-50 border-red-200 dark:bg-red-900/10 dark:border-red-800/30'
                            : update.type === 'success'
                            ? 'bg-green-50 border-green-200 dark:bg-green-900/10 dark:border-green-800/30'
                            : 'bg-blue-50 border-blue-200 dark:bg-blue-900/10 dark:border-blue-800/30'
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
                ) : (
                  <div className="text-center py-10">
                    <div className="mx-auto w-16 h-16 mb-4 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center">
                      <Info className="w-8 h-8 text-gray-500 dark:text-gray-400" />
                    </div>
                    <h3 className="text-lg font-medium mb-2 text-gray-900 dark:text-white">No updates yet</h3>
                    <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
                      There are no updates or notifications for this shipment at the moment.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
          
          {/* Shipment Progress Bar */}
          <Card className="border border-gray-200 dark:border-gray-800 shadow-sm overflow-hidden">
            <CardContent className="p-0">
              <div className="bg-gray-50 dark:bg-gray-800/50 p-4">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Shipment Progress</h3>
                
                <div className="mt-4 relative">
                  <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    {(() => {
                      // Calculate progress based on timeline
                      let progress = 0;
                      if (shipmentDetails.timeline) {
                        const completedSteps = shipmentDetails.timeline.filter(item => item.isCompleted).length;
                        progress = (completedSteps / shipmentDetails.timeline.length) * 100;
                      }
                      return (
                        <div 
                          className="h-full bg-blue-600 dark:bg-blue-500 rounded-full transition-all duration-500 ease-out"
                          style={{ width: `${progress}%` }}
                        ></div>
                      );
                    })()}
                  </div>
                  
                  <div className="mt-4 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span>Order Placed</span>
                    <span>Processing</span>
                    <span>In Transit</span>
                    <span>Delivered</span>
                  </div>
                </div>
              </div>
              
              <div className="border-t border-gray-200 dark:border-gray-800 p-4">
                <div className="flex flex-wrap gap-3">
                  <button className="inline-flex items-center bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md text-sm px-4 py-2.5 transition-colors shadow-sm">
                    <Download className="w-4 h-4 mr-2" />
                    Download POD
                  </button>
                  <button className="inline-flex items-center bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-white font-medium rounded-md text-sm px-4 py-2.5 border border-gray-200 dark:border-gray-700 transition-colors shadow-sm">
                    <Truck className="w-4 h-4 mr-2" />
                    Delivery Options
                  </button>
                  <button className="inline-flex items-center bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-white font-medium rounded-md text-sm px-4 py-2.5 border border-gray-200 dark:border-gray-700 transition-colors shadow-sm">
                    <AlertTriangle className="w-4 h-4 mr-2" />
                    Report Issue
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Help and Support */}
          <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border-none shadow-sm">
            <CardContent className="p-6">
              <div className="flex flex-col md:flex-row items-center">
                <div className="mb-4 md:mb-0 md:mr-6">
                  <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                    <Globe className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                  </div>
                </div>
                <div className="text-center md:text-left md:flex-1">
                  <h3 className="text-lg font-medium mb-2 text-gray-900 dark:text-white">Need Help With Your Shipment?</h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    Our support team is available 24/7 to assist you with any questions about your shipment.
                  </p>
                  <div className="flex flex-wrap justify-center md:justify-start gap-3">
                    <button className="inline-flex items-center bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md text-sm px-4 py-2 transition-colors shadow-sm">
                      Contact Support
                    </button>
                    <button className="inline-flex items-center bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-white font-medium rounded-md text-sm px-4 py-2 border border-gray-200 dark:border-gray-700 transition-colors shadow-sm">
                      FAQs
                    </button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      
      {/* Empty State */}
      {!isLoading && !errorMessage && !shipmentDetails && !searchQuery && (
        <div className="py-16 text-center bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 shadow-sm">
          <div className="mx-auto w-20 h-20 mb-6 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
            <Globe className="w-10 h-10 text-blue-600 dark:text-blue-400" />
          </div>
          <h3 className="text-xl font-medium mb-3 text-gray-900 dark:text-white">Track your shipment</h3>
          <p className="text-gray-600 dark:text-gray-300 max-w-md mx-auto mb-6">
            Enter your tracking number, order ID, or reference number above to track your shipment's status and location.
          </p>
          
          <div className="flex flex-wrap justify-center gap-4 mt-8">
            <button className="inline-flex items-center bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md text-sm px-5 py-2.5 transition-colors shadow-sm">
              <Package className="w-4 h-4 mr-2" />
              Create New Shipment
            </button>
            <button className="inline-flex items-center bg-white hover:bg-gray-100 text-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700 dark:text-white font-medium rounded-md text-sm px-5 py-2.5 border border-gray-200 dark:border-gray-700 transition-colors shadow-sm">
              <Truck className="w-4 h-4 mr-2" />
              View My Shipments
            </button>
          </div>
          
          <div className="max-w-3xl mx-auto mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-5 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div className="text-blue-600 dark:text-blue-400 mb-3">
                <Search className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-medium mb-2 text-gray-900 dark:text-white">Track by Number</h4>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Enter your AWB number or reference ID to see real-time status updates.
              </p>
            </div>
            
            <div className="p-5 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div className="text-blue-600 dark:text-blue-400 mb-3">
                <Clock className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-medium mb-2 text-gray-900 dark:text-white">Delivery Estimates</h4>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Get accurate delivery timeframes and notifications for your shipments.
              </p>
            </div>
            
            <div className="p-5 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div className="text-blue-600 dark:text-blue-400 mb-3">
                <MapPin className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-medium mb-2 text-gray-900 dark:text-white">Location Updates</h4>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Follow your package's journey with detailed location tracking.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}