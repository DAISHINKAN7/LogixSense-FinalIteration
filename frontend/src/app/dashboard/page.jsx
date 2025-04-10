// src/app/dashboard/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  Truck, 
  Package, 
  ArrowRight,
  RefreshCw,
  Clock,
  DollarSign
} from 'lucide-react';
import DestinationChart from '@/components/DestinationChart';
import WeightDistributionChart from '@/components/WeightDistributionChart';
import MonthlyTrendsChart from '@/components/MonthlyTrendsChart';
import CarrierDistributionChart from '@/components/CarrierDistributionChart';

export default function DashboardPage() {
  // State for API data
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Function to handle data refresh
  const refreshData = async () => {
    if (refreshing) return;
    setRefreshing(true);
    try {
      await fetchDashboardData();
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Error refreshing data:", error);
    } finally {
      setRefreshing(false);
    }
  };

  // Function to fetch all dashboard data in a single call
  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch all dashboard metrics in a single API call
      const response = await fetch('/api/dashboard/metrics');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch dashboard data: ${response.status}`);
      }
      
      const data = await response.json();
      setDashboardData(data);
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
      setError("Failed to load dashboard data. Please try again later.");
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch data on component mount
  useEffect(() => {
    fetchDashboardData();
  }, []);

  // Format last updated timestamp
  const formattedLastUpdated = lastUpdated.toLocaleString('en-IN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true
  });
  
  // If loading, show loading state
  if (loading && !dashboardData) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen p-4">
        <div className="animate-spin mb-4">
          <RefreshCw size={32} className="text-gray-400" />
        </div>
        <h2 className="text-xl font-semibold mb-2">Loading Dashboard</h2>
        <p className="text-gray-500">Please wait while we fetch your logistics data...</p>
      </div>
    );
  }

  // If error, show error state
  if (error && !dashboardData) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen p-4">
        <AlertTriangle size={32} className="text-red-500 mb-4" />
        <h2 className="text-xl font-semibold mb-2">Error Loading Dashboard</h2>
        <p className="text-red-500 mb-4">{error}</p>
        <button 
          onClick={refreshData} 
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
          disabled={refreshing}
        >
          {refreshing ? (
            <>
              <RefreshCw size={16} className="mr-2 animate-spin" />
              Trying Again...
            </>
          ) : (
            <>
              <RefreshCw size={16} className="mr-2" />
              Try Again
            </>
          )}
        </button>
      </div>
    );
  }

  // Extract data from the dashboard data
  const summaryData = dashboardData?.summary || {};
  const recentShipments = dashboardData?.recentShipments || [];
  const alerts = dashboardData?.alerts || [];

  // Render dashboard with fetched data
  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold tracking-tight">Logistics Dashboard</h1>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">
            Last updated: {formattedLastUpdated}
          </span>
          <button 
            className={`p-1 rounded-md hover:bg-gray-100 ${
              refreshing ? 'animate-spin text-blue-500' : ''
            }`}
            onClick={refreshData}
            disabled={refreshing}
            aria-label="Refresh dashboard data"
          >
            <RefreshCw size={20} />
          </button>
        </div>
      </div>

      {/* Key Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-white-500">
              Active Shipments
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{summaryData?.activeShipments?.toLocaleString() || "0"}</div>
                <div className="flex items-center mt-1">
                  {summaryData?.trends?.activeShipmentsTrend > 0 ? (
                    <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
                  )}
                  <span className={summaryData?.trends?.activeShipmentsTrend > 0 ? "text-green-500" : "text-red-500"}>
                    {Math.abs(summaryData?.trends?.activeShipmentsTrend || 0)}%
                  </span>
                  <span className="text-gray-500 text-xs ml-1">vs last month</span>
                </div>
              </div>
              <div className="h-12 w-12 bg-blue-50 rounded-full flex items-center justify-center">
                <Truck className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500">
              Total Cargo Weight
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{summaryData?.totalWeight?.toLocaleString() || "0"} kg</div>
                <div className="flex items-center mt-1">
                  {summaryData?.trends?.weightTrend > 0 ? (
                    <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
                  )}
                  <span className={summaryData?.trends?.weightTrend > 0 ? "text-green-500" : "text-red-500"}>
                    {Math.abs(summaryData?.trends?.weightTrend || 0)}%
                  </span>
                  <span className="text-gray-500 text-xs ml-1">vs last month</span>
                </div>
              </div>
              <div className="h-12 w-12 bg-green-50 rounded-full flex items-center justify-center">
                <Package className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500">
              Avg. Delivery Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{summaryData?.avgDeliveryTime || "0"} days</div>
                <div className="flex items-center mt-1">
                  {summaryData?.trends?.deliveryTimeTrend < 0 ? (
                    <TrendingDown className="w-4 h-4 text-green-500 mr-1" />
                  ) : (
                    <TrendingUp className="w-4 h-4 text-red-500 mr-1" />
                  )}
                  <span className={summaryData?.trends?.deliveryTimeTrend < 0 ? "text-green-500" : "text-red-500"}>
                    {Math.abs(summaryData?.trends?.deliveryTimeTrend || 0)} days
                  </span>
                  <span className="text-gray-500 text-xs ml-1">vs last month</span>
                </div>
              </div>
              <div className="h-12 w-12 bg-purple-50 rounded-full flex items-center justify-center">
                <Clock className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500">
              Total Revenue
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{summaryData?.totalRevenue || "₹ 0"}</div>
                <div className="flex items-center mt-1">
                  {summaryData?.trends?.revenueTrend > 0 ? (
                    <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
                  )}
                  <span className={summaryData?.trends?.revenueTrend > 0 ? "text-green-500" : "text-red-500"}>
                    {Math.abs(summaryData?.trends?.revenueTrend || 0)}%
                  </span>
                  <span className="text-gray-500 text-xs ml-1">vs last month</span>
                </div>
              </div>
              <div className="h-12 w-12 bg-amber-50 rounded-full flex items-center justify-center">
                <DollarSign className="w-6 h-6 text-amber-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-[420px]">
          <DestinationChart data={dashboardData?.destinations} />
        </div>
        <div className="h-[420px]">
          <WeightDistributionChart data={dashboardData?.weightDistribution} />
        </div>
      </div>

      {/* Alerts and Recent Shipments */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card className="h-full">
            <CardHeader className="pb-2 border-b">
              <CardTitle className="text-lg font-medium">
                Recent Shipments
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loading && !recentShipments.length ? (
                <div className="flex justify-center p-8">
                  <RefreshCw size={24} className="animate-spin text-gray-400" />
                </div>
              ) : recentShipments.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-3 px-4 font-medium text-gray-500">AWB</th>
                        <th className="text-left py-3 px-4 font-medium text-gray-500">Destination</th>
                        <th className="text-left py-3 px-4 font-medium text-gray-500">Status</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-500">Weight</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-500">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentShipments.map((shipment, index) => (
                        <tr 
                          key={shipment.id} 
                          className={`hover:bg-gray-50 ${
                            index !== recentShipments.length - 1 ? 'border-b' : ''
                          }`}
                        >
                          <td className="py-3 px-4 font-medium">{shipment.id}</td>
                          <td className="py-3 px-4">{shipment.destination}</td>
                          <td className="py-3 px-4">
                            <span className={`
                              inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                              ${shipment.status === 'Delivered' 
                                ? 'bg-green-100 text-green-800'
                                : shipment.status === 'In Transit'
                                ? 'bg-blue-100 text-blue-800'
                                : shipment.status === 'Customs Clearance'
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-purple-100 text-purple-800'
                              }
                            `}>
                              {shipment.status}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-right">{shipment.weight}</td>
                          <td className="py-3 px-4 text-right">{shipment.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center p-8 text-gray-500">
                  No recent shipments found
                </div>
              )}
              <div className="mt-4 text-right">
                <a href="/shipments" className="text-sm font-medium text-blue-600 hover:underline inline-flex items-center">
                  View all shipments <ArrowRight className="w-4 h-4 ml-1" />
                </a>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="h-full">
          <CardHeader className="pb-2 border-b">
            <CardTitle className="text-lg font-medium">
              Alerts & Notifications
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading && !alerts.length ? (
              <div className="flex justify-center p-8">
                <RefreshCw size={24} className="animate-spin text-gray-400" />
              </div>
            ) : alerts.length > 0 ? (
              <div className="space-y-4">
                {alerts.map((alert) => (
                  <div key={alert.id} className="flex items-start space-x-3">
                    {alert.type === 'warning' ? (
                      <AlertTriangle className="w-5 h-5 text-amber-500 mt-0.5" />
                    ) : alert.type === 'success' ? (
                      <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                    ) : (
                      <svg 
                        className="w-5 h-5 text-red-500 mt-0.5" 
                        viewBox="0 0 24 24" 
                        fill="none" 
                        stroke="currentColor" 
                        strokeWidth="2" 
                        strokeLinecap="round" 
                        strokeLinejoin="round"
                      >
                        <circle cx="12" cy="12" r="10" />
                        <line x1="12" y1="8" x2="12" y2="12" />
                        <line x1="12" y1="16" x2="12.01" y2="16" />
                      </svg>
                    )}
                    <div>
                      <p className="text-sm">{alert.message}</p>
                      <span className="text-xs text-gray-500">
                        {alert.type === 'warning' ? 'Warning' : alert.type === 'success' ? 'Success' : 'Error'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center p-8 text-gray-500">
                No alerts at this time
              </div>
            )}
            <div className="mt-4 text-center pt-4 border-t">
              <a href="/alerts" className="text-sm font-medium text-blue-600 hover:underline inline-flex items-center">
                View all alerts <ArrowRight className="w-4 h-4 ml-1" />
              </a>
            </div>
          </CardContent>
        </Card>
      </div>

{/* Additional Charts Row */}
<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-[400px]">
          <Card className="h-full">
            <CardHeader className="pb-2 border-b">
              <CardTitle className="text-lg font-medium">
                Monthly Shipment Trends
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              {loading ? (
                <div className="flex justify-center items-center h-64">
                  <RefreshCw className="h-8 w-8 text-gray-400 animate-spin" />
                </div>
              ) : (
                <div className="h-full">
                  <MonthlyTrendsChart data={dashboardData?.monthlyTrends} />
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        <div className="h-[400px]">
          <Card className="h-full">
            <CardHeader className="pb-2 border-b">
              <CardTitle className="text-lg font-medium">
                Shipments by Carrier
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              {loading ? (
                <div className="flex justify-center items-center h-64">
                  <RefreshCw className="h-8 w-8 text-gray-400 animate-spin" />
                </div>
              ) : (
                <div className="h-full">
                  <CarrierDistributionChart data={dashboardData?.carrierData} loading={loading} />
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}