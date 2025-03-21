// src/app/dashboard/page.jsx
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
    ArrowRight 
  } from 'lucide-react';
  
  export default function DashboardPage() {
    // In a real implementation, this data would be fetched from the API
    const metricsData = {
      activeShipments: 1243,
      activeTrend: +12.5,
      totalWeight: "48,392",
      weightTrend: +8.3,
      avgDeliveryTime: 4.7,
      deliveryTimeTrend: -0.3,
      totalRevenue: "₹ 24.7M",
      revenueTrend: +15.2,
      // Mock data for chart
      recentShipments: [
        { id: 'AWB10983762', destination: 'New York, USA', status: 'In Transit', weight: '342 kg', value: '₹ 284,500' },
        { id: 'AWB10983571', destination: 'Dubai, UAE', status: 'Delivered', weight: '128 kg', value: '₹ 95,200' },
        { id: 'AWB10983445', destination: 'Singapore', status: 'Customs Clearance', weight: '205 kg', value: '₹ 173,800' },
        { id: 'AWB10983390', destination: 'London, UK', status: 'Processing', weight: '178 kg', value: '₹ 152,600' },
        { id: 'AWB10983255', destination: 'Tokyo, Japan', status: 'Delivered', weight: '93 kg', value: '₹ 87,400' },
      ],
      alerts: [
        { id: 1, type: 'warning', message: 'Potential delay on AWB10983762 due to customs inspection' },
        { id: 2, type: 'success', message: 'AWB10983255 delivered ahead of schedule' },
        { id: 3, type: 'error', message: 'Missing documentation for AWB10983390, requires immediate action' },
      ]
    };
  
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">Last updated: March 15, 2025, 09:45 AM IST</span>
            <button className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800">
              <svg 
                width="20" 
                height="20" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              >
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                <path d="M21 3v5h-5" />
                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                <path d="M8 16H3v5" />
              </svg>
            </button>
          </div>
        </div>
  
        {/* Key Performance Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Active Shipments
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">{metricsData.activeShipments}</div>
                  <div className="flex items-center mt-1">
                    {metricsData.activeTrend > 0 ? (
                      <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
                    )}
                    <span className={metricsData.activeTrend > 0 ? "text-green-500" : "text-red-500"}>
                      {Math.abs(metricsData.activeTrend)}%
                    </span>
                    <span className="text-gray-500 dark:text-gray-400 text-xs ml-1">vs last month</span>
                  </div>
                </div>
                <div className="h-12 w-12 bg-blue-50 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
                  <Truck className="w-6 h-6 text-blue-600 dark:text-blue-500" />
                </div>
              </div>
            </CardContent>
          </Card>
  
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Total Cargo Weight
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">{metricsData.totalWeight} kg</div>
                  <div className="flex items-center mt-1">
                    {metricsData.weightTrend > 0 ? (
                      <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
                    )}
                    <span className={metricsData.weightTrend > 0 ? "text-green-500" : "text-red-500"}>
                      {Math.abs(metricsData.weightTrend)}%
                    </span>
                    <span className="text-gray-500 dark:text-gray-400 text-xs ml-1">vs last month</span>
                  </div>
                </div>
                <div className="h-12 w-12 bg-green-50 dark:bg-green-900/20 rounded-full flex items-center justify-center">
                  <Package className="w-6 h-6 text-green-600 dark:text-green-500" />
                </div>
              </div>
            </CardContent>
          </Card>
  
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Avg. Delivery Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">{metricsData.avgDeliveryTime} days</div>
                  <div className="flex items-center mt-1">
                    {metricsData.deliveryTimeTrend < 0 ? (
                      <TrendingDown className="w-4 h-4 text-green-500 mr-1" />
                    ) : (
                      <TrendingUp className="w-4 h-4 text-red-500 mr-1" />
                    )}
                    <span className={metricsData.deliveryTimeTrend < 0 ? "text-green-500" : "text-red-500"}>
                      {Math.abs(metricsData.deliveryTimeTrend)} days
                    </span>
                    <span className="text-gray-500 dark:text-gray-400 text-xs ml-1">vs last month</span>
                  </div>
                </div>
                <div className="h-12 w-12 bg-purple-50 dark:bg-purple-900/20 rounded-full flex items-center justify-center">
                  <svg 
                    className="w-6 h-6 text-purple-600 dark:text-purple-500" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                </div>
              </div>
            </CardContent>
          </Card>
  
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Total Revenue
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">{metricsData.totalRevenue}</div>
                  <div className="flex items-center mt-1">
                    {metricsData.revenueTrend > 0 ? (
                      <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
                    )}
                    <span className={metricsData.revenueTrend > 0 ? "text-green-500" : "text-red-500"}>
                      {Math.abs(metricsData.revenueTrend)}%
                    </span>
                    <span className="text-gray-500 dark:text-gray-400 text-xs ml-1">vs last month</span>
                  </div>
                </div>
                <div className="h-12 w-12 bg-amber-50 dark:bg-amber-900/20 rounded-full flex items-center justify-center">
                  <svg 
                    className="w-6 h-6 text-amber-600 dark:text-amber-500" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                  >
                    <line x1="12" y1="1" x2="12" y2="23" />
                    <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                  </svg>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
  
        {/* Alerts and Recent Shipments */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card className="h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-medium">Recent Shipments</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b dark:border-gray-700">
                        <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">AWB</th>
                        <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Destination</th>
                        <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Status</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Weight</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metricsData.recentShipments.map((shipment, index) => (
                        <tr 
                          key={shipment.id} 
                          className={`hover:bg-gray-50 dark:hover:bg-gray-800 ${
                            index !== metricsData.recentShipments.length - 1 ? 'border-b dark:border-gray-700' : ''
                          }`}
                        >
                          <td className="py-3 px-4 font-medium">{shipment.id}</td>
                          <td className="py-3 px-4">{shipment.destination}</td>
                          <td className="py-3 px-4">
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
                          </td>
                          <td className="py-3 px-4 text-right">{shipment.weight}</td>
                          <td className="py-3 px-4 text-right">{shipment.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-4 text-right">
                  <a href="/shipments" className="text-sm font-medium text-blue-600 dark:text-blue-500 hover:underline inline-flex items-center">
                    View all shipments <ArrowRight className="w-4 h-4 ml-1" />
                  </a>
                </div>
              </CardContent>
            </Card>
          </div>
  
          <Card className="h-full">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-medium">Alerts & Notifications</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {metricsData.alerts.map((alert) => (
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
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {alert.type === 'warning' ? 'Warning' : alert.type === 'success' ? 'Success' : 'Error'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 text-center pt-4 border-t dark:border-gray-700">
                <a href="/alerts" className="text-sm font-medium text-blue-600 dark:text-blue-500 hover:underline inline-flex items-center">
                  View all alerts <ArrowRight className="w-4 h-4 ml-1" />
                </a>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }