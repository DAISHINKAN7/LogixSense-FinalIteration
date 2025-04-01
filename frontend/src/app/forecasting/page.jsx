// // src/app/forecasting/page.jsx
// 'use client';

// import { useState } from 'react';
// import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
// import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// import ForecastingDashboard from '@/components/ForecastingDashboard';
// import DetailedForecasting from '@/components/DetailedForecasting'; // Using a new filename for our detailed view

// export default function ForecastingPage() {
//   const [activeTab, setActiveTab] = useState('dashboard');

//   return (
//     <div className="space-y-6">
//       <div className="flex items-center justify-between">
//         <h1 className="text-2xl font-bold tracking-tight">LogixSense Forecasting</h1>
//       </div>

//       <Tabs defaultValue="dashboard" value={activeTab} onValueChange={setActiveTab} className="space-y-4">
//         <TabsList className="grid w-full grid-cols-2">
//           <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
//           <TabsTrigger value="detailed">Detailed Analysis</TabsTrigger>
//         </TabsList>
        
//         <TabsContent value="dashboard">
//           <ForecastingDashboard />
//         </TabsContent>
        
//         <TabsContent value="detailed">
//           <DetailedForecasting />
//         </TabsContent>
//       </Tabs>
//     </div>
//   );
// }

// src/app/forecasting/page.jsx
'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { motion } from 'framer-motion';
import { TrendingUp, Zap, Calendar, Filter, BarChart2 } from 'lucide-react';

import EnhancedDashboard from '@/components/EnhancedDashboard';
import DetailedForecasting from '@/components/DetailedForecasting';

export default function ForecastingPage() {
  const [activeTab, setActiveTab] = useState('enhanced');

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-3xl font-bold tracking-tight">LogixSense Forecasting</h1>
          <p className="text-muted-foreground">
            Advanced AI-driven predictions for logistics operations
          </p>
        </motion.div>

        <motion.div 
          className="flex items-center gap-2"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Badge variant="outline" className="px-3 py-1 text-xs bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
            <Zap className="h-3 w-3 mr-1" />
            AI-Powered
          </Badge>
          <Badge variant="outline" className="px-3 py-1 text-xs bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
            <Calendar className="h-3 w-3 mr-1" />
            6-Month Horizon
          </Badge>
        </motion.div>
      </div>

      <Tabs defaultValue="enhanced" value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <div className="flex justify-between items-center">
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="enhanced" className="flex items-center justify-center gap-2">
              <BarChart2 className="h-4 w-4" />
              <span>Enhanced Dashboard</span>
            </TabsTrigger>
            <TabsTrigger value="detailed" className="flex items-center justify-center gap-2">
              <Filter className="h-4 w-4" />
              <span>Detailed Analysis</span>
            </TabsTrigger>
          </TabsList>
        </div>
        
        <TabsContent value="enhanced" className="space-y-6">
          <EnhancedDashboard />
        </TabsContent>
        
        <TabsContent value="detailed" className="space-y-6">
          <DetailedForecasting />
        </TabsContent>
      </Tabs>
    </div>
  );
}