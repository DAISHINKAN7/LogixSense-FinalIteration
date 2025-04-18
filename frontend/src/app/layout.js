// frontend/src/app/layout.js
'use client';

import { useState, useEffect } from 'react';
import { Inter } from 'next/font/google';
import Sidebar from '../components/Sidebar';
import Header from '../components/Header';
import './globals.css';
import { ThemeProvider } from '@/components/theme-provider';
import { TooltipProvider } from '@/components/ui/tooltip';
import Head from 'next/head';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({ children }) {
  const [sidebarState, setSidebarState] = useState('expanded'); // default expanded
  
  // Listen for sidebar state changes
  useEffect(() => {
    const handleStorageChange = () => {
      const newState = localStorage.getItem('sidebarState') || 'expanded';
      setSidebarState(newState);
    };
    
    // Initialize
    handleStorageChange();
    
    // Listen for changes
    window.addEventListener('storage', handleStorageChange);
    
    // Set page title dynamically based on current route
    const pathSegments = window.location.pathname.split('/').filter(Boolean);
    if (pathSegments.length > 0) {
      const currentPage = pathSegments[0];
      const formattedPageName = currentPage.charAt(0).toUpperCase() + currentPage.slice(1).replace(/-/g, ' ');
      document.title = `${formattedPageName} | LogixSense`;
    } else {
      document.title = 'LogixSense - AI-Driven Logistics Analytics';
    }
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="manifest" href="/site.webmanifest" />
        <meta name="theme-color" content="#3b82f6" />
      </head>
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <TooltipProvider>
            <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
              <Sidebar />
              
              {/* Main Content - No white space gap */}
              <div 
                className={`transition-all duration-300 ${
                  sidebarState === 'collapsed' ? 'md:ml-16' : 'md:ml-64'
                }`}
              >
                <Header />
                <main className="p-4">
                  {children}
                </main>
              </div>
            </div>
          </TooltipProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}