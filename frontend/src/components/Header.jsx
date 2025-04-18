'use client';

import { useState, useEffect } from 'react';
import { useTheme } from 'next-themes';
import { 
  Search, 
  Bell, 
  Sun, 
  Moon, 
  User,
  ChevronDown,
  HelpCircle,
  MessageSquare,
  Settings as SettingsIcon,
  LogOut
} from 'lucide-react';
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu';

export default function Header() {
  const [mounted, setMounted] = useState(false);
  const { theme, setTheme } = useTheme();
  const [notifications, setNotifications] = useState(3); // Example notification count
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  
  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <header className="sticky top-0 z-30 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800 shadow-md backdrop-blur-sm">
      <div className="px-6 py-3 flex items-center justify-between">
        {/* Left Section - Brand for mobile, search for desktop */}
        <div className="md:hidden flex items-center">
          <div className="h-8 w-8 bg-gradient-to-tr from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center mr-2">
            <span className="text-white font-bold">LS</span>
          </div>
          <span className="font-bold text-gray-800 dark:text-white">LogixSense</span>
        </div>

        {/* Search Bar - Animated */}
        <div className={`relative hidden md:block transition-all duration-300 ease-in-out ${isSearchFocused ? 'w-96' : 'w-64'}`}>
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <Search className={`w-5 h-5 transition-colors duration-300 ${isSearchFocused ? 'text-blue-500 dark:text-blue-400' : 'text-gray-500 dark:text-gray-400'}`} />
          </div>
          <input 
            type="text" 
            className={`w-full py-2 pl-10 pr-4 text-sm transition-all duration-300 ease-in-out
            bg-white dark:bg-gray-800 border border-transparent dark:border-gray-700
            rounded-full focus:outline-none
            ${isSearchFocused 
              ? 'ring-2 ring-blue-400 dark:ring-blue-600 shadow-lg' 
              : 'hover:bg-gray-50 dark:hover:bg-gray-700'}`}
            placeholder="Search shipments, orders, analytics..." 
            onFocus={() => setIsSearchFocused(true)}
            onBlur={() => setIsSearchFocused(false)}
          />
        </div>
        
        {/* Right Side - Actions */}
        <div className="flex items-center space-x-1 md:space-x-4">
          {/* Mobile Search Trigger */}
          <button className="md:hidden p-2 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300">
            <Search className="w-5 h-5" />
          </button>
          
          {/* Dark/Light Mode Toggle */}
          {mounted && (
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-full bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 transition-colors duration-300"
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? (
                <Sun className="w-5 h-5 text-amber-400" />
              ) : (
                <Moon className="w-5 h-5 text-indigo-600" />
              )}
            </button>
          )}
          
          {/* Notifications */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="relative p-2 rounded-full bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 transition-colors duration-300">
                <Bell className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                {notifications > 0 && (
                  <span className="absolute top-0 right-0 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white transform translate-x-1/2 -translate-y-1/2 bg-red-500 rounded-full">
                    {notifications}
                  </span>
                )}
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-80 p-2">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-lg">Notifications</span>
                <button 
                  className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                  onClick={() => setNotifications(0)}
                >
                  Mark all as read
                </button>
              </div>
              
              {/* Notification Items */}
              <div className="space-y-2 max-h-72 overflow-y-auto">
                <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 cursor-pointer">
                  <div className="flex items-start">
                    <div className="flex-shrink-0 mr-3">
                      <div className="h-8 w-8 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                        <MessageSquare className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                      </div>
                    </div>
                    <div>
                      <p className="text-sm font-medium">New message from logistics</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Delivery schedule updated for order #38291</p>
                      <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">2 minutes ago</p>
                    </div>
                  </div>
                </div>
                
                <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 cursor-pointer">
                  <div className="flex items-start">
                    <div className="flex-shrink-0 mr-3">
                      <div className="h-8 w-8 bg-amber-100 dark:bg-amber-900 rounded-full flex items-center justify-center">
                        <Bell className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                      </div>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Shipment alert</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Potential delay detected for container MSCU3878291</p>
                      <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">1 hour ago</p>
                    </div>
                  </div>
                </div>
              </div>
              
              <DropdownMenuSeparator className="my-2" />
              <DropdownMenuItem className="cursor-pointer justify-center text-center text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300">
                View all notifications
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          
          {/* Help */}
          <button className="p-2 rounded-full bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 transition-colors duration-300">
            <HelpCircle className="w-5 h-5 text-gray-600 dark:text-gray-300" />
          </button>
          
          {/* User Profile Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center space-x-2 p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors duration-300 focus:outline-none">
                <div className="relative w-8 h-8 overflow-hidden bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center">
                  <User className="w-5 h-5 text-white" />
                </div>
                <ChevronDown className="w-4 h-4 text-gray-600 dark:text-gray-300 hidden md:block" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 p-2">
              <div className="px-2 py-2 border-b border-gray-100 dark:border-gray-800">
                <p className="font-medium">Kunal Ajgaonkar</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Logistics Manager</p>
              </div>
              
              <div className="py-2">
                <DropdownMenuItem className="cursor-pointer">
                  <User className="mr-2 h-4 w-4" />
                  <span>Profile</span>
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer">
                  <MessageSquare className="mr-2 h-4 w-4" />
                  <span>Messages</span>
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer">
                  <SettingsIcon className="mr-2 h-4 w-4" />
                  <span>Settings</span>
                </DropdownMenuItem>
              </div>
              
              <DropdownMenuSeparator />
              
              <DropdownMenuItem className="cursor-pointer text-red-600 hover:text-red-700 dark:text-red-500 dark:hover:text-red-400">
                <LogOut className="mr-2 h-4 w-4" />
                <span>Logout</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      
      {/* Mobile Search - Expandable */}
      <div className="px-4 pb-3 md:hidden">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <Search className="w-5 h-5 text-gray-500 dark:text-gray-400" />
          </div>
          <input 
            type="text" 
            className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 p-2" 
            placeholder="Search..." 
          />
        </div>
      </div>
    </header>
  );
}