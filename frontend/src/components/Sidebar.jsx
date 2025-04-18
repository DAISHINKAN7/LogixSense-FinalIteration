'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  LayoutDashboard, 
  BarChart2, 
  TrendingUp, 
  MessageSquare, 
  Settings, 
  Menu, 
  X, 
  Truck,
  Globe,
  AlertCircle
} from 'lucide-react';

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const pathname = usePathname();

  const toggleSidebar = () => {
    const newState = !isCollapsed;
    setIsCollapsed(newState);
    // Inform the layout of the change
    localStorage.setItem('sidebarState', newState ? 'collapsed' : 'expanded');
    window.dispatchEvent(new Event('storage'));
  };

  const toggleMobileSidebar = () => setIsMobileOpen(!isMobileOpen);

  const navItems = [
    {
      title: 'Dashboard',
      icon: <LayoutDashboard size={20} />,
      href: '/dashboard',
    },
    {
      title: 'Analytics',
      icon: <BarChart2 size={20} />,
      href: '/analytics',
    },
    {
      title: 'Forecasting',
      icon: <TrendingUp size={20} />,
      href: '/forecasting',
    },
    {
      title: 'Global Shipping',
      icon: <Globe size={20} />,
      href: '/global-shipping',
    },
    {
      title: 'Shipment Tracking',
      icon: <Truck size={20} />,
      href: '/shipment-tracking',
    },
    {
      title: 'Risk Analysis',
      icon: <AlertCircle size={20} />,
      href: '/risk-analysis',
    },
    {
      title: 'AI Assistant',
      icon: <MessageSquare size={20} />,
      href: '/assistant',
    },
    {
      title: 'Settings',
      icon: <Settings size={20} />,
      href: '/settings',
    },
  ];

  // Desktop sidebar - fixed to side with no gap
  const sidebarClass = `
    fixed top-0 left-0 z-40 h-screen
    bg-indigo-800 dark:bg-gray-800
    transition-all duration-300 ease-in-out
    ${isCollapsed ? 'w-16' : 'w-64'} 
    hidden md:block
  `;

  // Mobile sidebar - fully hidden when collapsed
  const mobileSidebarClass = `
    fixed top-0 left-0 z-50 h-screen
    bg-indigo-800 dark:bg-gray-800
    w-64
    transform ${isMobileOpen ? 'translate-x-0' : '-translate-x-full'}
    transition-transform duration-300 ease-in-out
    md:hidden
  `;

  const renderNavItems = () => (
    <ul className="space-y-2 mt-6">
      {navItems.map((item) => {
        const isActive = pathname === item.href;
        return (
          <li key={item.href}>
            <Link 
              href={item.href}
              className={`
                flex items-center p-2 mx-3 rounded-lg
                ${isActive 
                  ? 'bg-indigo-700 text-white dark:bg-gray-700' 
                  : 'text-white hover:bg-indigo-700 dark:hover:bg-gray-700'}
                ${isCollapsed ? 'justify-center' : ''}
              `}
              onClick={() => {
                if (window.innerWidth < 768) {
                  toggleMobileSidebar();
                }
              }}
            >
              <div>
                {item.icon}
              </div>
              
              {!isCollapsed && <span className="ml-3">{item.title}</span>}
            </Link>
          </li>
        );
      })}
    </ul>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <aside className={sidebarClass}>
        <div className="h-full py-4 overflow-y-auto">
          <div className="flex items-center justify-between px-3 mb-6">
            {!isCollapsed && (
              <Link href="/dashboard" className="flex items-center space-x-3">
                <span className="h-8 w-8 bg-white rounded-full flex items-center justify-center">
                  <span className="text-indigo-800 font-bold">LS</span>
                </span>
                <span className="self-center text-xl font-semibold whitespace-nowrap text-white">LogixSense</span>
              </Link>
            )}
            {isCollapsed && (
              <Link href="/dashboard" className="flex items-center justify-center w-full">
                <span className="h-8 w-8 bg-white rounded-full flex items-center justify-center">
                  <span className="text-indigo-800 font-bold">LS</span>
                </span>
              </Link>
            )}
            {!isCollapsed && (
              <button
                type="button"
                onClick={toggleSidebar}
                className="inline-flex items-center p-1 rounded-lg text-gray-200 hover:bg-indigo-700"
              >
                <X size={18} />
              </button>
            )}
          </div>
          
          {renderNavItems()}
          
          {/* Expand button at bottom when collapsed */}
          {isCollapsed && (
            <div className="absolute bottom-4 w-full flex justify-center">
              <button
                type="button"
                onClick={toggleSidebar}
                className="p-2 rounded-lg text-gray-200 hover:bg-indigo-700"
              >
                <Menu size={20} />
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* Mobile Menu Button */}
      <div className="fixed top-4 left-4 z-40 md:hidden">
        <button
          type="button"
          onClick={toggleMobileSidebar}
          aria-label="Toggle navigation menu"
          className="p-2 rounded-lg bg-white dark:bg-gray-800 shadow-md"
        >
          <Menu size={24} className="text-indigo-800 dark:text-gray-200" />
        </button>
      </div>
      
      {/* Mobile Sidebar */}
      <aside className={mobileSidebarClass}>
        <div className="h-full py-4 overflow-y-auto">
          <div className="flex items-center justify-between px-3 mb-6">
            <Link href="/dashboard" className="flex items-center space-x-3">
              <span className="h-8 w-8 bg-white rounded-full flex items-center justify-center">
                <span className="text-indigo-800 font-bold">LS</span>
              </span>
              <span className="self-center text-xl font-semibold whitespace-nowrap text-white">LogixSense</span>
            </Link>
            <button
              type="button"
              onClick={toggleMobileSidebar}
              className="inline-flex items-center p-1 rounded-lg text-gray-200 hover:bg-indigo-700"
            >
              <X size={18} />
            </button>
          </div>
          
          {renderNavItems()}
        </div>
      </aside>

      {/* Overlay when mobile sidebar is open */}
      {isMobileOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={toggleMobileSidebar}
          aria-hidden="true"
        />
      )}
    </>
  );
}