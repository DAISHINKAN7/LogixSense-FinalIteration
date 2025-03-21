    // src/components/Sidebar.jsx
    'use client';

    import { useState } from 'react';
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
    LogOut,
    Truck,
    Globe,
    AlertCircle
    } from 'lucide-react';

    export default function Sidebar() {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [isMobileOpen, setIsMobileOpen] = useState(false);
    const pathname = usePathname();

    const toggleSidebar = () => setIsCollapsed(!isCollapsed);
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

    const sidebarClass = `
    fixed top-0 left-0 z-40 h-screen transition-all duration-300 
    bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700
    ${isCollapsed ? 'w-20' : 'w-64'} 
    hidden md:block
    `;

    // Change the mobile sidebar closing logic to ensure it's completely hidden when closed
    const mobileSidebarClass = `
    fixed top-0 left-0 z-50 h-screen transition-transform duration-300 ease-in-out
    bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700
    w-64 md:hidden
    ${isMobileOpen ? 'translate-x-0' : '-translate-x-full'}
    `;

    const renderNavItems = () => (
        <ul className="space-y-2 mt-5">
        {navItems.map((item) => (
            <li key={item.href}>
            <Link 
                href={item.href}
                className={`
                flex items-center p-3 text-base font-medium rounded-lg
                hover:bg-gray-100 dark:hover:bg-gray-700
                ${pathname === item.href ? 'bg-gray-100 dark:bg-gray-700 text-blue-600 dark:text-blue-500' : 'text-gray-900 dark:text-white'}
                ${isCollapsed ? 'justify-center' : ''}
                `}
            >
                {item.icon}
                {!isCollapsed && <span className="ml-3">{item.title}</span>}
            </Link>
            </li>
        ))}
        </ul>
    );

    return (
        <>
        {/* Desktop Sidebar */}
        <aside className={sidebarClass}>
            <div className="h-full px-3 py-4 overflow-y-auto">
            <div className="flex items-center justify-between">
                {!isCollapsed && (
                <Link href="/dashboard" className="flex items-center space-x-3">
                    <span className="h-8 w-8 bg-blue-600 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold">LS</span>
                    </span>
                    <span className="self-center text-xl font-semibold whitespace-nowrap dark:text-white">LogixSense</span>
                </Link>
                )}
                <button
                type="button"
                onClick={toggleSidebar}
                className="inline-flex items-center p-2 text-sm text-gray-500 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 dark:text-gray-400"
                >
                {isCollapsed ? <Menu size={20} /> : <X size={20} />}
                </button>
            </div>
            {renderNavItems()}
            </div>
        </aside>

        {/* Mobile Sidebar */}
        <div className="md:hidden">
            <button
            type="button"
            onClick={toggleMobileSidebar}
            className="inline-flex items-center p-2 ml-3 text-sm text-gray-500 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 dark:text-gray-400 focus:outline-none"
            >
            <Menu size={24} />
            </button>
        </div>
        
        <aside className={mobileSidebarClass}>
            <div className="h-full px-3 py-4 overflow-y-auto">
            <div className="flex items-center justify-between mb-5">
                <Link href="/dashboard" className="flex items-center space-x-3">
                <span className="h-8 w-8 bg-blue-600 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold">LS</span>
                </span>
                <span className="self-center text-xl font-semibold whitespace-nowrap dark:text-white">LogixSense</span>
                </Link>
                <button
                type="button"
                onClick={toggleMobileSidebar}
                className="inline-flex items-center p-2 text-sm text-gray-500 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 dark:text-gray-400"
                >
                <X size={20} />
                </button>
            </div>
            {renderNavItems()}
            </div>
        </aside>

        {/* Overlay */}
        {isMobileOpen && (
            <div 
            className="fixed inset-0 z-40 bg-gray-900 bg-opacity-50 md:hidden"
            onClick={toggleMobileSidebar}
            />
        )}
        </>
    );
    }