// src/app/page.jsx
'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to dashboard
    router.push('/dashboard');
  }, [router]);

  return (
    <div className="flex items-center justify-center h-screen bg-gray-50 dark:bg-gray-900">
      <div className="text-center">
        <div className="h-8 w-8 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-white font-bold">LS</span>
        </div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">LogixSense</h1>
        <p className="mt-2 text-gray-500 dark:text-gray-400">AI-Driven Logistics Analytics Platform</p>
        <div className="mt-4 text-blue-600 dark:text-blue-500">Redirecting to dashboard...</div>
      </div>
    </div>
  );
}