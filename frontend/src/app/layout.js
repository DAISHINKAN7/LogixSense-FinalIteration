// src/app/layout.js
import { Inter } from 'next/font/google';
import Sidebar from '../components/Sidebar';
import Header from '../components/Header';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'LogixSense - AI-Driven Logistics Analytics',
  description: 'Advanced logistics analytics platform powered by AI and machine learning',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
          <Sidebar />
          <div className="flex flex-col flex-1 overflow-hidden">
            <Header />
            <main className="flex-1 overflow-y-auto p-4 bg-gray-50 dark:bg-gray-900">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}