// src/app/forecasting/components/ForecastAnimations.jsx
'use client';

import { useEffect } from 'react';
import { motion, useAnimation } from 'framer-motion';

export const GridPattern = ({ size = 32, className = "", strokeWidth = 1, ...props }) => {
  return (
    <svg
      aria-hidden="true"
      className={`absolute inset-0 h-full w-full ${className}`}
      {...props}
    >
      <defs>
        <pattern
          id="grid-pattern"
          width={size}
          height={size}
          patternUnits="userSpaceOnUse"
          x="50%"
          y="50%"
          patternTransform="translate(-1 -1)"
        >
          <path
            d={`M.5 ${size}V0m${size} 0v${size}`}
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
          />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid-pattern)" />
    </svg>
  );
};

export const WaveAnimation = ({ className = "", delay = 0, ...props }) => {
  return (
    <motion.div
      className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}
      {...props}
    >
      <motion.div
        className="absolute inset-0 h-full w-full"
        initial={{ y: "100%" }}
        animate={{ 
          y: ["-100%", "100%"],
          transition: { 
            repeat: Infinity, 
            duration: 10,
            ease: "linear",
            delay 
          }
        }}
      >
        <svg viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg">
          <path 
            d="M 0 500 Q 250 300 500 500 Q 750 700 1000 500 L 1000 1000 L 0 1000 Z" 
            fill="currentColor"
            fillOpacity="0.05"
          />
        </svg>
      </motion.div>
    </motion.div>
  );
};

export const GlowEffect = ({ className = "", ...props }) => {
  const controls = useAnimation();
  
  useEffect(() => {
    const animate = async () => {
      await controls.start({
        opacity: [0.4, 0.8, 0.4],
        scale: [1, 1.05, 1],
        transition: { 
          duration: 4, 
          ease: "easeInOut",
          repeat: Infinity,
          repeatType: "reverse"
        }
      });
    };
    
    animate();
  }, [controls]);
  
  return (
    <motion.div
      animate={controls}
      className={`absolute inset-0 rounded-full blur-3xl pointer-events-none ${className}`}
      {...props}
    />
  );
};

export const LoadingSpinner = ({ size = "md", className = "" }) => {
  const sizeClasses = {
    sm: "h-4 w-4",
    md: "h-8 w-8",
    lg: "h-12 w-12",
  };
  
  return (
    <div className={`${className} flex items-center justify-center`}>
      <motion.div
        className={`${sizeClasses[size]} border-4 border-t-blue-500 border-r-transparent border-b-green-500 border-l-transparent rounded-full`}
        animate={{ rotate: 360 }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
};

export const NumberCounter = ({ end, duration = 2, className = "", prefix = "", suffix = "" }) => {
  return (
    <motion.span
      className={className}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {prefix}
      <motion.span
        initial={{ count: 0 }}
        animate={{ count: end }}
        transition={{ duration }}
      >
        {({ count }) => Math.round(count).toLocaleString()}
      </motion.span>
      {suffix}
    </motion.span>
  );
};

export const Pulse = ({ children, className = "", ...props }) => {
  return (
    <motion.div
      className={`relative ${className}`}
      {...props}
    >
      <motion.div
        className="absolute inset-0 rounded-md bg-blue-500/20 dark:bg-blue-500/10"
        animate={{ 
          scale: [1, 1.05, 1],
          opacity: [0.7, 0.3, 0.7] 
        }}
        transition={{ 
          duration: 2,
          repeat: Infinity,
          repeatType: "reverse"
        }}
      />
      {children}
    </motion.div>
  );
};

export const ShiftingBackground = ({ className = "", ...props }) => {
  return (
    <motion.div
      className={`absolute inset-0 z-0 ${className}`}
      {...props}
    >
      <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="shifting-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.1">
              <animate
                attributeName="stopColor"
                values="#3b82f6; #10b981; #8b5cf6; #3b82f6"
                dur="10s"
                repeatCount="indefinite"
              />
            </stop>
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.1">
              <animate
                attributeName="stopColor"
                values="#10b981; #8b5cf6; #3b82f6; #10b981"
                dur="10s"
                repeatCount="indefinite"
              />
            </stop>
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#shifting-gradient)" />
      </svg>
    </motion.div>
  );
};

export default {
  GridPattern,
  WaveAnimation,
  GlowEffect,
  LoadingSpinner,
  NumberCounter,
  Pulse,
  ShiftingBackground
};