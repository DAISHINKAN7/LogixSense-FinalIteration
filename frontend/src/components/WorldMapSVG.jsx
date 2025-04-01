"use client";
import { ComposableMap, Geographies, Geography, Line, Marker } from "react-simple-maps";
import { useState } from "react";

const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

export default function WorldShippingMap() {
  const [tooltip, setTooltip] = useState("");

  // Shipping hubs with their coordinates (longitude, latitude)
  const shippingHubs = [
    { 
      coordinates: [-90.0490, 35.1495],  // Memphis
      name: "Memphis",
      color: "#3B82F6",
      description: "North American Logistics Hub"
    },
    { 
      coordinates: [55.2708, 25.2048],   // Dubai
      name: "Dubai", 
      color: "#10B981",
      description: "Middle Eastern Distribution Center"
    },
    { 
      coordinates: [-0.1276, 51.5074],   // London
      name: "London", 
      color: "#6366F1",
      description: "European Shipping Gateway"
    },
    { 
      coordinates: [77.1025, 28.7041],   // Delhi
      name: "Delhi", 
      color: "#F43F5E",
      description: "South Asian Logistics Network"
    },
    { 
      coordinates: [103.8198, 1.3521],   // Singapore
      name: "Singapore", 
      color: "#8B5CF6",
      description: "Southeast Asian Maritime Hub"
    },
    { 
      coordinates: [139.6917, 35.6895],  // Tokyo
      name: "Tokyo", 
      color: "#F97316",
      description: "East Asian Shipping Center"
    }
  ];

  // Shipping routes - all connected to Delhi
  const routes = [
    {
      from: shippingHubs[3],  // Delhi
      to: shippingHubs[0],    // Memphis
      color: "#FF6B6B",
      weight: 3
    },
    {
      from: shippingHubs[3],  // Delhi
      to: shippingHubs[1],    // Dubai
      color: "#4E6FFF",
      weight: 2
    },
    {
      from: shippingHubs[3],  // Delhi
      to: shippingHubs[2],    // London
      color: "#FDAB3D",
      weight: 2
    },
    {
      from: shippingHubs[3],  // Delhi
      to: shippingHubs[4],    // Singapore
      color: "#10B981",
      weight: 3
    },
    {
      from: shippingHubs[3],  // Delhi
      to: shippingHubs[5],    // Tokyo
      color: "#8B5CF6",
      weight: 2
    }
  ];

  return (
    <div className="relative w-full h-full">
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ 
          scale: 180,
          center: [20, 30]  // Centered on Europe/Middle East
        }}
        className="w-full h-full"
        style={{ 
          backgroundColor: '#F0F9FF',
          borderRadius: '8px',
          overflow: 'hidden'
        }}
      >
        {/* World Geographies with more vibrant colors */}
        <Geographies 
          geography={geoUrl}
          className="stroke-slate-300"
        >
          {({ geographies }) =>
            geographies.map((geo) => {
              // More nuanced color assignment
              const colorMap = {
                'United States of America': '#1E88E5',
                'China': '#5EEAD4',
                'Russia': '#FECACA',
                'Canada': '#CC5500',
                'Brazil': '#6EE7B7',
                'Australia': '#FDE047',
                'India': '#9B5DE5',
                'European Union': '#A78BFA',
                'Japan': '#5EAAA8',
                'United Kingdom': '#FF6B6B',
                'United Arab Emirates': '#0074E4',
                'Singapore': '#FF0080'

              };

              const defaultColor = "#CBD5E1";
              const geoColor = colorMap[geo.properties.name] || defaultColor;

              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  onMouseEnter={() => setTooltip(geo.properties.name)}
                  onMouseLeave={() => setTooltip("")}
                  style={{
                    default: { 
                      fill: geoColor, 
                      outline: "none",
                      opacity: 0.6
                    },
                    hover: { 
                      fill: geoColor, 
                      outline: "none",
                      opacity: 0.8
                    }
                  }}
                />
              );
            })
          }
        </Geographies>

        {/* Animated Shipping Routes */}
        {routes.map((route, index) => (
          <g key={index}>
            {/* Gradient Line with Complex Animation */}
            <defs>
              <linearGradient 
                id={`route-gradient-${index}`} 
                x1="0%" 
                y1="0%" 
                x2="100%" 
                y2="0%"
              >
                <stop 
                  offset="0%" 
                  stopColor={route.color} 
                  stopOpacity="0"
                />
                <stop 
                  offset="50%" 
                  stopColor={route.color} 
                  stopOpacity="1"
                />
                <stop 
                  offset="100%" 
                  stopColor={route.color} 
                  stopOpacity="0"
                />
              </linearGradient>
            </defs>

            {/* Static Route Base */}
            <Line
              from={route.from.coordinates}
              to={route.to.coordinates}
              stroke={route.color}
              strokeWidth={1}
              strokeLinecap="round"
              className="opacity-20"
            />
            
            {/* Animated Flight Path */}
            <Line
              from={route.from.coordinates}
              to={route.to.coordinates}
              stroke={`url(#route-gradient-${index})`}
              strokeWidth={route.weight}
              strokeLinecap="round"
              style={{
                animation: `flight-path-${index} 4s linear infinite`,
                strokeDasharray: "20,20"
              }}
            />
          </g>
        ))}

        {/* Shipping Hubs */}
        {shippingHubs.map((hub, index) => (
          <Marker 
            key={index} 
            coordinates={hub.coordinates}
          >
            <g 
              transform="translate(-6, -6)"
              className="group cursor-pointer"
            >
              {/* Orbital Pulsing Background */}
              <circle
                r={15}
                fill={hub.color}
                opacity={0.2}
                className="animate-ping group-hover:animate-pulse"
              />
              
              {/* Main Marker with Subtle Glow */}
              <circle
                r={6}
                fill="white"
                stroke={hub.color}
                strokeWidth={3}
                filter="url(#hub-glow)"
                className="transition-all duration-300 group-hover:scale-110"
              />

              {/* Tooltip */}
              <text
                textAnchor="middle"
                y={-15}
                fontSize={10}
                fontWeight={600}
                fill="black"
                className="opacity-0 group-hover:opacity-100 transition-opacity"
              >
                {hub.name}
              </text>
              <text
                textAnchor="middle"
                y={-3}
                fontSize={8}
                fill="#666"
                className="opacity-0 group-hover:opacity-100 transition-opacity"
              >
                {hub.description}
              </text>
            </g>
          </Marker>
        ))}

        {/* Glow Filter for Hubs */}
        <defs>
          <filter id="hub-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
      </ComposableMap>

      {/* Dynamic Styles for Route Animations */}
      <style jsx global>{`
        ${routes.map((route, index) => `
          @keyframes flight-path-${index} {
            0% {
              stroke-dashoffset: 40;
              opacity: 0.3;
            }
            50% {
              opacity: 1;
            }
            100% {
              stroke-dashoffset: 0;
              opacity: 0.3;
            }
          }
        `).join('\n')}
      `}</style>

      {/* Country Tooltip */}
      {tooltip && (
        <div className="absolute top-2 left-1/2 transform -translate-x-1/2 bg-white px-3 py-1 text-sm rounded shadow border z-10">
          {tooltip}
        </div>
      )}
    </div>
  );
}