// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://localhost:8000/api/:path*',
          basePath: false,
        },
      ]
    },
    serverRuntimeConfig: {
      apiTimeout: 120000 // 2 minutes
    }
  }
  
  module.exports = nextConfig