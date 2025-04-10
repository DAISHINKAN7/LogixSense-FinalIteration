import React from 'react';
import { Box, Typography, Breadcrumbs, Link } from '@mui/material';
import { usePathname } from 'next/navigation';

const PageHeader = ({ title, subtitle }) => {
  const pathname = usePathname();
  
  // Generate breadcrumbs based on the current path
  const generateBreadcrumbs = () => {
    const paths = pathname.split('/').filter(path => path);
    
    return paths.map((path, index) => {
      const href = `/${paths.slice(0, index + 1).join('/')}`;
      const isLast = index === paths.length - 1;
      const name = path.charAt(0).toUpperCase() + path.slice(1).replace(/-/g, ' ');
      
      return isLast ? (
        <Typography key={path} color="text.primary" fontWeight="medium">
          {name}
        </Typography>
      ) : (
        <Link key={path} underline="hover" color="inherit" href={href}>
          {name}
        </Link>
      );
    });
  };
  
  return (
    <Box sx={{ mb: 4 }}>
      <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 1 }}>
        <Link underline="hover" color="inherit" href="/">
          Dashboard
        </Link>
        {generateBreadcrumbs()}
      </Breadcrumbs>
      
      <Typography variant="h4" component="h1" gutterBottom>
        {title}
      </Typography>
      
      {subtitle && (
        <Typography variant="body1" color="text.secondary">
          {subtitle}
        </Typography>
      )}
    </Box>
  );
};

export default PageHeader;