// StatusBadge Component
export const StatusBadge = ({ status }) => {
    // Define color and text mapping for different statuses
    const statusConfig = {
      'In Transit': { 
        color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300', 
        text: 'In Transit' 
      },
      'Delivered': { 
        color: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300', 
        text: 'Delivered' 
      },
      'Pending': { 
        color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300', 
        text: 'Pending' 
      },
      'Delayed': { 
        color: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300', 
        text: 'Delayed' 
      },
      'Processing': { 
        color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300', 
        text: 'Processing' 
      },
      'default': { 
        color: 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300', 
        text: 'Unknown' 
      }
    };
  
    // Get status configuration, fallback to default if not found
    const { color, text } = statusConfig[status] || statusConfig['default'];
  
    return (
      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${color}`}>
        {text}
      </span>
    );
  };
  
  export default StatusBadge;