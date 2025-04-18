import React, { useState, useEffect, useRef } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle,
  CardDescription 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Avatar } from '@/components/ui/avatar';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  AlertCircle, 
  Send, 
  Bot, 
  User, 
  RefreshCw,
  Loader,
  MessageSquare,
  Database,
  BarChart4,
  Search,
  Target,
  PenTool,
  Rows,
  Sparkles,
  Clock,
  ChevronRight,
  LineChart as LineChartIcon,
  PieChart as PieChartIcon,
  BarChart,
  Zap,
  BrainCircuit,
  Server
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  Tooltip, 
  TooltipContent, 
  TooltipTrigger 
} from '@/components/ui/tooltip';
import { ScrollArea } from '@/components/ui/scroll-area';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ChartRenderer from '@/components/ChartRenderer';

// Markdown components customization
const enhancedMarkdownComponents = {
  h1: props => <h1 className="text-2xl font-bold my-4 text-white" {...props} />,
  h2: props => <h2 className="text-xl font-bold my-3 pb-1 border-b border-[#333] text-purple-400" {...props} />,
  h3: props => <h3 className="text-lg font-bold my-2 text-white" {...props} />,
  p: props => <p className="mb-4" {...props} />,
  ul: props => <ul className="list-disc pl-6 mb-4 space-y-1" {...props} />,
  ol: props => <ol className="list-decimal pl-6 mb-4" {...props} />,
  li: props => <li className="mb-2" {...props} />,
  table: props => <table className="border-collapse table-auto w-full mb-4 bg-[#1a1a1a]" {...props} />,
  th: props => <th className="border border-[#333] px-4 py-2 text-left bg-[#222]" {...props} />,
  td: props => <td className="border border-[#333] px-4 py-2" {...props} />,
  code: props => <code className="bg-[#1a1a1a] rounded p-1 font-mono text-sm" {...props} />,
  pre: props => <pre className="bg-[#1a1a1a] rounded-md p-4 overflow-x-auto mb-4 font-mono text-sm border border-[#333]" {...props} />,
  strong: props => <strong className="text-white font-bold" {...props} />
};

const AIRecommendationBadge = ({ recommendation }) => {
  if (!recommendation) return null;
  
  return (
    <div className="my-3 p-2 rounded bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-700/30">
      <div className="flex items-start">
        <BrainCircuit className="h-4 w-4 mr-2 mt-0.5 text-purple-400" />
        <div>
          <span className="text-sm text-purple-300 font-medium">AI Recommendation:</span>
          <p className="text-sm text-[#A0A0A0] mt-1">{recommendation}</p>
        </div>
      </div>
    </div>
  );
};

// Enhanced AI features banner component
const EnhancedAIFeaturesBanner = ({ assistantStatus }) => {
  // Check if enhanced features are available
  const hasEnhancedFeatures = 
    assistantStatus?.details?.logistics_analyzer?.status === 'online' ||
    (assistantStatus?.details?.enhanced_features?.full_database_analysis === true);
  
  if (!hasEnhancedFeatures) return null;
  
  return (
    <div className="bg-gradient-to-r from-purple-900/30 to-cyan-900/30 border border-purple-800/50 rounded-md p-3 mb-4">
      <div className="flex items-center text-purple-300 mb-2">
        <BrainCircuit className="h-5 w-5 mr-2 text-purple-400" />
        <h3 className="font-medium">Enhanced AI Capabilities Active</h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
        <div className="flex items-center text-cyan-300">
          <Server className="h-4 w-4 mr-1.5 text-cyan-400" />
          <span>Full Database Analysis</span>
        </div>
        <div className="flex items-center text-green-300">
          <Zap className="h-4 w-4 mr-1.5 text-green-400" />
          <span>Comprehensive Insights</span>
        </div>
        <div className="flex items-center text-pink-300">
          <BarChart4 className="h-4 w-4 mr-1.5 text-pink-400" />
          <span>Advanced Visualizations</span>
        </div>
      </div>
    </div>
  );
};

// Visualization tools for chart type selection
const VisualizationTools = ({ onSelectChartType }) => {
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className="text-[#A0A0A0] text-sm mr-1">Chart type:</span>
      <Button 
        variant="outline" 
        size="sm" 
        className="border-[#333] bg-[#111] hover:bg-[#222] text-cyan-400 h-8"
        onClick={() => onSelectChartType('bar')}
      >
        <BarChart size={14} className="mr-1" />
        Bar
      </Button>
      <Button 
        variant="outline" 
        size="sm" 
        className="border-[#333] bg-[#111] hover:bg-[#222] text-green-400 h-8"
        onClick={() => onSelectChartType('line')}
      >
        <LineChartIcon size={14} className="mr-1" />
        Line
      </Button>
      <Button 
        variant="outline" 
        size="sm" 
        className="border-[#333] bg-[#111] hover:bg-[#222] text-purple-400 h-8"
        onClick={() => onSelectChartType('pie')}
      >
        <PieChartIcon size={14} className="mr-1" />
        Pie
      </Button>
    </div>
  );
};

const AutoVisualizationIndicator = () => {
  return (
    <div className="flex items-center justify-end mt-1 mb-2">
      <Badge className="bg-cyan-900/30 text-cyan-400 border border-cyan-700 px-2 py-0.5 text-xs">
        <BarChart4 size={10} className="mr-1" />
        Auto-Visualization
      </Badge>
    </div>
  );
};

const ResponseQualityIndicator = ({ analysisType, responseLength }) => {
  // Determine response quality based on length and analysis type
  let quality = "basic";
  if (responseLength > 1000 && analysisType === "comprehensive") {
    quality = "comprehensive";
  } else if (responseLength > 500) {
    quality = "detailed";
  }
  
  // Don't show for basic responses
  if (quality === "basic") return null;
  
  const indicators = {
    detailed: {
      icon: <Search size={10} className="mr-1" />,
      text: "Detailed Response",
      classes: "bg-green-900/30 text-green-400 border border-green-700"
    },
    comprehensive: {
      icon: <BrainCircuit size={10} className="mr-1" />,
      text: "Comprehensive Analysis",
      classes: "bg-purple-900/30 text-purple-400 border border-purple-700"
    }
  };
  
  const indicator = indicators[quality];
  
  return (
    <Badge className={`px-2 py-0.5 text-xs ${indicator.classes}`}>
      {indicator.icon}
      {indicator.text}
    </Badge>
  );
};

// Vibrant Modern Message component for chat
// Full updated MessageBubble component 
const MessageBubble = ({ message, isLoading, onRequestVisualization }) => {
  const isUser = message.role === 'user';
  
  // Vibrant color options for user messages
  const userColors = {
    bg: "bg-gradient-to-r from-[#7C3AED] to-[#C026D3]",
    border: "border-[#9D4EDD]",
    text: "text-white"
  };
  
  // Vibrant assistant colors (different from user)
  const assistantColors = {
    bg: "bg-[#111111]",
    border: "border-[#333333]",
    accentBorder: "border-l-[3px] border-l-[#06B6D4]",
    text: "text-white"
  };

  // Check if message has visualization data
  const hasVisualization = message.visualization && 
    message.visualization.data && 
    message.visualization.data.length > 0;
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} max-w-[85%] group`}>
        <div className="flex-shrink-0">
          <Avatar className={`${isUser ? 'ml-3' : 'mr-3'} h-9 w-9 ${
            isUser 
              ? 'bg-gradient-to-r from-[#7C3AED] to-[#C026D3]' 
              : 'bg-gradient-to-r from-[#0891B2] to-[#06B6D4]'
          }`}>
            {isUser ? 
              <User size={18} className="text-white" /> : 
              <Bot size={18} className="text-white" />
            }
          </Avatar>
        </div>
        
        <div 
          className={`rounded-lg px-4 py-3 ${
            isUser 
              ? `${userColors.bg} ${userColors.text}` 
              : `${assistantColors.bg} ${assistantColors.text} border ${assistantColors.border} ${assistantColors.accentBorder}`
          }`}
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="flex space-x-1">
                <div className="h-2 w-2 rounded-full bg-cyan-400 animate-ping"></div>
                <div className="h-2 w-2 rounded-full bg-cyan-400 animate-ping" style={{animationDelay: '0.2s'}}></div>
                <div className="h-2 w-2 rounded-full bg-cyan-400 animate-ping" style={{animationDelay: '0.4s'}}></div>
              </div>
            </div>
          ) : (
            <div>
              <EnhancedResponseDisplay 
                content={message.content} 
                hasVisualization={hasVisualization} 
              />
              
              {/* Analysis Insights Badge - for responses with comprehensive analysis */}
              {!isUser && (
                <div className="mt-2 mb-3">
                  <ResponseQualityIndicator 
                    analysisType={message.analysisType} 
                    responseLength={message.content.length} 
                  />
                </div>
              )}
              
              {/* Visualization section */}
              {hasVisualization && (
                <>
                  {/* AI Recommended Visualization Indicator */}
                  {message.visualization.is_ai_recommended && (
                    <div className="flex items-center justify-end mt-1 mb-2">
                      <Badge className="bg-cyan-900/30 text-cyan-400 border border-cyan-700 px-2 py-0.5 text-xs">
                        <BrainCircuit size={10} className="mr-1" />
                        AI-Recommended Visualization
                      </Badge>
                    </div>
                  )}
                  {/* Regular Auto Visualization Indicator */}
                  {message.autoVisualization && !message.visualization.is_ai_recommended && <AutoVisualizationIndicator />}
                  <div className="mt-4 border border-[#333] rounded-md p-4 bg-[#0A0A0A]">
                    <Tabs defaultValue="chart" className="w-full">
                      <div className="flex justify-between items-center mb-3">
                        <TabsList className="bg-[#1A1A1A]">
                          <TabsTrigger value="chart" className="data-[state=active]:bg-[#222]">Chart</TabsTrigger>
                          <TabsTrigger value="data" className="data-[state=active]:bg-[#222]">Data</TabsTrigger>
                        </TabsList>
                        
                        {/* Chart type selector */}
                        {!isUser && (
                          <div className="flex items-center">
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              className="h-7 px-2 text-[#A0A0A0] hover:text-white hover:bg-[#222]"
                              onClick={() => onRequestVisualization(message.content, 'bar')}
                            >
                              <BarChart size={14} className="mr-1" />
                              Bar
                            </Button>
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              className="h-7 px-2 text-[#A0A0A0] hover:text-white hover:bg-[#222]"
                              onClick={() => onRequestVisualization(message.content, 'line')}
                            >
                              <LineChartIcon size={14} className="mr-1" />
                              Line
                            </Button>
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              className="h-7 px-2 text-[#A0A0A0] hover:text-white hover:bg-[#222]"
                              onClick={() => onRequestVisualization(message.content, 'pie')}
                            >
                              <PieChartIcon size={14} className="mr-1" />
                              Pie
                            </Button>
                          </div>
                        )}
                      </div>
                      
                      <TabsContent value="chart" className="mt-0">
                        {/* AI recommendation display */}
                        {message.visualization.ai_recommendation && (
                          <AIRecommendationBadge recommendation={message.visualization.ai_recommendation} />
                        )}
                        <ChartRenderer 
                          type={message.visualization.visualization_type}
                          data={message.visualization.data}
                          config={message.visualization.config}
                          description={message.visualization.description}
                        />
                      </TabsContent>
                      
                      <TabsContent value="data" className="mt-0">
                        <div className="max-h-[300px] overflow-auto border border-[#333] rounded-md p-2 bg-[#1A1A1A]">
                          <pre className="text-sm text-[#A0A0A0]">
                            {JSON.stringify(message.visualization.data, null, 2)}
                          </pre>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </div>
                </>
              )}
              
              {/* Visualization request button for user queries */}
              {isUser && !hasVisualization && (
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="mt-2 text-[#A0A0A0] hover:text-white hover:bg-transparent p-0"
                  onClick={() => onRequestVisualization(message.content)}
                >
                  <BarChart4 size={14} className="mr-1" />
                  Visualize this query
                </Button>
              )}
              
              <div className={`mt-2 text-xs opacity-70 text-right flex items-center justify-end`}>
                <Clock size={12} className="mr-1 opacity-50" />
                {message.timestamp}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Vibrant Suggestion pill component
const SuggestionPill = ({ suggestion, onClick, icon, colorScheme }) => {
  // Colorful suggestion pills - each with a unique vibrant color
  const colorSchemes = {
    purple: "border-purple-500 bg-purple-500/10 hover:bg-purple-500/20 text-purple-400",
    cyan: "border-cyan-500 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400",
    pink: "border-pink-500 bg-pink-500/10 hover:bg-pink-500/20 text-pink-400",
    green: "border-green-500 bg-green-500/10 hover:bg-green-500/20 text-green-400",
    orange: "border-orange-500 bg-orange-500/10 hover:bg-orange-500/20 text-orange-400",
  };
  
  const colors = colorSchemes[colorScheme] || colorSchemes.purple;
  
  return (
    <Button 
      variant="outline" 
      className={`mr-2 mb-2 text-sm whitespace-nowrap rounded-full border ${colors} transition-all duration-300`}
      onClick={() => onClick(suggestion)}
    >
      {icon}
      {suggestion.length > 40 ? `${suggestion.substring(0, 40)}...` : suggestion}
    </Button>
  );
};

// Analysis category badges component
const AnalysisCategoryBadges = ({ onAnalysisCategoryClick }) => {
  const categories = [
    { name: "Destinations", icon: <Target size={12} className="mr-1" /> },
    { name: "Weights", icon: <PieChartIcon size={12} className="mr-1" /> },
    { name: "Carriers", icon: <Server size={12} className="mr-1" /> },
    { name: "Time Trends", icon: <LineChartIcon size={12} className="mr-1" /> },
    { name: "Risks", icon: <AlertCircle size={12} className="mr-1" /> }
  ];
  
  return (
    <div className="flex flex-wrap gap-1 mb-2">
      {categories.map((category, index) => (
        <Badge 
          key={index}
          className="bg-[#222] hover:bg-[#333] cursor-pointer border-0 px-3 py-1"
          onClick={() => onAnalysisCategoryClick(`Analyze our ${category.name.toLowerCase()}`)}
        >
          {category.icon}
          {category.name}
        </Badge>
      ))}
    </div>
  );
};

const EnhancedResponseDisplay = ({ content, hasVisualization }) => {
  // Process content to detect sections and highlight them
  const processContent = (text) => {
    // Look for section headers (commonly used in enhanced responses)
    const sectionPattern = /\n## (.*?)\n/g;
    let sections = [];
    let match;
    
    while ((match = sectionPattern.exec(text)) !== null) {
      sections.push(match[1]); // Get the section title
    }
    
    // If we found sections, enhance the display
    if (sections.length > 0) {
      return (
        <div className="enhanced-response">
          {/* Table of Contents if there are multiple sections */}
          {sections.length > 2 && (
            <div className="mb-3 p-2 rounded bg-[#111] border border-[#222]">
              <div className="text-sm text-purple-400 mb-1">Contents:</div>
              <div className="flex flex-wrap gap-2">
                {sections.map((section, i) => (
                  <Badge key={i} className="bg-[#222] hover:bg-[#333] cursor-default border-0">
                    {section}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          
          {/* The actual markdown content */}
          <div className="prose dark:prose-invert max-w-none">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]} 
              components={enhancedMarkdownComponents}
            >
              {text}
            </ReactMarkdown>
          </div>
        </div>
      );
    }
    
    // Default rendering if no sections detected
    return (
      <div className="prose dark:prose-invert max-w-none">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]} 
          components={enhancedMarkdownComponents}
        >
          {text}
        </ReactMarkdown>
      </div>
    );
  };
  
  return processContent(content);
};

// Main LogixAssistant component
const LogixAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [assistantStatus, setAssistantStatus] = useState(null);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check assistant status on mount
  useEffect(() => {
    checkAssistantStatus();
    fetchSuggestions();
  }, []);

  // Check if the assistant is available
  const checkAssistantStatus = async () => {
    try {
      const response = await fetch('/api/assistant/status');
      const data = await response.json();
      console.log('Assistant status response:', data); // Debug log
      
      setAssistantStatus(data);
      
      if (data.status !== 'online') {
        setError("Assistant service is currently offline. Please check server logs.");
      } else {
        setError(null);
      }
    } catch (err) {
      console.error('Failed to check assistant status:', err);
      setError("Failed to connect to the assistant. Please check if the server is running.");
      setAssistantStatus({ status: 'error' });
    }
  };

  // Fetch query suggestions
  const fetchSuggestions = async () => {
    try {
      const response = await fetch('/api/assistant/suggestions');
      const data = await response.json();
      setSuggestions(data.suggestions || []);
    } catch (err) {
      console.error('Failed to fetch suggestions:', err);
      setSuggestions([]);
    }
  };

  // Format the current time for message timestamp
  const getFormattedTime = () => {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;
    
    // Add user message to the chat
    const userMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: getFormattedTime()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);
    
    // Add temporary assistant message with loading state
    const tempAssistantMessage = {
      role: 'assistant',
      content: '',
      timestamp: getFormattedTime()
    };
    
    setMessages(prev => [...prev, tempAssistantMessage]);
    
    // Set up a timeout for slow responses
    const timeoutId = setTimeout(() => {
      // Update the loading message to show it's still working
      setMessages(prev => {
        const updated = [...prev];
        const lastMessage = updated[updated.length - 1];
        if (lastMessage.role === 'assistant' && lastMessage.content === '') {
          updated[updated.length - 1] = {
            ...lastMessage,
            content: "I'm analyzing your entire database to provide a comprehensive response. This might take a moment for complex queries..."
          };
        }
        return updated;
      });
    }, 8000); // Show message after 8 seconds of loading
    
    try {
      // Send the query to the API with enhanced options
      const response = await fetch('/api/assistant/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          options: {
            detailed: true,
            comprehensive_analysis: true,
            use_entire_database: true,
            generate_visualization: true, // Request automatic visualization
            response_timeout: 60000 // 60 second timeout for the response
          }
        }),
      });
      
      clearTimeout(timeoutId); // Clear the timeout
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get response');
      }
      
      const data = await response.json();
      
      // Determine if this used comprehensive analysis
      const usedComprehensiveAnalysis = 
        data.context && 
        (data.context.analysis_used === 'comprehensive' || 
         data.context.used_full_database === true);
      
      // Update the last message (the temporary one) with the assistant's response
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: data.answer,
          timestamp: getFormattedTime(),
          visualization: data.visualization,
          autoVisualization: data.visualization ? true : false,
          analysisType: usedComprehensiveAnalysis ? 'comprehensive' : 'regular',
          responseLength: data.context?.response_length || data.answer.length
        };
        return updated;
      });
      
      // If no visualization was provided but one would be useful, request one
      if (!data.visualization && isVisualizableQuery(userMessage.content)) {
        handleRequestVisualization(userMessage.content);
      }
      
    } catch (err) {
      clearTimeout(timeoutId); // Clear the timeout
      console.error('Failed to get assistant response:', err);
      setError(err.message || 'Failed to get response from the assistant');
      
      // Update temporary message to show error
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: 'I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists.',
          timestamp: getFormattedTime(),
          error: true
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
      // Focus the input field again
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };
  
  // Add a helper function to detect if a query is likely to benefit from visualization
  const isVisualizableQuery = (query) => {
    query = query.toLowerCase();
    
    // Keywords that suggest data that can be visualized
    const visualKeywords = [
      "distribution", "breakdown", "trend", "compare", "analysis", "analyze",
      "show", "display", "chart", "graph", "plot", "visualize", "visualization",
      "destination", "weight", "carrier", "airline", "commodity", "product",
      "time", "monthly", "percentage", "proportion", "count", "total",
      "top", "summary", "overview", "statistics", "stats", "numbers"
    ];
    
    // Check if the query contains any of these keywords
    return visualKeywords.some(keyword => query.includes(keyword));
  };
  
  // Update the RequestVisualization function to handle automatically generating visualizations
  const handleRequestVisualization = async (query, vizType = null) => {
    setIsLoading(true);
    
    try {
      // If no specific visualization type is requested, try to infer it
      if (!vizType) {
        query = query.toLowerCase();
        if (query.includes("time") || query.includes("trend") || query.includes("period") || query.includes("month")) {
          vizType = "line";
        } else if (query.includes("distribution") || query.includes("breakdown") || 
                  query.includes("proportion") || query.includes("percentage")) {
          vizType = "pie";
        } else {
          vizType = "bar"; // Default
        }
      }
      
      const response = await fetch('/api/assistant/visualize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          visualization_type: vizType,
          timeout: 30000 // 30 second timeout for visualization generation
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate visualization');
      }
      
      const vizData = await response.json();
      
      // Check if we got valid visualization data
      if (!vizData || !vizData.data || vizData.data.length === 0) {
        console.log("No visualization data received");
        return;
      }
      
      // Find the most recent assistant message to attach the visualization to
      setMessages(prev => {
        const updated = [...prev];
        
        // Find the last assistant message that responded to this query
        const userMsgIndex = updated.findIndex(msg => 
          msg.role === 'user' && msg.content === query
        );
        
        if (userMsgIndex !== -1 && userMsgIndex < updated.length - 1) {
          // Find the next assistant message after this user query
          const assistantMsgIndex = userMsgIndex + 1;
          
          if (assistantMsgIndex < updated.length && 
              updated[assistantMsgIndex].role === 'assistant') {
            // Attach visualization to this assistant message
            updated[assistantMsgIndex] = {
              ...updated[assistantMsgIndex],
              visualization: vizData
            };
          }
        } else {
          // If we can't find a match, add as a new assistant message
          updated.push({
            role: 'assistant',
            content: `Here's a visualization of the data related to your query:`,
            timestamp: getFormattedTime(),
            visualization: vizData
          });
        }
        
        return updated;
      });
      
    } catch (err) {
      console.error('Failed to generate visualization:', err);
      // Don't show error to user if this was an automatic visualization attempt
      if (vizType) {
        setError(err.message || 'Failed to generate visualization');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Handle pressing Enter to send
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Handle clicking a suggestion
  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  // Handle clicking an analysis category
  const handleAnalysisCategoryClick = (query) => {
    setInput(query);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  // Handle initializing the vector database
  const handleInitializeVectorDB = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/assistant/init-vector-index', {
        method: 'POST',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to initialize vector database');
      }
      
      const data = await response.json();
      
      // Add system message
      setMessages(prev => [
        ...prev, 
        {
          role: 'assistant',
          content: `Vector database initialized successfully with ${data.message}. The enhanced AI features are now ready to use.`,
          timestamp: getFormattedTime()
        }
      ]);
      
      // Refresh assistant status
      checkAssistantStatus();
      
    } catch (err) {
      console.error('Failed to initialize vector database:', err);
      setError(err.message || 'Failed to initialize vector database');
    } finally {
      setIsLoading(false);
    }
  };

  // Welcome message on first load
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          role: 'assistant',
          content: "👋 Hello! I'm your LogixSense AI Assistant with enhanced database analysis capabilities. I can analyze your entire logistics database to provide comprehensive insights, answer detailed questions about shipments, and visualize trends in your supply chain operations. Ask me anything about your logistics data!",
          timestamp: getFormattedTime()
        }
      ]);
    }
  }, [messages.length]);

  // Icon maps for suggestion categories with colors
  const suggestionConfig = [
    { 
      pattern: /(destination|where|locate)/i, 
      icon: <Target size={14} className="mr-2 text-purple-400" />,
      color: "purple" 
    },
    { 
      pattern: /(analyze|trend|compare|performance)/i, 
      icon: <BarChart4 size={14} className="mr-2 text-cyan-400" />,
      color: "cyan" 
    },
    { 
      pattern: /(show|find|what|identify|visualize|chart|graph)/i, 
      icon: <Search size={14} className="mr-2 text-pink-400" />,
      color: "pink" 
    },
    { 
      pattern: /(shipping|delivery|transit)/i, 
      icon: <MessageSquare size={14} className="mr-2 text-green-400" />,
      color: "green" 
    },
    { 
      pattern: /.*/i, 
      icon: <PenTool size={14} className="mr-2 text-orange-400" />,
      color: "orange" 
    }
  ];

  // Get the right icon and color for a suggestion
  const getConfigForSuggestion = (suggestion) => {
    for (const config of suggestionConfig) {
      if (config.pattern.test(suggestion)) {
        return config;
      }
    }
    return suggestionConfig[suggestionConfig.length - 1]; // Default
  };

  return (
    <Card className="w-full h-full flex flex-col bg-[#0A0A0A] border-[#222]">
      <CardHeader className="border-b border-[#222] px-6 py-4 bg-gradient-to-r from-[#0A0A0A] to-[#111]">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="flex items-center text-white text-xl font-light">
              <div className="bg-gradient-to-r from-[#7C3AED] to-[#C026D3] p-1.5 rounded-md mr-3">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
              LOGIXSENSE <span className="font-bold ml-2">ASSISTANT</span>
              
              {/* Add enhanced AI badge */}
              {assistantStatus?.details?.logistics_analyzer?.status === 'online' && (
                <Badge className="ml-3 bg-purple-900/30 text-purple-400 border border-purple-700">
                  <BrainCircuit size={12} className="mr-1" />
                  ENHANCED AI
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="text-[#A0A0A0] mt-1 flex items-center">
              <div className="w-2 h-2 bg-cyan-400 rounded-full mr-2"></div>
              Enterprise logistics intelligence system with comprehensive database analysis
            </CardDescription>
          </div>
          
          <div className="flex items-center space-x-3">
            {assistantStatus ? (
              <Tooltip>
                <TooltipTrigger asChild>
                <Badge className={`px-3 py-1 ${
                  assistantStatus.status === 'online' 
                    ? 'bg-green-900/30 text-green-400 border border-green-700' 
                    : 'bg-red-900/30 text-red-400 border border-red-700'
                }`}>
                    <div className={`w-2 h-2 rounded-full mr-2 ${
                      assistantStatus.status === 'online' 
                        ? 'bg-green-400 animate-pulse' 
                        : 'bg-red-400'
                    }`}></div>
                    {assistantStatus.status === 'online' 
                        ? 'ONLINE' 
                        : 'OFFLINE'}
                </Badge>
                </TooltipTrigger>
                <TooltipContent className="bg-[#1A1A1A] border-[#333]">
                  {assistantStatus.status === 'online' 
                    ? `Assistant is online and ready to help` 
                    : `Assistant is currently unavailable`}
                </TooltipContent>
              </Tooltip>
            ) : (
              <Badge className="bg-gray-900/30 text-gray-400 border border-gray-700 px-3 py-1">
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-pulse mr-2"></div>
                CONNECTING
              </Badge>
            )}
            
            <Button 
              variant="outline" 
              size="sm" 
              onClick={checkAssistantStatus}
              disabled={isLoading}
              className="border-[#333] bg-[#111] hover:bg-[#222] text-white"
            >
              <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
            </Button>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleInitializeVectorDB}
                  disabled={isLoading}
                  className="border-cyan-800/50 bg-cyan-900/20 hover:bg-cyan-800/30 text-cyan-400"
                >
                  <Database size={14} className="mr-1.5" />
                  Initialize DB
                </Button>
              </TooltipTrigger>
              <TooltipContent className="bg-[#1A1A1A] border-[#333]">
                Initialize the vector database with your logistics data
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="flex-grow flex flex-col h-[calc(100%-10rem)] pb-0 pt-4">
        {error && (
          <Alert variant="destructive" className="mb-4 bg-red-900/20 border border-red-600/30 text-white">
            <AlertCircle className="h-4 w-4 text-red-400" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        {/* Enhanced AI Features Banner */}
        <EnhancedAIFeaturesBanner assistantStatus={assistantStatus} />
        
        <ScrollArea className="flex-grow pr-4 mb-4 custom-scrollbar">
          <div className="space-y-2">
            {messages.map((message, index) => (
              <MessageBubble 
                key={index} 
                message={message} 
                isLoading={isLoading && index === messages.length - 1 && message.role === 'assistant'} 
                onRequestVisualization={handleRequestVisualization}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
        
        {/* Analysis Category Quick Access */}
        {assistantStatus?.details?.logistics_analyzer?.status === 'online' && messages.length <= 3 && (
          <div className="mb-3">
            <Label className="text-sm mb-2 flex items-center text-[#A0A0A0]">
              <BrainCircuit size={14} className="mr-2 text-purple-400" />
              DATA ANALYSIS CATEGORIES
            </Label>
            <AnalysisCategoryBadges onAnalysisCategoryClick={handleAnalysisCategoryClick} />
          </div>
        )}
        
        {/* Suggestions with vibrant colors */}
        {suggestions.length > 0 && messages.length <= 2 && (
          <div className="mb-4 bg-[#111] p-4 rounded-md border border-[#222]">
            <Label className="text-sm mb-3 flex items-center text-[#A0A0A0]">
              <Rows size={14} className="mr-2 text-gradient-purple" />
              SUGGESTED QUERIES
            </Label>
            <div className="flex flex-wrap">
              {suggestions.slice(0, 4).map((suggestion, index) => {
                const config = getConfigForSuggestion(suggestion);
                return (
                  <SuggestionPill 
                    key={index} 
                    suggestion={suggestion} 
                    onClick={handleSuggestionClick}
                    icon={config.icon}
                    colorScheme={config.color}
                  />
                );
              })}
            </div>
          </div>
        )}
        
        <div className="pt-4 pb-4 border-t border-[#222] mt-auto">
          <div className="flex gap-2">
            <div className="relative flex-grow">
              <Textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask me about your logistics data or request a comprehensive analysis..."
                  className="flex-grow min-h-[56px] resize-none bg-[#111] border-[#222] focus:border-purple-500 rounded-md placeholder:text-[#666] text-white"
                  disabled={isLoading || assistantStatus?.status !== 'online'}
              />
              {/* Character count indicator */}
              {input.length > 0 && (
                <div className="absolute bottom-2 right-2 text-xs text-[#666]">
                  {input.length} characters
                </div>
              )}
            </div>
            <Button 
              onClick={handleSendMessage} 
              disabled={!input.trim() || isLoading || assistantStatus?.status !== 'online'}
              className="self-end bg-gradient-to-r from-[#7C3AED] to-[#C026D3] hover:opacity-90 text-white h-[56px] px-5"
            >
              {isLoading ? (
                <Loader size={16} className="animate-spin" />
              ) : (
                <>
                  <Send size={16} className="mr-2" />
                  Send
                </>
              )}
            </Button>
          </div>
          <div className="mt-2 text-xs text-[#666] flex items-center justify-between">
            <div>
              Press <kbd className="rounded border border-[#333] px-1 py-0.5 bg-[#111] text-[#A0A0A0]">Enter</kbd> to send,
              <kbd className="rounded border border-[#333] px-1 py-0.5 bg-[#111] text-[#A0A0A0] ml-1">Shift + Enter</kbd> for new line
            </div>
            {assistantStatus?.details?.vector_db && (
              <span className="flex items-center text-[#A0A0A0] bg-[#111] px-2 py-1 rounded border border-[#222]">
                <BarChart4 size={12} className="mr-1 text-cyan-400" />
                <span className="text-cyan-400">{assistantStatus.details.vector_db.document_count}</span>
                <span className="ml-1">documents indexed</span>
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Add this to your global CSS
const globalStyles = `
  .text-gradient-purple {
    background: linear-gradient(to right, #7C3AED, #C026D3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .custom-scrollbar::-webkit-scrollbar {
    width: 6px;
  }
  
  .custom-scrollbar::-webkit-scrollbar-track {
    background: #111;
    border-radius: 8px;
  }
  
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background: #333;
    border-radius: 8px;
  }
  
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: #444;
  }
`;
export default LogixAssistant;