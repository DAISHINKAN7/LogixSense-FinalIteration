// src/app/assistant/page.jsx
'use client';

import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Send, Loader2, Bot, User, FileText, Info, RefreshCw, ChevronRight, X } from 'lucide-react';

export default function AssistantPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: "Hello! I'm LogixSense AI Assistant. How can I help you with your logistics operations today?",
      timestamp: new Date().toISOString(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const endOfMessagesRef = useRef(null);
  
  // Mock suggestions
  const suggestions = [
    "Analyze my shipment trends over the last month",
    "What's the best shipping route to Singapore?",
    "Help me optimize my shipping costs",
    "Identify potential delays in my supply chain",
    "Generate a risk assessment report"
  ];

  // Function to handle user messages
  const handleSendMessage = (content = inputValue) => {
    if (!content.trim()) return;

    // Add user message
    const userMessage = {
      id: messages.length + 1,
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setShowSuggestions(false);
    setIsLoading(true);

    // Simulate AI response (would be an API call in production)
    setTimeout(() => {
      let responseContent = '';

      // Determine response based on message content
      if (content.toLowerCase().includes('shipment') && content.toLowerCase().includes('trend')) {
        responseContent = "Based on your shipment data for the last month, I've noticed a 12% increase in volume to Asian destinations, particularly Singapore and Japan. The average shipment weight has decreased by 5%, indicating a shift towards higher-value, lower-weight goods. Would you like me to generate a detailed trend report?";
      } else if (content.toLowerCase().includes('route') && content.toLowerCase().includes('singapore')) {
        responseContent = "For shipments to Singapore, I recommend route SK-104 via Bangkok. This route has shown 97.8% on-time performance over the last quarter with an average transit time of 3.2 days. Alternative route SK-108 via Hong Kong has lower costs but 4.5 days average transit time. Would you like to see a comparison of these routes?";
      } else if (content.toLowerCase().includes('cost') && content.toLowerCase().includes('optimize')) {
        responseContent = "I've analyzed your shipping patterns and identified several cost optimization opportunities. By consolidating shipments to Japan and South Korea, you could save approximately ₹280,000 monthly. Additionally, switching carriers for European routes could reduce costs by 8-12%. Would you like me to prepare a detailed cost optimization plan?";
      } else if (content.toLowerCase().includes('delay') || content.toLowerCase().includes('risk')) {
        responseContent = "I've detected potential delays in your Dubai supply chain due to reported port congestion. Current risk level is moderate (62/100). I recommend scheduling shipments 2 days earlier than usual for the next 2 weeks to mitigate delays. Would you like a comprehensive risk assessment for all your active routes?";
      } else {
        responseContent = "I understand you're interested in " + content + ". Based on your logistics data, I can help analyze this further. Would you like me to generate a detailed report on this topic, or would you prefer specific recommendations?";
      }

      const assistantMessage = {
        id: messages.length + 2,
        role: 'assistant',
        content: responseContent,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 2000);
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-[calc(100vh-130px)]">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold tracking-tight">AI Assistant</h1>
        <div className="flex items-center space-x-2">
          <button className="flex items-center space-x-1 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 px-3 py-1.5 rounded-md text-sm">
            <Info size={16} />
            <span>Help</span>
          </button>
          <button className="flex items-center space-x-1 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 px-3 py-1.5 rounded-md text-sm">
            <RefreshCw size={16} />
            <span>New Chat</span>
          </button>
        </div>
      </div>
      
      <Card className="flex-1 flex flex-col overflow-hidden">
        <CardHeader className="pb-3 border-b dark:border-gray-700">
          <CardTitle className="flex items-center text-lg">
            <Bot className="mr-2 h-5 w-5" />
            <span>LogixSense AI Assistant</span>
          </CardTitle>
        </CardHeader>
        
        <CardContent className="flex-1 overflow-y-auto p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === 'assistant' ? 'justify-start' : 'justify-end'
                }`}
              >
                <div
                  className={`flex max-w-[80%] rounded-lg p-4 ${
                    message.role === 'assistant'
                      ? 'bg-gray-100 dark:bg-gray-800'
                      : 'bg-blue-600 text-white'
                  }`}
                >
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      {message.role === 'assistant' ? (
                        <Bot className="h-4 w-4" />
                      ) : (
                        <User className="h-4 w-4" />
                      )}
                      <span className="text-xs">
                        {message.role === 'assistant' ? 'AI Assistant' : 'You'}
                      </span>
                      <span className="text-xs opacity-70">
                        {new Date(message.timestamp).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit'
                        })}
                      </span>
                    </div>
                    <p className="text-sm">{message.content}</p>
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="flex max-w-[80%] rounded-lg p-4 bg-gray-100 dark:bg-gray-800">
                  <div className="flex items-center space-x-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">LogixSense AI is thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={endOfMessagesRef} />
          </div>
        </CardContent>
        
        {/* Suggestions */}
        {showSuggestions && messages.length < 3 && (
          <div className="px-4 py-3 border-t dark:border-gray-700">
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
              Try asking about:
            </p>
            <div className="flex flex-wrap gap-2">
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  className="flex items-center space-x-1 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 px-3 py-1.5 rounded-full text-sm transition-colors"
                  onClick={() => handleSendMessage(suggestion)}
                >
                  <span>{suggestion}</span>
                  <ChevronRight className="h-4 w-4" />
                </button>
              ))}
            </div>
          </div>
        )}
        
        {/* Input area */}
        <div className="p-4 border-t dark:border-gray-700">
          <div className="relative">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder="Type your message here..."
              className="w-full border border-gray-300 dark:border-gray-600 rounded-lg py-2 pl-4 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800"
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={isLoading || !inputValue.trim()}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 rounded-md bg-blue-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
          <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
            LogixSense AI can analyze shipment data, identify trends, optimize routes, and more. Your interactions help improve the system.
          </p>
        </div>
      </Card>
    </div>
  );
}