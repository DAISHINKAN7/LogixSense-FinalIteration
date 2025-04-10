import React, { useState, useEffect, useRef } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle,
  CardDescription 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Avatar } from '@/components/ui/avatar';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { 
  AlertCircle, 
  Send, 
  Sparkles, 
  Bot, 
  User, 
  RefreshCw,
  CornerDownLeft,
  Loader,
  Lightbulb,
  MessageSquare
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  Tooltip, 
  TooltipContent, 
  TooltipTrigger 
} from '@/components/ui/tooltip';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Markdown components customization
const MarkdownComponents = {
  h1: props => <h1 className="text-2xl font-bold my-4" {...props} />,
  h2: props => <h2 className="text-xl font-bold my-3" {...props} />,
  h3: props => <h3 className="text-lg font-bold my-2" {...props} />,
  p: props => <p className="mb-4" {...props} />,
  ul: props => <ul className="list-disc pl-6 mb-4" {...props} />,
  ol: props => <ol className="list-decimal pl-6 mb-4" {...props} />,
  li: props => <li className="mb-1" {...props} />,
  table: props => <table className="border-collapse table-auto w-full mb-4" {...props} />,
  th: props => <th className="border border-gray-300 dark:border-gray-700 px-4 py-2 text-left" {...props} />,
  td: props => <td className="border border-gray-300 dark:border-gray-700 px-4 py-2" {...props} />,
  code: props => <code className="bg-gray-100 dark:bg-gray-800 rounded p-1 font-mono text-sm" {...props} />,
  pre: props => <pre className="bg-gray-100 dark:bg-gray-800 rounded p-4 overflow-x-auto mb-4 font-mono text-sm" {...props} />
};

// Message component for chat
const MessageBubble = ({ message, isLoading }) => {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} max-w-[85%]`}>
        <div className="flex-shrink-0">
          <Avatar className={`${isUser ? 'ml-3' : 'mr-3'} h-8 w-8`}>
            {isUser ? <User size={16} /> : <Bot size={16} />}
          </Avatar>
        </div>
        
        <div 
          className={`rounded-lg px-4 py-2 ${
            isUser 
            ? 'bg-primary text-primary-foreground' 
            : 'bg-muted text-foreground'
          }`}
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <Loader size={14} className="animate-spin" />
              <span>Thinking...</span>
            </div>
          ) : (
            <div className="prose dark:prose-invert max-w-none">
              <ReactMarkdown 
                remarkPlugins={[remarkGfm]} 
                components={MarkdownComponents}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
          
          <div className={`mt-1 text-xs opacity-70 text-right`}>
            {message.timestamp}
          </div>
        </div>
      </div>
    </div>
  );
};

// Suggestion pill component
const SuggestionPill = ({ suggestion, onClick }) => {
  return (
    <Button 
      variant="outline" 
      className="mr-2 mb-2 text-sm whitespace-nowrap"
      onClick={() => onClick(suggestion)}
    >
      <Lightbulb size={14} className="mr-1" />
      {suggestion.length > 40 ? `${suggestion.substring(0, 40)}...` : suggestion}
    </Button>
  );
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

  // Handle sending a message to the assistant
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
    
    // Add temporary assistant message
    const tempAssistantMessage = {
      role: 'assistant',
      content: '',
      timestamp: getFormattedTime()
    };
    
    setMessages(prev => [...prev, tempAssistantMessage]);
    
    try {
      // Send the query to the API
      const response = await fetch('/api/assistant/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          options: {
            detailed: true
          }
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get response');
      }
      
      const data = await response.json();
      
      // Update the last message (the temporary one) with the assistant's response
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: data.answer,
          timestamp: getFormattedTime()
        };
        return updated;
      });
      
    } catch (err) {
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
          content: `Vector database initialized successfully with ${data.message}`,
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
          content: "👋 Hello! I'm your LogixSense AI Assistant. I can help you analyze your logistics data, answer questions about shipments, and provide insights on your supply chain operations. How can I assist you today?",
          timestamp: getFormattedTime()
        }
      ]);
    }
  }, [messages.length]);

  return (
    <Card className="w-full h-full flex flex-col">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="flex items-center">
              <Sparkles className="mr-2 h-5 w-5 text-primary" />
              LogixSense AI Assistant
            </CardTitle>
            <CardDescription>
              Ask questions about your logistics data and get AI-powered insights
            </CardDescription>
          </div>
          
          <div className="flex items-center space-x-2">
            {assistantStatus ? (
              <Tooltip>
                <TooltipTrigger asChild>
                <Badge variant={assistantStatus.status === 'online' ? 'outline' : 'destructive'}>
                    {assistantStatus.status === 'online' 
                        ? 'Online' 
                        : assistantStatus.status === 'error' 
                        ? 'Connection Error' 
                        : 'Offline'}
                </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  {assistantStatus.status === 'online' 
                    ? `Assistant is online and ready to help` 
                    : `Assistant is currently unavailable`}
                </TooltipContent>
              </Tooltip>
            ) : (
              <Badge variant="outline">Checking status...</Badge>
            )}
            
            <Button 
              variant="outline" 
              size="sm" 
              onClick={checkAssistantStatus}
              disabled={isLoading}
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
                >
                  Initialize DB
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                Initialize the vector database with your logistics data
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="flex-grow flex flex-col h-[calc(100%-10rem)] pb-0">
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        <ScrollArea className="flex-grow pr-4 mb-4">
          <div className="space-y-4">
            {messages.map((message, index) => (
              <MessageBubble 
                key={index} 
                message={message} 
                isLoading={isLoading && index === messages.length - 1 && message.role === 'assistant'} 
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
        
        {/* Suggestions */}
        {suggestions.length > 0 && messages.length <= 2 && (
          <div className="mb-4">
            <Label className="text-sm mb-2 flex items-center">
              <MessageSquare className="h-4 w-4 mr-1" />
              Suggested Questions
            </Label>
            <div className="flex flex-wrap">
              {suggestions.slice(0, 4).map((suggestion, index) => (
                <SuggestionPill 
                  key={index} 
                  suggestion={suggestion} 
                  onClick={handleSuggestionClick} 
                />
              ))}
            </div>
          </div>
        )}
        
        <div className="pt-4 pb-4 border-t">
          <div className="flex space-x-2">
            <Textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask me about your logistics data..."
                className="flex-grow min-h-12 resize-none"
                disabled={isLoading || assistantStatus?.status !== 'online'}
            />
            <Button 
              onClick={handleSendMessage} 
              disabled={!input.trim() || isLoading || assistantStatus?.status !== 'online'}
              className="self-end"
            >
              {isLoading ? (
                <Loader size={16} className="animate-spin" />
              ) : (
                <>
                  <Send size={16} className="mr-1" />
                  Send
                </>
              )}
            </Button>
          </div>
          <div className="mt-2 text-xs text-muted-foreground flex items-center justify-between">
            <div>
              Press <kbd className="rounded border px-1 py-0.5 bg-muted">Enter</kbd> to send,
              <kbd className="rounded border px-1 py-0.5 bg-muted ml-1">Shift + Enter</kbd> for new line
            </div>
            {assistantStatus?.details?.vector_db && (
              <span>
                {assistantStatus.details.vector_db.document_count} documents indexed
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default LogixAssistant;