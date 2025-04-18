'use client';

import React from 'react';
import LogixAssistant from '@/components/LogixAssistant';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Sparkles, Info, BookOpen, BrainCircuit, Database, Server } from 'lucide-react';

const AssistantPage = () => {
  return (
    <div className="container mx-auto p-4 h-[calc(100vh-4rem)]">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">AI Assistant</h1>
        <p className="text-muted-foreground">
          Get comprehensive insights from your logistics data through enhanced natural language processing
        </p>
      </div>

      <Tabs defaultValue="assistant" className="h-[calc(100%-6rem)]">
        <TabsList className="mb-4">
          <TabsTrigger value="assistant">
            <Sparkles className="h-4 w-4 mr-2" />
            Assistant
          </TabsTrigger>
          <TabsTrigger value="about">
            <Info className="h-4 w-4 mr-2" />
            About
          </TabsTrigger>
          <TabsTrigger value="guide">
            <BookOpen className="h-4 w-4 mr-2" />
            User Guide
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="assistant" className="h-full">
          <LogixAssistant />
        </TabsContent>
        
        <TabsContent value="about">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BrainCircuit className="mr-2 h-5 w-5 text-primary" />
                About LogixSense Enhanced AI Assistant
              </CardTitle>
              <CardDescription>
                Understand how the AI Assistant works and what it can do for you
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">What is LogixSense Enhanced AI Assistant?</h3>
                <p>
                  LogixSense AI Assistant is an intelligent natural language interface that helps you analyze and 
                  understand your logistics data. The enhanced version now processes your entire database for each query,
                  providing comprehensive analysis and deeper insights than ever before.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Enhanced Features</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                  <Card className="bg-[#0A0A0A] border border-[#222]">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center">
                        <Database className="h-4 w-4 mr-2 text-cyan-400" />
                        Full Database Analysis
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm pt-0">
                      <p>Analyzes your entire database for each query, not just samples or similar records, providing more accurate and comprehensive insights.</p>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-[#0A0A0A] border border-[#222]">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center">
                        <BrainCircuit className="h-4 w-4 mr-2 text-purple-400" />
                        Specialized Analytics
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm pt-0">
                      <p>Dedicated analysis engines for different data categories like destinations, weights, carriers, and time trends.</p>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-[#0A0A0A] border border-[#222]">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center">
                        <Server className="h-4 w-4 mr-2 text-green-400" />
                        Comprehensive Context
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm pt-0">
                      <p>Provides the LLM with complete database statistics and comprehensive analytics for more insightful responses.</p>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-[#0A0A0A] border border-[#222]">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center">
                        <Sparkles className="h-4 w-4 mr-2 text-pink-400" />
                        Richer Visualizations
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm pt-0">
                      <p>Data visualizations are now based on complete dataset analysis rather than limited samples.</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Technology Stack</h3>
                <p>
                  The enhanced assistant is powered by:
                </p>
                <ul className="list-disc pl-6 mt-2">
                  <li><strong>Ollama with Mistral 7B</strong> - For natural language understanding and generation</li>
                  <li><strong>FAISS Vector Database</strong> - For efficient similarity search of logistics data</li>
                  <li><strong>LogisticsAnalyzer</strong> - New component for comprehensive database analysis</li>
                  <li><strong>Sentence Transformers</strong> - For generating embeddings of text data</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Data Privacy & Security</h3>
                <p>
                  The AI Assistant processes all data locally. Your logistics data never leaves your servers, 
                  as all processing happens on-premise. The system uses your own deployment of Ollama, 
                  ensuring complete data privacy.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="guide">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BookOpen className="mr-2 h-5 w-5 text-primary" />
                LogixSense Enhanced AI Assistant Guide
              </CardTitle>
              <CardDescription>
                Learn how to use the AI Assistant effectively
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-medium mb-2">Getting Started</h3>
                <ol className="list-decimal pl-6">
                  <li className="mb-2">
                    <strong>Initialize the Vector Database</strong>: Before using the assistant for the first time, 
                    click the "Initialize DB" button in the top right corner. This will process your logistics data 
                    and make it searchable.
                  </li>
                  <li className="mb-2">
                    <strong>Check the Status Indicator</strong>: The badge in the top right shows if the assistant is online and ready.
                    Look for the "ENHANCED AI" badge which indicates comprehensive database analysis is available.
                  </li>
                  <li className="mb-2">
                    <strong>Ask Your First Question</strong>: Type a question in the input field at the bottom of the chat and press Enter or click Send.
                  </li>
                </ol>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Example Questions for Enhanced Analysis</h3>
                <p>Try asking questions like:</p>
                <ul className="list-disc pl-6 mt-2">
                  <li>"Analyze our destinations and show the distribution of shipments"</li>
                  <li>"Show me a comprehensive breakdown of package weights across different carriers"</li>
                  <li>"Perform a complete analysis of our shipping performance over time"</li>
                  <li>"Compare all carriers by total weight shipped and delivery times"</li>
                  <li>"What are the risk factors across our entire logistics operation?"</li>
                  <li>"Give me insights on the relationship between commodity types and destinations"</li>
                  <li>"Analyze monthly trends across all shipping metrics"</li>
                  <li>"What is the financial impact of our shipping weight distribution?"</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Tips for Better Analysis</h3>
                <ul className="list-disc pl-6">
                  <li>
                    <strong>Use Analysis Categories</strong>: Ask for analysis in specific categories like "destinations", 
                    "weights", "carriers", "time trends", or "risks" to get specialized insights.
                  </li>
                  <li>
                    <strong>Request Comparative Analysis</strong>: Ask the assistant to "compare" or "analyze the relationship between" 
                    different factors for deeper insights.
                  </li>
                  <li>
                    <strong>Ask for Comprehensive Analysis</strong>: Using phrases like "analyze", "comprehensive breakdown", or 
                    "complete analysis" triggers the enhanced analysis features.
                  </li>
                  <li>
                    <strong>Visualization Options</strong>: After getting a response, you can request different visualization types 
                    (bar, line, pie charts) to better understand the data.
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Troubleshooting</h3>
                <ul className="list-disc pl-6">
                  <li>
                    <strong>Enhanced AI Unavailable</strong>: If you don't see the "ENHANCED AI" badge, try clicking the refresh button
                    or reinitializing the database.
                  </li>
                  <li>
                    <strong>Slow Analysis</strong>: Comprehensive database analysis may take slightly longer than basic queries,
                    especially for large datasets.
                  </li>
                  <li>
                    <strong>Analysis Limitations</strong>: The assistant can only analyze data fields that exist in your 
                    logistics database schema.
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AssistantPage;