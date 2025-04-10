'use client';

import React from 'react';
import LogixAssistant from '@/components/LogixAssistant';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Sparkles, Info, BookOpen } from 'lucide-react';

const AssistantPage = () => {
  return (
    <div className="container mx-auto p-4 h-[calc(100vh-4rem)]">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">AI Assistant</h1>
        <p className="text-muted-foreground">
          Get insights from your logistics data through natural language queries
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
                <Sparkles className="mr-2 h-5 w-5 text-primary" />
                About LogixSense AI Assistant
              </CardTitle>
              <CardDescription>
                Understand how the AI Assistant works and what it can do for you
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">What is LogixSense AI Assistant?</h3>
                <p>
                  LogixSense AI Assistant is an intelligent natural language interface that helps you analyze and 
                  understand your logistics data. It uses advanced AI technologies to interpret your questions, 
                  retrieve relevant information, and provide insightful answers.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Technology Stack</h3>
                <p>
                  The assistant is powered by:
                </p>
                <ul className="list-disc pl-6 mt-2">
                  <li><strong>Ollama with Mistral 7B</strong> - For natural language understanding and generation</li>
                  <li><strong>FAISS Vector Database</strong> - For efficient similarity search of logistics data</li>
                  <li><strong>Sentence Transformers</strong> - For generating embeddings of text data</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Features</h3>
                <ul className="list-disc pl-6">
                  <li>Natural language queries about your logistics data</li>
                  <li>Contextual responses based on your historical shipment information</li>
                  <li>Data-driven insights and analysis</li>
                  <li>Trend identification and anomaly detection</li>
                  <li>Performance metrics and KPI summaries</li>
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
                LogixSense AI Assistant Guide
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
                  </li>
                  <li className="mb-2">
                    <strong>Ask Your First Question</strong>: Type a question in the input field at the bottom of the chat and press Enter or click Send.
                  </li>
                </ol>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Example Questions</h3>
                <p>Try asking questions like:</p>
                <ul className="list-disc pl-6 mt-2">
                  <li>"What are our top shipping destinations?"</li>
                  <li>"Show me the distribution of package weights"</li>
                  <li>"Analyze shipping performance for the last month"</li>
                  <li>"Which carriers have the best delivery times?"</li>
                  <li>"What commodities do we ship most frequently?"</li>
                  <li>"Are there any delivery delays I should be aware of?"</li>
                  <li>"Compare shipping volumes between different countries"</li>
                  <li>"What's the average weight of shipments to Europe?"</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Tips for Better Results</h3>
                <ul className="list-disc pl-6">
                  <li>
                    <strong>Be Specific</strong>: Include specific metrics, time periods, or regions in your questions 
                    for more targeted answers.
                  </li>
                  <li>
                    <strong>One Topic at a Time</strong>: For complex analyses, break down your questions into smaller, 
                    focused queries.
                  </li>
                  <li>
                    <strong>Follow-up Questions</strong>: You can ask follow-up questions to dig deeper into specific aspects 
                    of the previous response.
                  </li>
                  <li>
                    <strong>Data Limitations</strong>: The assistant can only analyze data that's available in your 
                    logistics database.
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Troubleshooting</h3>
                <ul className="list-disc pl-6">
                  <li>
                    <strong>Assistant Offline</strong>: If the status shows "Offline," try clicking the refresh button. 
                    If it remains offline, check if the backend services are running.
                  </li>
                  <li>
                    <strong>Inaccurate Responses</strong>: If responses seem inaccurate, try reinitializing the vector 
                    database, especially if your logistics data has been updated recently.
                  </li>
                  <li>
                    <strong>Slow Responses</strong>: Complex queries involving large amounts of data may take longer to process. 
                    Please be patient during peak usage times.
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