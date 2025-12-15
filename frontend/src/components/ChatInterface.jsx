import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Activity } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import ReasoningAccordion from './ReasoningAccordion';

export default function ChatInterface() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = { role: 'user', content: input };
        const botMessageId = Date.now();
        const initialBotMessage = {
            id: botMessageId,
            role: 'assistant',
            content: '',
            logs: [],
            sources: [],
            isThinking: true
        };

        setMessages(prev => [...prev, userMessage, initialBotMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: userMessage.content }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const text = decoder.decode(value);
                const lines = text.split('\n');

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        setMessages(prev => prev.map(msg => {
                            if (msg.id !== botMessageId) return msg;

                            if (data.type === 'log') {
                                // Avoid duplicates if necessary, but simple push is okay
                                return { ...msg, logs: [...msg.logs, data.content] };
                            } else if (data.type === 'final') {
                                return {
                                    ...msg,
                                    content: data.data.answer,
                                    sources: data.data.sources,
                                    isThinking: false
                                };
                            } else if (data.type === 'error') {
                                return { ...msg, content: `Error: ${data.content}`, isThinking: false };
                            }
                            return msg;
                        }));
                    } catch (e) { console.error(e); }
                }
            }
        } catch (error) {
            setMessages(prev => prev.map(msg => {
                if (msg.id !== botMessageId) return msg;
                return { ...msg, content: `Connection error. Please ensure backend is running.`, isThinking: false };
            }));
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-screen max-w-5xl mx-auto border-x border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm relative">

            {/* Header - Clean & Minimal */}
            <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-800 bg-white dark:bg-slate-900 sticky top-0 z-10 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-slate-100 dark:bg-slate-800 rounded-md text-slate-700 dark:text-slate-200">
                        <Activity size={20} strokeWidth={1.5} />
                    </div>
                    <div>
                        <h1 className="font-semibold text-lg text-slate-900 dark:text-slate-100 tracking-tight leading-none">MEGA-RAG</h1>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Medical Evidence Analysis System</p>
                    </div>
                </div>
                <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-50 dark:bg-slate-800 border border-slate-100 dark:border-slate-700">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
                    <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-500 dark:text-slate-400">System Active</span>
                </div>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-8 bg-white dark:bg-slate-900 scroll-smooth">
                {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-[70vh] text-slate-400 gap-4">
                        <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-xl border border-slate-100 dark:border-slate-700">
                            <Bot size={32} strokeWidth={1} />
                        </div>
                        <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Ready to analyze medical queries</p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-4 max-w-3xl ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}>

                        {/* Avatar */}
                        <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center border ${msg.role === 'user'
                            ? 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'
                            : 'bg-blue-600 border-blue-600 text-white'
                            }`}>
                            {msg.role === 'user' ? <User size={16} strokeWidth={2} /> : <Bot size={16} strokeWidth={2} />}
                        </div>

                        {/* Content Container */}
                        <div className="flex flex-col gap-2 min-w-0 max-w-full">

                            {msg.role === 'assistant' && msg.logs && msg.logs.length > 0 && (
                                <ReasoningAccordion logs={msg.logs} />
                            )}

                            <div className={`px-5 py-3.5 rounded-lg text-sm leading-relaxed shadow-sm border ${msg.role === 'user'
                                ? 'bg-slate-50 dark:bg-slate-800 border-slate-100 dark:border-slate-700 text-slate-800 dark:text-slate-100'
                                : 'bg-white dark:bg-slate-900 border-slate-100 dark:border-slate-800 text-slate-700 dark:text-slate-300'
                                }`}>
                                {msg.isThinking && !msg.content ? (
                                    <div className="flex gap-1 items-center h-5">
                                        <span className="w-1 h-1 bg-slate-400 rounded-full animate-pulse"></span>
                                        <span className="w-1 h-1 bg-slate-400 rounded-full animate-pulse delay-75"></span>
                                        <span className="w-1 h-1 bg-slate-400 rounded-full animate-pulse delay-150"></span>
                                        <span className="ml-2 text-slate-400 text-xs font-medium">Processing...</span>
                                    </div>
                                ) : (
                                    <div className="prose prose-sm prose-slate dark:prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-slate-800 prose-pre:border prose-pre:border-slate-700">
                                        <ReactMarkdown>
                                            {msg.content}
                                        </ReactMarkdown>
                                    </div>
                                )}
                            </div>

                            {/* Citations Footer */}
                            {msg.sources && msg.sources.length > 0 && (
                                <div className="mt-1 flex flex-wrap gap-2">
                                    {msg.sources.map((src, i) => (
                                        <div key={i} className="flex items-center gap-1.5 px-2 py-1 bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800 rounded text-[11px] text-slate-500 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-default">
                                            <span className="text-[10px]">📄</span>
                                            {src.replace('.pdf', '')}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {/* Input - Footer */}
            <div className="p-6 bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800">
                <form onSubmit={handleSubmit} className="relative max-w-3xl mx-auto">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Enter medical query..."
                        className="w-full p-4 pr-14 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all text-sm font-medium placeholder:text-slate-400 shadow-sm"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 top-2 bottom-2 aspect-square flex items-center justify-center bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors disabled:opacity-50 disabled:bg-slate-300"
                    >
                        <Send size={18} strokeWidth={2} />
                    </button>
                </form>
                <div className="mt-3 flex justify-center gap-6 text-[10px] text-slate-400 font-medium tracking-wide uppercase">
                    <span>Evidence-Based</span>
                    <span>•</span>
                    <span>Clinical Verification</span>
                    <span>•</span>
                    <span>Privacy Focused</span>
                </div>
            </div>
        </div>
    );
}
