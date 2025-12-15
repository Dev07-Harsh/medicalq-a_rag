import React, { useState } from 'react';
import { ChevronDown, ChevronRight, FileText, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';

export default function ReasoningAccordion({ logs }) {
    const [isOpen, setIsOpen] = useState(false);

    // Helper to parse log type for icon/color
    const getLogStyle = (text) => {
        if (text.includes("[RETRIEVE]")) return { icon: <FileText size={14} />, color: "text-blue-600 bg-blue-50 border-blue-100" };
        if (text.includes("✅")) return { icon: <CheckCircle2 size={14} />, color: "text-emerald-600 bg-emerald-50 border-emerald-100" };
        if (text.includes("⚠️") || text.includes("unsupported")) return { icon: <AlertTriangle size={14} />, color: "text-amber-600 bg-amber-50 border-amber-100" };
        if (text.includes("[CORRECT]")) return { icon: <RefreshCw size={14} />, color: "text-purple-600 bg-purple-50 border-purple-100" };
        return { icon: <div className="w-1.5 h-1.5 rounded-full bg-slate-400" />, color: "text-slate-600 bg-slate-50 border-slate-100" };
    };

    return (
        <div className="border border-slate-200 dark:border-slate-700 rounded-md overflow-hidden bg-white dark:bg-slate-900 shadow-sm mb-3">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors"
            >
                <div className="flex items-center gap-2">
                    <span>REASONING PROCESS</span>
                    {isOpen ? (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-200 text-slate-600">Active</span>
                    ) : (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-200 text-slate-600">{logs.length} Steps</span>
                    )}
                </div>
                {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </button>

            {isOpen && (
                <div className="p-3 bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800">
                    <div className="space-y-2 font-mono text-[11px] leading-relaxed">
                        {logs.map((log, index) => {
                            const { icon, color } = getLogStyle(log);
                            return (
                                <div key={index} className="flex gap-3 items-start group">
                                    <div className={`mt-0.5 flex-shrink-0 p-1 rounded border ${color} dark:bg-opacity-10 dark:border-opacity-20`}>
                                        {icon}
                                    </div>
                                    <span className="text-slate-600 dark:text-slate-400 pt-0.5 break-words">
                                        {log}
                                    </span>
                                </div>
                            );
                        })}
                        {logs.length === 0 && (
                            <div className="text-slate-400 italic pl-1">Animating reasoning trace...</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
