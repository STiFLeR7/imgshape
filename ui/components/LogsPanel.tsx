import React, { useEffect, useRef } from 'react';
import { Terminal } from 'lucide-react';
import { LogEntry } from '../types';

interface LogsPanelProps {
  logs: LogEntry[];
}

const LogsPanel: React.FC<LogsPanelProps> = ({ logs }) => {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="bg-surface border border-gray-800 rounded-xl overflow-hidden shadow-sm flex flex-col h-48 md:h-64">
      <div className="px-4 py-2 border-b border-gray-800 bg-surfaceHighlight/50 flex items-center">
        <Terminal className="w-4 h-4 mr-2 text-gray-400" />
        <h3 className="text-xs font-mono uppercase tracking-wider text-gray-400">System Logs</h3>
      </div>
      <div className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-1.5 custom-scrollbar bg-black/40">
        {logs.length === 0 ? (
            <div className="text-gray-600 italic opacity-50">Waiting for system events...</div>
        ) : (
            logs.map((log) => (
            <div key={log.id} className="flex items-start space-x-3 hover:bg-white/5 p-0.5 rounded px-2 -mx-2 transition-colors">
                <span className="text-gray-600 shrink-0">{log.timestamp}</span>
                <span className={`shrink-0 w-16 text-center rounded px-1 text-[10px] font-bold uppercase tracking-wide
                ${log.level === 'info' ? 'bg-cyan-900/30 text-cyan-400' : ''}
                ${log.level === 'success' ? 'bg-green-900/30 text-green-400' : ''}
                ${log.level === 'warning' ? 'bg-orange-900/30 text-orange-400' : ''}
                ${log.level === 'error' ? 'bg-red-900/30 text-red-400' : ''}
                `}>
                {log.level}
                </span>
                <span className={`break-all ${log.level === 'error' ? 'text-red-300' : 'text-gray-300'}`}>
                {log.message}
                </span>
            </div>
            ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
};

export default LogsPanel;