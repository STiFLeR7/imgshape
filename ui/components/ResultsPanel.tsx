import React, { useState } from 'react';
import { Code2, AlignLeft } from 'lucide-react';

interface ResultsPanelProps {
  data: any | null;
  status: 'idle' | 'loading' | 'success' | 'error';
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({ data, status }) => {
  const [view, setView] = useState<'pretty' | 'raw'>('pretty');

  const renderContent = () => {
    if (status === 'idle') {
      return (
        <div className="h-full flex flex-col items-center justify-center text-gray-600 space-y-3">
          <Code2 className="w-12 h-12 opacity-20" />
          <p className="text-sm">Ready for analysis</p>
        </div>
      );
    }

    if (status === 'loading') {
      return (
        <div className="h-full flex items-center justify-center">
            <div className="flex flex-col items-center space-y-3 animate-pulse">
                <div className="w-full max-w-md h-4 bg-gray-800 rounded"></div>
                <div className="w-full max-w-sm h-4 bg-gray-800 rounded"></div>
                <div className="w-full max-w-lg h-4 bg-gray-800 rounded"></div>
            </div>
        </div>
      );
    }

    if (!data) return null;

    if (view === 'raw') {
      return (
        <pre className="p-4 text-xs font-mono text-gray-300 whitespace-pre-wrap break-all leading-5">
          {JSON.stringify(data, null, 2)}
        </pre>
      );
    }

    return (
      <div className="p-4 text-xs font-mono">
         <JsonTree data={data} />
      </div>
    );
  };

  return (
    <div className="flex-1 bg-surface border border-gray-800 rounded-xl overflow-hidden shadow-sm flex flex-col min-h-[400px]">
      <div className="px-4 py-2 border-b border-gray-800 bg-surfaceHighlight/30 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Analysis Results</h3>
        <div className="flex bg-gray-900 rounded-lg p-0.5 border border-gray-700">
          <button
            onClick={() => setView('pretty')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
              view === 'pretty' ? 'bg-gray-700 text-white shadow-sm' : 'text-gray-400 hover:text-white'
            }`}
          >
            Pretty
          </button>
          <button
            onClick={() => setView('raw')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
              view === 'raw' ? 'bg-gray-700 text-white shadow-sm' : 'text-gray-400 hover:text-white'
            }`}
          >
            Raw
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-auto custom-scrollbar bg-[#0d1117]">
        {renderContent()}
      </div>
    </div>
  );
};

// Simple Recursive JSON Tree Component for "Pretty" view
const JsonTree = ({ data, level = 0 }: { data: any, level?: number }) => {
  const indent = level * 12;

  if (typeof data !== 'object' || data === null) {
    let color = 'text-purple-400';
    if (typeof data === 'string') color = 'text-green-400';
    if (typeof data === 'number') color = 'text-orange-400';
    if (typeof data === 'boolean') color = 'text-blue-400';
    
    return (
        <span className={`${color} break-words`}>
            {JSON.stringify(data)}
        </span>
    );
  }

  const isArray = Array.isArray(data);
  const keys = Object.keys(data);
  
  if (keys.length === 0) return <span className="text-gray-500">{isArray ? '[]' : '{}'}</span>;

  return (
    <div style={{ marginLeft: level > 0 ? '12px' : '0' }}>
      <span className="text-gray-500">{isArray ? '[' : '{'}</span>
      {keys.map((key, index) => (
        <div key={key} className="pl-2">
           {!isArray && <span className="text-cyan-300 mr-1">"{key}":</span>}
           <JsonTree data={data[key]} level={0} />
           {index < keys.length - 1 && <span className="text-gray-500">,</span>}
        </div>
      ))}
      <span className="text-gray-500">{isArray ? ']' : '}'}</span>
    </div>
  );
}

export default ResultsPanel;