import React, { useState } from 'react';
import { Code2, AlignLeft, FileText } from 'lucide-react';

interface ResultsPanelProps {
  data: any | null;
  status: 'idle' | 'loading' | 'success' | 'error';
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({ data, status }) => {
  const [view, setView] = useState<'pretty' | 'raw' | 'text'>('pretty');

  const toText = (data: any, indent = 0): string => {
    const spaces = ' '.repeat(indent);
    if (typeof data !== 'object' || data === null) {
      return String(data);
    }
    if (Array.isArray(data)) {
      return data.map((item, i) => `${spaces}- [${i}]: ${toText(item, indent + 2).trim()}`).join('\n');
    }
    return Object.entries(data).map(([key, value]) => {
        if (typeof value === 'object' && value !== null) {
            // Check if it's empty
            if (Object.keys(value).length === 0) return `${spaces}${key}: {}`;
            return `${spaces}${key}:\n${toText(value, indent + 2)}`;
        }
        return `${spaces}${key}: ${value}`;
    }).join('\n');
  };

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

    // v4.2 Specialized Views
    const profiles = data.profiles || {};
    const medical = profiles.medical;
    const satellite = profiles.satellite;
    const ocr = profiles.ocr;
    const semantic = profiles.semantic;

    if (view === 'pretty') {
        return (
            <div className="p-6 space-y-6">
                {/* Domain Specialized Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {medical && (
                        <div className="bg-blue-900/10 border border-blue-500/30 rounded-lg p-4">
                            <h4 className="text-blue-400 text-xs font-bold uppercase mb-2 flex items-center gap-2">
                                🩺 Medical Profile
                            </h4>
                            <div className="grid grid-cols-2 gap-2 text-[10px]">
                                <div className="text-gray-400">HU Range</div>
                                <div className="text-white font-mono">[{medical.hu_range?.join(', ')}]</div>
                                <div className="text-gray-400">Slice Consistency</div>
                                <div className="text-white font-mono">{(medical.slice_consistency * 100).toFixed(1)}%</div>
                            </div>
                        </div>
                    )}
                    {satellite && (
                        <div className="bg-emerald-900/10 border border-emerald-500/30 rounded-lg p-4">
                            <h4 className="text-emerald-400 text-xs font-bold uppercase mb-2 flex items-center gap-2">
                                🛰️ Satellite Profile
                            </h4>
                            <div className="grid grid-cols-2 gap-2 text-[10px]">
                                <div className="text-gray-400">GSD Estimate</div>
                                <div className="text-white font-mono">{satellite.gsd_estimate?.toFixed(3)}m</div>
                                <div className="text-gray-400">Cloud Cover</div>
                                <div className="text-white font-mono">{(satellite.cloud_cover_estimate * 100).toFixed(1)}%</div>
                            </div>
                        </div>
                    )}
                    {ocr && (
                        <div className="bg-orange-900/10 border border-orange-500/30 rounded-lg p-4">
                            <h4 className="text-orange-400 text-xs font-bold uppercase mb-2 flex items-center gap-2">
                                📝 OCR Profile
                            </h4>
                            <div className="grid grid-cols-2 gap-2 text-[10px]">
                                <div className="text-gray-400">Text Density</div>
                                <div className="text-white font-mono">{(ocr.text_density * 100).toFixed(1)}%</div>
                                <div className="text-gray-400">Orientation Var</div>
                                <div className="text-white font-mono">{ocr.orientation_variance?.toFixed(4)}</div>
                            </div>
                        </div>
                    )}
                    {semantic && semantic.latent_embedding && (
                        <div className="bg-purple-900/10 border border-purple-500/30 rounded-lg p-4 col-span-full">
                            <h4 className="text-purple-400 text-xs font-bold uppercase mb-2 flex items-center gap-2">
                                🧠 Semantic Signature
                            </h4>
                            <div className="space-y-2">
                                <div className="text-[10px] text-gray-400">Latent Centroid (first 16 components)</div>
                                <div className="flex flex-wrap gap-1">
                                    {semantic.latent_embedding.slice(0, 16).map((val: number, i: number) => (
                                        <span key={i} className="px-1.5 py-0.5 bg-purple-500/20 rounded text-[9px] text-purple-300 font-mono">
                                            {val.toFixed(3)}
                                        </span>
                                    ))}
                                    <span className="text-gray-600 text-[9px]">...</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Fallback to tree for rest of data */}
                <div className="text-[10px] font-mono border-t border-gray-800 pt-4">
                    <JsonTree data={data} />
                </div>
            </div>
        );
    }

    if (view === 'raw') {
      return (
        <pre className="p-4 text-xs font-mono text-gray-300 whitespace-pre-wrap break-all leading-5">
          {JSON.stringify(data, null, 2)}
        </pre>
      );
    }

    if (view === 'text') {
        return (
            <pre className="p-4 text-xs font-mono text-gray-300 whitespace-pre-wrap break-all leading-5">
              {toText(data)}
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
    <div className="flex-1 bg-surface border border-gray-800 rounded-xl overflow-hidden shadow-sm flex flex-col min-h-[400px] relative z-0">
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
          <button
            onClick={() => setView('text')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
              view === 'text' ? 'bg-gray-700 text-white shadow-sm' : 'text-gray-400 hover:text-white'
            }`}
          >
            Text
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