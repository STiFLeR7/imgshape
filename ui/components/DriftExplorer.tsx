import React, { useState } from 'react';
import { GitCompare, AlertTriangle, CheckCircle, Info, Brain } from 'lucide-react';

interface DriftExplorerProps {
  basePath: string;
  currPath: string;
  results: any | null;
  onCalculate: (base: string, curr: string) => void;
  isLoading: boolean;
}

const DriftExplorer: React.FC<DriftExplorerProps> = ({ 
  basePath, 
  currPath, 
  results, 
  onCalculate, 
  isLoading 
}) => {
  const [base, setBase] = useState(basePath);
  const [curr, setCurr] = useState(currPath);

  const drift = results?.drift;

  return (
    <div className="flex flex-col h-full space-y-6">
      <div className="bg-surface border border-gray-800 rounded-xl p-6 shadow-sm">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-accent" />
          Atlas Drift Explorer
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="space-y-2">
            <label className="text-xs font-semibold text-gray-400 uppercase">Baseline Dataset</label>
            <input 
              type="text" 
              value={base}
              onChange={(e) => setBase(e.target.value)}
              placeholder="/data/baseline"
              className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-accent outline-none text-sm"
            />
          </div>
          <div className="space-y-2">
            <label className="text-xs font-semibold text-gray-400 uppercase">Current Dataset</label>
            <input 
              type="text" 
              value={curr}
              onChange={(e) => setCurr(e.target.value)}
              placeholder="/data/current"
              className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-accent outline-none text-sm"
            />
          </div>
        </div>

        <button
          onClick={() => onCalculate(base, curr)}
          disabled={isLoading || !base || !curr}
          className="w-full bg-accent hover:bg-accentHover disabled:bg-gray-700 py-2.5 rounded-lg text-sm font-bold flex items-center justify-center gap-2 transition-all"
        >
          {isLoading ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <GitCompare className="w-4 h-4" />}
          Run Semantic Comparison
        </button>
      </div>

      {drift && (
        <div className="flex-1 space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {/* Status Header */}
          <div className={`p-4 rounded-xl border flex items-center gap-4 ${drift.is_significant ? 'bg-red-900/10 border-red-500/30' : 'bg-green-900/10 border-green-500/30'}`}>
            <div className={`p-3 rounded-full ${drift.is_significant ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500'}`}>
              {drift.is_significant ? <AlertTriangle className="w-6 h-6" /> : <CheckCircle className="w-6 h-6" />}
            </div>
            <div>
              <h3 className={`font-bold ${drift.is_significant ? 'text-red-400' : 'text-green-400'}`}>
                {drift.is_significant ? 'Significant Drift Detected' : 'Dataset Stable'}
              </h3>
              <p className="text-xs text-gray-400">
                Overall drift score: <span className="font-mono font-bold text-gray-200">{(drift.overall_drift * 100).toFixed(1)}%</span>
              </p>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricCard 
              label="Semantic (Embeddings)" 
              value={drift.semantic_drift} 
              icon={<Brain className="w-4 h-4" />} 
              color="purple" 
            />
            <MetricCard 
              label="Signal (Color/Entropy)" 
              value={drift.signal_drift} 
              icon={<Info className="w-4 h-4" />} 
              color="blue" 
            />
            <MetricCard 
              label="Spatial (Resolution)" 
              value={drift.spatial_drift} 
              icon={<Info className="w-4 h-4" />} 
              color="emerald" 
            />
          </div>

          {/* Rationale */}
          <div className="bg-surface border border-gray-800 rounded-xl overflow-hidden shadow-sm">
            <div className="px-4 py-3 border-b border-gray-800 bg-surfaceHighlight/30 text-xs font-bold text-gray-400 uppercase">
              Drift Rationale
            </div>
            <div className="p-4 space-y-3">
              {drift.rationale.map((reason: string, i: number) => (
                <div key={i} className="flex items-start gap-3 text-sm text-gray-300">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent mt-1.5 flex-none" />
                  {reason}
                </div>
              ))}
              {drift.rationale.length === 0 && (
                <p className="text-sm text-gray-500 italic">No significant shifts identified.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const MetricCard = ({ label, value, icon, color }: { label: string, value: number, icon: React.ReactNode, color: string }) => {
  const percent = (value * 100).toFixed(1);
  return (
    <div className="bg-surface border border-gray-800 rounded-xl p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{label}</span>
        <div className={`text-${color}-400`}>{icon}</div>
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold text-white">{percent}%</span>
      </div>
      <div className="mt-3 w-full bg-gray-800 h-1 rounded-full overflow-hidden">
        <div 
          className={`h-full bg-${color}-500 transition-all duration-1000`} 
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
};

export default DriftExplorer;
