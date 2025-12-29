import React from 'react';
import { FileText, PieChart, Database, FileOutput } from 'lucide-react';
import { ReportConfig } from '../types';

interface ReportGeneratorProps {
  config: ReportConfig;
  onChange: (updates: Partial<ReportConfig>) => void;
  onGenerate: () => void;
  isLoading: boolean;
  hasResults: boolean;
}

const ReportGenerator: React.FC<ReportGeneratorProps> = ({
  config,
  onChange,
  onGenerate,
  isLoading,
  hasResults
}) => {
  return (
    <div className="bg-surface border border-gray-800 rounded-xl p-6 flex flex-col h-full shadow-lg">
      <div className="flex items-center mb-6 pb-4 border-b border-gray-800">
        <div className="p-2 bg-pink-500/20 rounded-lg mr-3">
          <FileText className="w-5 h-5 text-pink-400" />
        </div>
        <h3 className="text-lg font-semibold text-gray-200">Report Settings</h3>
      </div>

      <div className="space-y-6 flex-1">
        <div>
          <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Output Format
          </label>
          <div className="grid grid-cols-3 gap-3">
            {(['markdown', 'html', 'pdf'] as const).map((fmt) => (
              <button
                key={fmt}
                onClick={() => onChange({ format: fmt })}
                className={`py-3 px-2 rounded-lg text-sm font-medium border transition-all capitalize ${
                  config.format === fmt
                    ? 'bg-pink-500/20 border-pink-500 text-pink-400'
                    : 'bg-surfaceHighlight border-gray-700 text-gray-400 hover:text-gray-200'
                }`}
              >
                {fmt}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Content
          </label>
          
          <div 
            className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all ${
              config.include_metadata 
                ? 'bg-surfaceHighlight border-pink-500/50' 
                : 'bg-transparent border-gray-800'
            }`}
            onClick={() => onChange({ include_metadata: !config.include_metadata })}
          >
            <div className="flex items-center gap-3">
              <Database className={`w-4 h-4 ${config.include_metadata ? 'text-pink-400' : 'text-gray-500'}`} />
              <span className="text-sm text-gray-300">Include Metadata</span>
            </div>
            <div className={`w-4 h-4 rounded border flex items-center justify-center ${
              config.include_metadata ? 'bg-pink-500 border-pink-500' : 'border-gray-600'
            }`}>
              {config.include_metadata && <div className="w-2 h-2 bg-white rounded-sm" />}
            </div>
          </div>

          <div 
            className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all ${
              config.include_charts 
                ? 'bg-surfaceHighlight border-pink-500/50' 
                : 'bg-transparent border-gray-800'
            }`}
            onClick={() => onChange({ include_charts: !config.include_charts })}
          >
            <div className="flex items-center gap-3">
              <PieChart className={`w-4 h-4 ${config.include_charts ? 'text-pink-400' : 'text-gray-500'}`} />
              <span className="text-sm text-gray-300">Include Visualization</span>
            </div>
            <div className={`w-4 h-4 rounded border flex items-center justify-center ${
              config.include_charts ? 'bg-pink-500 border-pink-500' : 'border-gray-600'
            }`}>
              {config.include_charts && <div className="w-2 h-2 bg-white rounded-sm" />}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-auto pt-4 border-t border-gray-800">
        <button
          onClick={onGenerate}
          disabled={!hasResults || isLoading}
          className="w-full bg-pink-600 hover:bg-pink-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-bold py-3 rounded-lg shadow-lg shadow-pink-900/20 transition-all flex items-center justify-center space-x-2"
        >
          {isLoading ? (
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <FileOutput className="w-5 h-5" />
          )}
          <span>Generate Report</span>
        </button>
        {!hasResults && (
          <p className="text-center text-xs text-red-400 mt-2">
            Run analysis first to generate report
          </p>
        )}
      </div>
    </div>
  );
};

export default ReportGenerator;