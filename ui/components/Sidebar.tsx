import React, { useRef } from 'react';
import { 
  Upload, 
  Search, 
  Fingerprint, 
  Zap, 
  CheckCircle2,
  Cpu,
  Target,
  Gauge,
  Download,
  Copy
} from 'lucide-react';
import { AppState, V4Config, V3Config, TaskType, DeploymentTarget, Priority } from '../types';

interface SidebarProps {
  state: AppState;
  onStateChange: (updates: Partial<AppState>) => void;
  onConfigV4Change: (updates: Partial<V4Config>) => void;
  onConfigV3Change: (updates: Partial<V3Config>) => void;
  onAnalyze: () => void;
  onFingerprint: () => void;
  onRecommend: () => void;
  onDownloadJson: () => void;
  onCopyJson: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  state,
  onStateChange,
  onConfigV4Change,
  onConfigV3Change,
  onAnalyze,
  onFingerprint,
  onRecommend,
  onDownloadJson,
  onCopyJson
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      onStateChange({ file, filePreviewUrl: url, datasetPath: '' });
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const url = URL.createObjectURL(file);
      onStateChange({ file, filePreviewUrl: url, datasetPath: '' });
    }
  };

  return (
    <aside className="w-80 bg-surface border-r border-gray-800 flex flex-col h-[calc(100vh-64px)] overflow-y-auto custom-scrollbar">
      <div className="p-6 space-y-6">
        
        {/* Version Selector */}
        <div>
          <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            API Version
          </label>
          <div className="relative">
            <select
              value={state.version}
              onChange={(e) => onStateChange({ version: e.target.value as 'v3' | 'v4' })}
              className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-4 py-2.5 appearance-none focus:ring-2 focus:ring-accent focus:border-transparent outline-none transition-all"
            >
              <option value="v4">v4.0.0 (Atlas)</option>
              <option value="v3">v3.x (Legacy)</option>
            </select>
            <div className="absolute right-3 top-3 pointer-events-none text-gray-400">
              <Zap className="w-4 h-4" />
            </div>
          </div>
        </div>

        {/* Upload Zone */}
        <div 
          className="relative border-2 border-dashed border-gray-700 rounded-xl p-6 text-center hover:border-accent hover:bg-accent/5 transition-all cursor-pointer group"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            accept="image/png,image/jpeg" 
            onChange={handleFileChange}
          />
          <div className="flex flex-col items-center space-y-2">
            <div className="p-3 bg-surfaceHighlight rounded-full group-hover:scale-110 transition-transform">
              <Upload className="w-6 h-6 text-accent" />
            </div>
            <p className="text-sm font-medium text-gray-300">
              {state.file ? state.file.name : "Drop image or click to browse"}
            </p>
            {!state.file && (
              <p className="text-xs text-gray-500">Supports JPG, PNG (Max 10MB)</p>
            )}
          </div>
        </div>

        {/* Dataset Path */}
        <div>
          <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Or Server Dataset Path
          </label>
          <input
            type="text"
            value={state.datasetPath}
            onChange={(e) => onStateChange({ datasetPath: e.target.value, file: null, filePreviewUrl: null })}
            placeholder="/mnt/data/dataset_v1"
            className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-accent outline-none text-sm placeholder-gray-600"
          />
        </div>

        <div className="h-px bg-gray-800" />

        {/* Configuration Forms */}
        {state.version === 'v4' ? (
          <div className="space-y-5 animate-in fade-in slide-in-from-left-4 duration-300">
            <div>
              <label className="flex items-center text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 gap-2">
                <Target className="w-3 h-3" /> Task Type
              </label>
              <select
                value={state.v4Config.task}
                onChange={(e) => onConfigV4Change({ task: e.target.value as TaskType })}
                className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-accent outline-none"
              >
                <option value="classification">Classification</option>
                <option value="detection">Detection</option>
                <option value="segmentation">Segmentation</option>
                <option value="generation">Generation</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div>
              <label className="flex items-center text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 gap-2">
                <Cpu className="w-3 h-3" /> Deployment
              </label>
              <select
                value={state.v4Config.deployment}
                onChange={(e) => onConfigV4Change({ deployment: e.target.value as DeploymentTarget })}
                className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-accent outline-none"
              >
                <option value="cloud">Cloud (AWS/GCP)</option>
                <option value="edge">Edge (Jetson/RPI)</option>
                <option value="mobile">Mobile (iOS/Android)</option>
                <option value="embedded">Embedded (MCU)</option>
              </select>
            </div>

            <div>
              <label className="flex items-center text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 gap-2">
                <Gauge className="w-3 h-3" /> Priority
              </label>
              <div className="grid grid-cols-2 gap-2">
                {(['accuracy', 'speed', 'size', 'balanced'] as Priority[]).map((p) => (
                  <button
                    key={p}
                    onClick={() => onConfigV4Change({ priority: p })}
                    className={`px-3 py-2 rounded-lg text-xs font-medium border transition-all ${
                      state.v4Config.priority === p
                        ? 'bg-accent/20 border-accent text-accent'
                        : 'bg-surfaceHighlight border-transparent text-gray-400 hover:text-white'
                    }`}
                  >
                    {p.charAt(0).toUpperCase() + p.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            
            <div>
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 block">
                    Max Model Size (MB)
                </label>
                <input 
                    type="number"
                    min="1"
                    placeholder="Optional (e.g. 50)"
                    value={state.v4Config.maxModelSize || ''}
                    onChange={(e) => onConfigV4Change({ maxModelSize: e.target.value ? parseInt(e.target.value) : undefined })}
                    className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-accent outline-none"
                />
            </div>
          </div>
        ) : (
          <div className="space-y-4 animate-in fade-in slide-in-from-left-4 duration-300">
            <div>
              <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Preferences
              </label>
              <input
                type="text"
                value={state.v3Config.prefs}
                onChange={(e) => onConfigV3Change({ prefs: e.target.value })}
                placeholder="fast, accurate"
                className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-accent outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Target Model
              </label>
              <input
                type="text"
                value={state.v3Config.model}
                onChange={(e) => onConfigV3Change({ model: e.target.value })}
                placeholder="resnet50"
                className="w-full bg-surfaceHighlight border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-accent outline-none"
              />
            </div>
          </div>
        )}
      </div>

      {/* Action Footer */}
      <div className="mt-auto p-6 bg-surfaceHighlight/30 border-t border-gray-800 space-y-3">
        <button
          onClick={onAnalyze}
          disabled={state.status === 'loading'}
          className="w-full flex items-center justify-center space-x-2 bg-accent hover:bg-accentHover disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg shadow-lg shadow-accent/20 transition-all active:scale-[0.98]"
        >
          {state.status === 'loading' ? (
             <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
             <Search className="w-5 h-5" />
          )}
          <span>Start Analysis</span>
        </button>

        {state.version === 'v4' && (
          <button
            onClick={onFingerprint}
            disabled={state.status === 'loading'}
            className="w-full flex items-center justify-center space-x-2 bg-surface border border-gray-700 hover:bg-gray-800 text-gray-300 font-medium py-2.5 rounded-lg transition-all"
          >
            <Fingerprint className="w-4 h-4" />
            <span>Fingerprint Only</span>
          </button>
        )}

        {state.version === 'v3' && (
          <button
            onClick={onRecommend}
            className="w-full flex items-center justify-center space-x-2 text-gray-400 hover:text-white text-sm py-2 transition-colors"
          >
            <CheckCircle2 className="w-4 h-4" />
            <span>Get Recommendations</span>
          </button>
        )}
        
        {state.results && (
             <div className="flex space-x-2 pt-2 border-t border-gray-700/50 mt-2">
                 <button onClick={onDownloadJson} className="flex-1 flex items-center justify-center py-2 text-xs text-gray-400 hover:text-accent bg-surfaceHighlight rounded">
                     <Download className="w-3 h-3 mr-1" /> JSON
                 </button>
                 <button onClick={onCopyJson} className="flex-1 flex items-center justify-center py-2 text-xs text-gray-400 hover:text-accent bg-surfaceHighlight rounded">
                     <Copy className="w-3 h-3 mr-1" /> Copy
                 </button>
             </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;