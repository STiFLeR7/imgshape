import React, { useRef } from 'react';
import { 
  X, 
  Upload, 
  Settings, 
  Zap, 
  Target, 
  Cpu, 
  Gauge 
} from 'lucide-react';
import { 
  AppState, 
  V4Config, 
  V3Config, 
  TaskType, 
  DeploymentTarget, 
  Priority 
} from '../types';
import { Tooltip } from './Tooltip';

interface ControlDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  state: AppState;
  onStateChange: (updates: Partial<AppState>) => void;
  onConfigV4Change: (updates: Partial<V4Config>) => void;
  onConfigV3Change: (updates: Partial<V3Config>) => void;
}

const ControlDrawer: React.FC<ControlDrawerProps> = ({
  isOpen,
  onClose,
  state,
  onStateChange,
  onConfigV4Change,
  onConfigV3Change
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
    <>
      {/* Backdrop */}
      <div 
        className={`fixed inset-0 bg-black/50 backdrop-blur-sm z-[90] transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        onClick={onClose}
      />

      {/* Drawer */}
      <div 
        className={`fixed inset-y-0 left-0 w-80 bg-slate-900 border-r border-slate-800 shadow-2xl z-[100] transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="h-full flex flex-col p-6 overflow-y-auto custom-scrollbar">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Settings className="w-5 h-5 text-slate-400" />
              Configuration
            </h2>
            <button 
              onClick={onClose} 
              className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 transition-colors"
              aria-label="Close settings"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="space-y-6">
            {/* API Version Selector */}
            <div>
              <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                API Version
              </label>
              <Tooltip text="Choose between v3 (Legacy) and v4 (Atlas) engines" position="right">
                <div className="relative w-full">
                  <select
                    value={state.version}
                    onChange={(e) => onStateChange({ version: e.target.value as 'v3' | 'v4' })}
                    className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-4 py-2.5 appearance-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none transition-all cursor-pointer"
                  >
                    <option value="v4">v4.0.0 (Atlas)</option>
                    <option value="v3">v3.x (Legacy)</option>
                  </select>
                  <div className="absolute right-3 top-3 pointer-events-none text-slate-400">
                    <Zap className="w-4 h-4" />
                  </div>
                </div>
              </Tooltip>
            </div>

            {/* Upload Zone */}
            <Tooltip text="Upload an image to analyze its dataset properties" position="right">
              <div 
                className="w-full relative border-2 border-dashed border-slate-700 rounded-xl p-6 text-center hover:border-emerald-500 hover:bg-emerald-500/5 transition-all cursor-pointer group"
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
                  <div className="p-3 bg-slate-800 rounded-full group-hover:scale-110 transition-transform">
                    <Upload className="w-6 h-6 text-emerald-500" />
                  </div>
                  <p className="text-sm font-medium text-gray-300">
                    {state.file ? state.file.name : "Drop image or click to browse"}
                  </p>
                  {!state.file && (
                    <p className="text-xs text-gray-500">Supports JPG, PNG (Max 10MB)</p>
                  )}
                </div>
              </div>
            </Tooltip>

            {/* Dataset Path */}
            <div>
              <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Or Server Dataset Path
              </label>
              <Tooltip text="Absolute path to the dataset on the server" position="top">
                <input
                  type="text"
                  value={state.datasetPath}
                  onChange={(e) => onStateChange({ datasetPath: e.target.value, file: null, filePreviewUrl: null })}
                  placeholder="/mnt/data/dataset_v1"
                  className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-emerald-500 outline-none text-sm placeholder-slate-600"
                />
              </Tooltip>
            </div>

            <div className="h-px bg-slate-800" />

            {/* v4 Configuration */}
            {state.version === 'v4' ? (
              <div className="space-y-5 animate-in fade-in slide-in-from-left-4 duration-300">
                <div>
                  <label className="flex items-center text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 gap-2">
                    <Target className="w-3 h-3" /> Task Type
                  </label>
                  <Tooltip text="Select the primary AI task for the dataset" position="right">
                    <select
                      value={state.v4Config.task}
                      onChange={(e) => onConfigV4Change({ task: e.target.value as TaskType })}
                      className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-emerald-500 outline-none cursor-pointer"
                    >
                      <option value="classification">Classification</option>
                      <option value="detection">Detection</option>
                      <option value="segmentation">Segmentation</option>
                      <option value="generation">Generation</option>
                      <option value="other">Other</option>
                    </select>
                  </Tooltip>
                </div>

                <div>
                  <label className="flex items-center text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 gap-2">
                    <Cpu className="w-3 h-3" /> Deployment
                  </label>
                  <Tooltip text="Target hardware environment for the model" position="right">
                    <select
                      value={state.v4Config.deployment}
                      onChange={(e) => onConfigV4Change({ deployment: e.target.value as DeploymentTarget })}
                      className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-emerald-500 outline-none cursor-pointer"
                    >
                      <option value="cloud">Cloud (AWS/GCP)</option>
                      <option value="edge">Edge (Jetson/RPI)</option>
                      <option value="mobile">Mobile (iOS/Android)</option>
                      <option value="embedded">Embedded (MCU)</option>
                    </select>
                  </Tooltip>
                </div>

                <div>
                  <label className="flex items-center text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 gap-2">
                    <Gauge className="w-3 h-3" /> Priority
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    {(['accuracy', 'speed', 'size', 'balanced'] as Priority[]).map((p) => (
                      <Tooltip key={p} text={`Optimize for ${p}`} position="top">
                        <button
                          onClick={() => onConfigV4Change({ priority: p })}
                          className={`w-full px-3 py-2 rounded-lg text-xs font-medium border transition-all ${
                            state.v4Config.priority === p
                              ? 'bg-emerald-500/20 border-emerald-500 text-emerald-500'
                              : 'bg-slate-800 border-transparent text-gray-400 hover:text-white'
                          }`}
                        >
                          {p.charAt(0).toUpperCase() + p.slice(1)}
                        </button>
                      </Tooltip>
                    ))}
                  </div>
                </div>
                
                <div>
                  <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 block">
                    Max Model Size (MB)
                  </label>
                  <Tooltip text="Optional limit for the generated model size" position="top">
                    <input 
                      type="number"
                      min="1"
                      placeholder="Optional (e.g. 50)"
                      value={state.v4Config.maxModelSize || ''}
                      onChange={(e) => onConfigV4Change({ maxModelSize: e.target.value ? parseInt(e.target.value) : undefined })}
                      className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-emerald-500 outline-none"
                    />
                  </Tooltip>
                </div>
              </div>
            ) : (
              /* v3 Configuration */
              <div className="space-y-4 animate-in fade-in slide-in-from-left-4 duration-300">
                <div>
                  <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    Preferences
                  </label>
                  <Tooltip text="Comma-separated preferences (e.g. fast, accurate)" position="top">
                    <input
                      type="text"
                      value={state.v3Config.prefs}
                      onChange={(e) => onConfigV3Change({ prefs: e.target.value })}
                      placeholder="fast, accurate"
                      className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-emerald-500 outline-none"
                    />
                  </Tooltip>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    Target Model
                  </label>
                  <Tooltip text="Specific model architecture to target (e.g. resnet50)" position="top">
                    <input
                      type="text"
                      value={state.v3Config.model}
                      onChange={(e) => onConfigV3Change({ model: e.target.value })}
                      placeholder="resnet50"
                      className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-emerald-500 outline-none"
                    />
                  </Tooltip>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default ControlDrawer;