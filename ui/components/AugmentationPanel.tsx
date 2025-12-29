import React from 'react';
import { Sliders, RotateCw, Contrast, Sun, Droplet, LayoutGrid } from 'lucide-react';
import { AugmentationConfig } from '../types';

interface AugmentationPanelProps {
  config: AugmentationConfig;
  onChange: (updates: Partial<AugmentationConfig>) => void;
  onExecute: () => void;
  isLoading: boolean;
  disabled: boolean;
}

const AugmentationPanel: React.FC<AugmentationPanelProps> = ({ 
  config, 
  onChange, 
  onExecute, 
  isLoading,
  disabled 
}) => {
  return (
    <div className="bg-surface border border-gray-800 rounded-xl p-6 flex flex-col h-full shadow-lg">
      <div className="flex items-center mb-6 pb-4 border-b border-gray-800">
        <div className="p-2 bg-purple-500/20 rounded-lg mr-3">
          <LayoutGrid className="w-5 h-5 text-purple-400" />
        </div>
        <h3 className="text-lg font-semibold text-gray-200">Augmentation Config</h3>
      </div>

      <div className="space-y-6 overflow-y-auto custom-scrollbar flex-1 pr-2">
        {/* Quantity */}
        <div className="bg-surfaceHighlight/30 p-4 rounded-lg">
          <label className="flex justify-between text-sm font-medium text-gray-300 mb-2">
            <span>Images to Generate</span>
            <span className="text-accent">{config.num_to_generate}</span>
          </label>
          <input
            type="range"
            min="1"
            max="20"
            value={config.num_to_generate}
            onChange={(e) => onChange({ num_to_generate: parseInt(e.target.value) })}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-accent"
          />
        </div>

        {/* Color Adjustments */}
        <div className="space-y-4">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-2">
            <Sliders className="w-3 h-3" /> Color & Signal
          </h4>
          
          <div>
            <label className="flex items-center justify-between text-xs text-gray-400 mb-1">
              <span className="flex items-center gap-1"><Sun className="w-3 h-3" /> Brightness</span>
              <span>{config.brightness.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.brightness}
              onChange={(e) => onChange({ brightness: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
          </div>

          <div>
            <label className="flex items-center justify-between text-xs text-gray-400 mb-1">
              <span className="flex items-center gap-1"><Contrast className="w-3 h-3" /> Contrast</span>
              <span>{config.contrast.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.contrast}
              onChange={(e) => onChange({ contrast: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
          </div>

          <div>
            <label className="flex items-center justify-between text-xs text-gray-400 mb-1">
              <span className="flex items-center gap-1"><Droplet className="w-3 h-3" /> Saturation</span>
              <span>{config.saturation.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.saturation}
              onChange={(e) => onChange({ saturation: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
          </div>
        </div>

        {/* Geometry */}
        <div className="space-y-4 pt-2 border-t border-gray-800">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-2">
            <RotateCw className="w-3 h-3" /> Geometry
          </h4>
          
          <div>
            <label className="flex items-center justify-between text-xs text-gray-400 mb-1">
              <span>Max Rotation</span>
              <span>{config.rotation}°</span>
            </label>
            <input
              type="range"
              min="0"
              max="45"
              step="5"
              value={config.rotation}
              onChange={(e) => onChange({ rotation: parseInt(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
            />
          </div>
        </div>

        {/* Toggles */}
        <div className="grid grid-cols-2 gap-3 pt-2">
          {[
            { key: 'color_jitter', label: 'Color Jitter' },
            { key: 'rotate', label: 'Rotate' },
            { key: 'blur', label: 'Blur' },
            { key: 'crop', label: 'Crop' },
          ].map((toggle) => (
            <button
              key={toggle.key}
              onClick={() => onChange({ [toggle.key]: !config[toggle.key as keyof AugmentationConfig] })}
              className={`p-2 rounded text-xs font-medium border transition-all ${
                config[toggle.key as keyof AugmentationConfig]
                  ? 'bg-accent/20 border-accent text-accent'
                  : 'bg-surfaceHighlight border-gray-700 text-gray-400 hover:bg-gray-800'
              }`}
            >
              {toggle.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-gray-800">
        <button
          onClick={onExecute}
          disabled={disabled || isLoading}
          className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-bold py-3 rounded-lg shadow-lg shadow-purple-900/20 transition-all flex items-center justify-center space-x-2"
        >
          {isLoading ? (
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <LayoutGrid className="w-5 h-5" />
          )}
          <span>Generate Batch</span>
        </button>
        {disabled && <p className="text-center text-xs text-gray-500 mt-2">Upload image or set path first</p>}
      </div>
    </div>
  );
};

export default AugmentationPanel;