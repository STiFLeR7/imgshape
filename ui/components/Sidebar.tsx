import React from 'react';
import { 
  Settings, 
  LayoutDashboard,
  LayoutGrid,
  FileText,
  GitCompare,
  CircleHelp
} from 'lucide-react';
import { AppState } from '../types';
import { Tooltip } from './Tooltip';

interface SidebarProps {
  state: AppState;
  onStateChange: (updates: Partial<AppState>) => void;
  onOpenSettings: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  state,
  onStateChange,
  onOpenSettings
}) => {
  return (
    <aside className="w-20 bg-slate-900 border-r border-slate-800 flex flex-col h-[calc(100vh-64px)] z-40 transition-all duration-300">
      
      {/* Navigation Icons */}
      <div className="flex flex-col flex-1 py-6 items-center space-y-4">
        
        <Tooltip text="Dashboard" position="right">
          <button
             onClick={() => onStateChange({ activeView: 'dashboard' })}
             className={`p-3 rounded-xl transition-all duration-200 ${
               state.activeView === 'dashboard' 
               ? 'bg-emerald-500/10 text-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.1)]' 
               : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800'
             }`}
          >
            <LayoutDashboard className="w-6 h-6" />
          </button>
        </Tooltip>

        <Tooltip text="Drift Analysis" position="right">
          <button
            onClick={() => onStateChange({ activeView: 'drift' })}
            className={`p-3 rounded-xl transition-all duration-200 ${
              state.activeView === 'drift' 
              ? 'bg-blue-500/10 text-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.1)]' 
              : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800'
            }`}
          >
            <GitCompare className="w-6 h-6" />
          </button>
        </Tooltip>
        
        {state.version === 'v4' && (
          <Tooltip text="Augmentation" position="right">
            <button
              onClick={() => onStateChange({ activeView: 'augmentation' })}
              className={`p-3 rounded-xl transition-all duration-200 ${
                state.activeView === 'augmentation' 
                ? 'bg-purple-500/10 text-purple-400 shadow-[0_0_15px_rgba(168,85,247,0.1)]' 
                : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800'
              }`}
            >
              <LayoutGrid className="w-6 h-6" />
            </button>
          </Tooltip>
        )}
        
        <Tooltip text="Reports" position="right">
          <button
             onClick={() => onStateChange({ activeView: 'report' })}
             className={`p-3 rounded-xl transition-all duration-200 ${
               state.activeView === 'report' 
               ? 'bg-pink-500/10 text-pink-400 shadow-[0_0_15px_rgba(236,72,153,0.1)]' 
               : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800'
             }`}
          >
            <FileText className="w-6 h-6" />
          </button>
        </Tooltip>

        <div className="flex-1" />

        <div className="h-px w-8 bg-slate-800" />

        <Tooltip text="Configuration" position="right">
          <button
             onClick={onOpenSettings}
             className="p-3 rounded-xl text-slate-500 hover:text-emerald-500 hover:bg-slate-800 transition-all duration-200"
          >
            <Settings className="w-6 h-6" />
          </button>
        </Tooltip>

        <Tooltip text="Help & Docs" position="right">
          <a
             href="https://github.com/STiFLeR7/imgshape"
             target="_blank"
             rel="noopener noreferrer"
             className="p-3 rounded-xl text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-all duration-200"
          >
            <CircleHelp className="w-6 h-6" />
          </a>
        </Tooltip>
      </div>
    </aside>
  );
};

export default Sidebar;