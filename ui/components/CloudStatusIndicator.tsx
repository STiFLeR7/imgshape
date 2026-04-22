import React from 'react';
import { CloudOff } from 'lucide-react';

const CloudStatusIndicator: React.FC = () => (
  <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 cursor-help group relative">
    <CloudOff className="w-3 h-3 text-amber-500" />
    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">LOCAL MODE</span>
    <div className="absolute top-full mt-2 right-0 w-48 p-2 bg-slate-800 border border-slate-700 text-[10px] rounded shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
      Cloud database inactive (>90 days). Data is being saved to local storage.
    </div>
  </div>
);

export default CloudStatusIndicator;
