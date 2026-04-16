import React from 'react';
import { LucideIcon } from 'lucide-react';

interface KPICardProps {
  label: string;
  value: string | number;
  subtext?: string;
  icon: LucideIcon;
  color?: string;
  loading?: boolean;
}

const KPICard: React.FC<KPICardProps> = ({ label, value, subtext, icon: Icon, color = 'emerald', loading }) => {
  const colorMap: Record<string, string> = {
    emerald: 'group-hover:text-emerald-400',
    amber: 'group-hover:text-amber-400',
    indigo: 'group-hover:text-indigo-400',
    purple: 'group-hover:text-purple-400',
    pink: 'group-hover:text-pink-400'
  };

  const textMap: Record<string, string> = {
    emerald: 'text-emerald-500/80',
    amber: 'text-amber-500/80',
    indigo: 'text-indigo-500/80',
    purple: 'text-purple-500/80',
    pink: 'text-pink-500/80'
  };

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5 hover:border-slate-600 transition-all group">
      <div className="flex justify-between items-start mb-2">
        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{label}</span>
        <Icon className={`w-4 h-4 text-slate-500 ${colorMap[color]} transition-colors`} />
      </div>
      {loading ? (
        <div className="h-8 w-3/4 bg-slate-700 animate-pulse rounded mt-1"></div>
      ) : (
        <div className="text-2xl font-bold font-mono text-white">{value}</div>
      )}
      {subtext && <div className={`text-[10px] ${textMap[color]} mt-1`}>{subtext}</div>}
    </div>
  );
};

export default KPICard;
