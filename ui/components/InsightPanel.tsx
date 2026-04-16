import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';

interface InsightPanelProps {
  profiles?: any;
  loading?: boolean;
}

const InsightPanel: React.FC<InsightPanelProps> = ({ profiles, loading }) => {
  const medical = profiles?.medical;
  const satellite = profiles?.satellite;
  const ocr = profiles?.ocr;

  const renderProgress = (label: string, value: number, bgClass: string, unit: string = '%') => (
    <div className="space-y-1.5" key={label}>
        <div className="flex justify-between text-[10px] font-bold">
            <span className="text-slate-400 uppercase tracking-tighter">{label}</span>
            <span className="text-white font-mono">{value.toFixed(1)}{unit}</span>
        </div>
        <div className="w-full bg-slate-900 h-1 rounded-full overflow-hidden border border-slate-800">
            <div 
                className={`h-full ${bgClass} transition-all duration-1000`} 
                style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
            />
        </div>
    </div>
  );

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 h-full flex flex-col">
       <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-6">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Domain Intelligence
        </h3>

        <div className="flex-1 space-y-6 overflow-y-auto custom-scrollbar">
            {loading ? (
                <div className="space-y-4">
                    {[1, 2, 3].map(i => (
                        <div key={i} className="h-10 bg-slate-900/50 animate-pulse rounded-lg" />
                    ))}
                </div>
            ) : medical ? (
                <div className="space-y-4">
                    <div className="text-[10px] font-bold text-blue-400 px-2 py-1 bg-blue-500/10 rounded border border-blue-500/20 inline-block uppercase">MEDICAL CONTEXT</div>
                    {renderProgress("Slice Consistency", (medical.slice_consistency || 0) * 100, "bg-blue-500")}
                    {renderProgress("Contrast Range", ((medical.hu_range?.[1] || 0) - (medical.hu_range?.[0] || 0)) / 20, "bg-blue-500", " HU")}
                </div>
            ) : satellite ? (
                <div className="space-y-4">
                    <div className="text-[10px] font-bold text-emerald-400 px-2 py-1 bg-emerald-500/10 rounded border border-emerald-500/20 inline-block uppercase">SATELLITE CONTEXT</div>
                    {renderProgress("Cloud Cover", (satellite.cloud_cover_estimate || 0) * 100, "bg-amber-500")}
                    {renderProgress("Ground Resolution", (1.0 - (satellite.gsd_estimate || 0)) * 100, "bg-emerald-500", "m")}
                </div>
            ) : ocr ? (
                <div className="space-y-4">
                    <div className="text-[10px] font-bold text-orange-400 px-2 py-1 bg-orange-500/10 rounded border border-orange-500/20 inline-block uppercase">OCR CONTEXT</div>
                    {renderProgress("Text Density", (ocr.text_density || 0) * 100, "bg-orange-500")}
                    {renderProgress("Alignment Score", (1.0 - (ocr.orientation_variance || 0)) * 100, "bg-orange-500")}
                </div>
            ) : (
                <div className="h-full flex flex-col items-center justify-center opacity-30 text-slate-500 gap-2">
                    <Info className="w-8 h-8" />
                    <span className="text-[10px] font-bold uppercase tracking-widest text-center">General Purpose Profile<br/>No Specialized Insights</span>
                </div>
            )}
        </div>
    </div>
  );
};

export default InsightPanel;
