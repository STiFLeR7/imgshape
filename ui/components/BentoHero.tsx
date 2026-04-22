import React from 'react';
import { Copy, Rocket } from 'lucide-react';

const BentoHero: React.FC = () => {
  const copyToClipboard = () => {
    navigator.clipboard.writeText('pip install imgshape');
  };

  return (
    <section className="max-w-7xl mx-auto px-6 mb-24 text-center pt-12">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-slate-800 bg-slate-900/50 mb-8">
        <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">v4.2.0 NOW STABLE</span>
      </div>
      <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 tracking-tight">
        Build Better Vision Models with <span className="text-blue-400">Clean Data</span>
      </h1>
      <p className="text-slate-400 text-xl max-w-2xl mx-auto mb-10 leading-relaxed">
        Deterministic fingerprinting, DINOv2-powered drift detection, and automated preprocessing—from local CLI to Cloud PaaS.
      </p>
      <div className="flex flex-col md:flex-row items-center justify-center gap-4">
        <div 
          onClick={copyToClipboard}
          className="flex items-center gap-3 bg-slate-900 border border-blue-500/30 px-6 py-4 rounded-xl font-mono text-sm text-blue-400 cursor-pointer hover:border-blue-500 transition-all shadow-[0_0_20px_-5px_rgba(59,130,246,0.2)]"
        >
          <span className="text-slate-600">$</span>
          <span>pip install imgshape</span>
          <Copy className="w-4 h-4 ml-2" />
        </div>
        <button className="px-8 py-4 bg-transparent border-2 border-purple-500/50 text-purple-400 rounded-xl font-bold hover:bg-purple-500/10 transition-all flex items-center gap-2">
          Launch Cloud Dashboard
          <Rocket className="w-5 h-5" />
        </button>
      </div>
    </section>
  );
};

export default BentoHero;
