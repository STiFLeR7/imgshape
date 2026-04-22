import React, { useState } from 'react';
import { Copy } from 'lucide-react';

const InstallationMatrix: React.FC = () => {
  const [config, setConfig] = useState({
    build: 'Stable (v4.2.0)',
    os: 'Linux',
    package: 'Pip',
    compute: 'CUDA 11.8'
  });

  const getCommand = () => {
    let cmd = 'pip install imgshape';
    if (config.compute.includes('CUDA')) {
        cmd = `pip install "imgshape[full]" --index-url https://download.vision.ai/whl/${config.compute.toLowerCase().replace(' ', '')}`;
    }
    return cmd;
  };

  const builds = ['Stable (v4.2.0)', 'Nightly', 'LTS', 'Preview'];
  const oss = ['Linux', 'macOS', 'Windows'];
  const packages = ['Pip', 'Conda', 'Docker'];
  const computes = ['CPU', 'CUDA 11.8', 'CUDA 12.1', 'ROCm 5.6'];

  const Selector = ({ label, options, field }: { label: string, options: string[], field: keyof typeof config }) => (
    <div className="space-y-4">
      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{label}</label>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {options.map(opt => (
          <button 
            key={opt}
            onClick={() => setConfig({...config, [field]: opt})}
            className={`px-4 py-3 rounded-lg border text-xs font-bold transition-all ${config[field] === opt ? 'border-blue-500 bg-blue-500/10 text-blue-400' : 'border-slate-800 text-slate-500 hover:border-slate-700'}`}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="grid grid-cols-12 gap-6 mb-32">
      <div className="col-span-12 lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl p-8 cyber-glow relative overflow-hidden">
        <div className="space-y-8 relative z-10">
          <Selector label="01. SELECT BUILD" options={builds} field="build" />
          <Selector label="02. OPERATING SYSTEM" options={oss} field="os" />
          <Selector label="03. PACKAGE MANAGER" options={packages} field="package" />
          <Selector label="04. COMPUTE PLATFORM" options={computes} field="compute" />
        </div>
      </div>
      <div className="col-span-12 lg:col-span-4 bg-slate-950 border border-slate-800 rounded-2xl p-6 flex flex-col">
        <div className="flex items-center justify-between mb-6">
           <span className="text-[10px] font-bold text-slate-500 uppercase">Terminal Output</span>
        </div>
        <div className="flex-1 font-mono text-xs text-blue-400 leading-relaxed bg-black/40 p-4 rounded-lg border border-blue-500/20">
          <span className="text-emerald-500"># Run this command in your terminal</span><br/>
          <div className="mt-2">{getCommand()}</div>
        </div>
        <button 
          onClick={() => navigator.clipboard.writeText(getCommand())}
          className="mt-6 w-full flex items-center justify-center gap-3 bg-slate-900 border border-slate-800 py-4 rounded-xl hover:bg-slate-800 transition-all text-sm font-bold"
        >
          <Copy className="w-4 h-4 text-blue-400" />
          Copy to Clipboard
        </button>
      </div>
    </div>
  );
};

export default InstallationMatrix;
