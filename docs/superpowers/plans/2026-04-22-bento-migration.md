# imgshape Bento Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the current UI into the "Atlas Bento" v4.2.0 layout with real project data and local-first resilience.

**Architecture:** Component-based migration using Tailwind CSS for high-fidelity styling. Logic is separated into functional components with a central orchestrator in `App.tsx`.

**Tech Stack:** React, TypeScript, Tailwind CSS, Lucide React (icons), Vitest (testing).

---

### Task 1: Styling Utilities & Foundation

**Files:**
- Modify: `ui/index.html` (Inject global styles for now, or use a CSS file if available)
- Create: `ui/components/Styles.ts` (Shared Tailwind class constants)

- [ ] **Step 1: Add STITCH global styles**
Add the following to a `<style>` block in `ui/index.html` or a new CSS file:
```css
.cyber-glow {
  background: linear-gradient(135deg, rgba(173, 198, 255, 0.1) 0%, rgba(173, 198, 255, 0) 50%);
}
.accent-border {
  position: relative;
}
.accent-border::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 1px;
  height: 20px;
  background: #adc6ff;
}
```

- [ ] **Step 2: Commit**
```bash
git add ui/index.html
git commit -m "style: add STITCH cyber-glow and accent-border utilities"
```

---

### Task 2: Implement BentoHero Component

**Files:**
- Create: `ui/components/BentoHero.tsx`
- Test: `ui/components/BentoHero.test.tsx`

- [ ] **Step 1: Write failing test for BentoHero**
```tsx
import { render, screen } from '@testing-library/react';
import BentoHero from './BentoHero';

test('renders real branding and version', () => {
  render(<BentoHero />);
  expect(screen.getByText(/imgshape/i)).toBeDefined();
  expect(screen.getByText(/v4.2.0 NOW STABLE/i)).toBeDefined();
});
```

- [ ] **Step 2: Implement BentoHero**
```tsx
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
```

- [ ] **Step 3: Run tests and commit**
```bash
npm test ui/components/BentoHero.test.tsx
git add ui/components/BentoHero.tsx
git commit -m "feat: add BentoHero component with real v4.2.0 branding"
```

---

### Task 3: Implement InstallationMatrix Component

**Files:**
- Create: `ui/components/InstallationMatrix.tsx`

- [ ] **Step 1: Implement InstallationMatrix**
```tsx
import React, { useState } from 'react';
import { Terminal, Laptop, Window, Copy } from 'lucide-react';

const InstallationMatrix: React.FC = () => {
  const [config, setConfig] = useState({
    build: 'Stable (v4.2.0)',
    os: 'Linux',
    package: 'Pip',
    compute: 'CUDA 11.8'
  });

  const getCommand = () => {
    let cmd = 'pip install imgshape';
    if (config.compute.includes('CUDA')) cmd = `pip install "imgshape[full]" --index-url https://download.vision.ai/whl/${config.compute.toLowerCase().replace(' ', '')}`;
    return cmd;
  };

  return (
    <div className="grid grid-cols-12 gap-6 mb-32">
      <div className="col-span-12 lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl p-8 cyber-glow relative overflow-hidden">
        <div className="space-y-8 relative z-10">
          <div>
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-4">01. SELECT BUILD</label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {['Stable (v4.2.0)', 'Nightly', 'LTS', 'Preview'].map(b => (
                <button 
                  key={b}
                  onClick={() => setConfig({...config, build: b})}
                  className={`px-4 py-3 rounded-lg border text-sm font-bold transition-all ${config.build === b ? 'border-blue-500 bg-blue-500/10 text-blue-400' : 'border-slate-800 text-slate-500 hover:border-slate-700'}`}
                >
                  {b}
                </button>
              ))}
            </div>
          </div>
          {/* Repeat for OS, Package, Compute similar to above using STITCH selectors */}
        </div>
      </div>
      <div className="col-span-12 lg:col-span-4 bg-slate-950 border border-slate-800 rounded-2xl p-6 flex flex-col">
        <div className="flex items-center justify-between mb-6">
           <span className="text-[10px] font-bold text-slate-500 uppercase">Terminal Output</span>
        </div>
        <div className="flex-1 font-mono text-sm text-blue-400 leading-relaxed bg-black/40 p-4 rounded-lg border border-blue-500/20">
          <span className="text-emerald-500"># Run this command</span><br/>
          {getCommand()}
        </div>
        <button className="mt-6 w-full flex items-center justify-center gap-3 bg-slate-900 border border-slate-800 py-4 rounded-xl hover:bg-slate-800 transition-all text-sm font-bold">
          <Copy className="w-4 h-4 text-blue-400" />
          Copy to Clipboard
        </button>
      </div>
    </div>
  );
};

export default InstallationMatrix;
```

- [ ] **Step 2: Commit**
```bash
git add ui/components/InstallationMatrix.tsx
git commit -m "feat: add interactive InstallationMatrix component"
```

---

### Task 4: Integration in App.tsx

**Files:**
- Modify: `ui/App.tsx`
- Create: `ui/components/CloudStatusIndicator.tsx`

- [ ] **Step 1: Create CloudStatusIndicator**
```tsx
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
```

- [ ] **Step 2: Update App.tsx to include Hero and Matrix**
Modify `ui/App.tsx` to include `<BentoHero />` and `<InstallationMatrix />` in the `dashboard` view.

- [ ] **Step 3: Commit**
```bash
git add ui/App.tsx ui/components/CloudStatusIndicator.tsx
git commit -m "feat: integrate Bento components and Local Mode indicator into App"
```
