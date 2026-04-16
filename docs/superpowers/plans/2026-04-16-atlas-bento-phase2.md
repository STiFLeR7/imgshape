# Atlas Bento Dashboard Phase 2: Semantic Map & Insights

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the interactive Semantic Map and domain-specific insight panels.

**Architecture:** Create `SemanticMap` component using Recharts and `InsightPanel` for domain-specific metrics.

**Tech Stack:** React 19, Recharts, Lucide-React

---

### Task 1: Create SemanticMap Component

**Files:**
- Create: `ui/components/SemanticMap.tsx`

- [ ] **Step 1: Implement SemanticMap with Recharts scatter plot**

```tsx
import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip as ReTooltip, ResponsiveContainer, Cell } from 'recharts';
import { Brain } from 'lucide-react';

interface SemanticMapProps {
  embedding?: number[];
  loading?: boolean;
}

const SemanticMap: React.FC<SemanticMapProps> = ({ embedding, loading }) => {
  // Generate pseudo-points around the centroid for visualization if we only have centroid
  const data = embedding ? embedding.slice(0, 100).map((val, i) => ({
    x: Math.cos(i) * val + (Math.random() - 0.5),
    y: Math.sin(i) * val + (Math.random() - 0.5),
    value: val
  })) : [];

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Brain className="w-4 h-4 text-purple-400" />
            Semantic Signature Map
        </h3>
        <div className="flex gap-2 text-[9px] font-bold uppercase tracking-tighter">
            <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-500 border border-slate-800">DINOv2 L/14</span>
            <span className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">Active</span>
        </div>
      </div>
      
      <div className="flex-1 min-h-[250px] relative">
        {loading ? (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900/20 backdrop-blur-[2px] rounded-lg">
                <div className="flex flex-col items-center gap-3">
                    <div className="w-8 h-8 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
                    <span className="text-[10px] font-bold text-purple-400 uppercase tracking-widest">Projecting Embeddings...</span>
                </div>
            </div>
        ) : embedding ? (
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" dataKey="y" hide />
                    <ZAxis type="number" dataKey="value" range={[20, 100]} />
                    <ReTooltip 
                        cursor={{ strokeDasharray: '3 3' }} 
                        content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                return (
                                    <div className="bg-slate-900 border border-slate-700 p-2 rounded text-[10px] font-mono text-purple-300">
                                        Value: {payload[0].value.toFixed(4)}
                                    </div>
                                );
                            }
                            return null;
                        }}
                    />
                    <Scatter name="Embeddings" data={data}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={`rgba(168, 85, 247, ${0.3 + (entry.value / 10)})`} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        ) : (
            <div className="absolute inset-0 flex items-center justify-center border-2 border-dashed border-slate-800 rounded-lg">
                <span className="text-slate-600 text-xs font-medium italic">No semantic data available</span>
            </div>
        )}
      </div>
    </div>
  );
};

export default SemanticMap;
```

- [ ] **Step 2: Commit**

```bash
git add ui/components/SemanticMap.tsx
git commit -m "feat: implement interactive SemanticMap component with Recharts"
```

---

### Task 2: Create InsightPanel Component

**Files:**
- Create: `ui/components/InsightPanel.tsx`

- [ ] **Step 1: Implement InsightPanel with domain metrics**

```tsx
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

  const renderProgress = (label: string, value: number, color: string, unit: string = '%') => (
    <div className="space-y-1.5">
        <div className="flex justify-between text-[10px] font-bold">
            <span className="text-slate-400 uppercase tracking-tighter">{label}</span>
            <span className="text-white font-mono">{value.toFixed(1)}{unit}</span>
        </div>
        <div className="w-full bg-slate-900 h-1 rounded-full overflow-hidden border border-slate-800">
            <div 
                className={`h-full bg-${color}-500 transition-all duration-1000`} 
                style={{ width: `${Math.min(100, value)}%` }}
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

        <div className="flex-1 space-y-6">
            {loading ? (
                <div className="space-y-4">
                    {[1, 2, 3].map(i => (
                        <div key={i} className="h-10 bg-slate-900/50 animate-pulse rounded-lg" />
                    ))}
                </div>
            ) : medical ? (
                <div className="space-y-4">
                    <div className="text-[10px] font-bold text-blue-400 px-2 py-1 bg-blue-500/10 rounded border border-blue-500/20 inline-block">MEDICAL CONTEXT</div>
                    {renderProgress("Slice Consistency", medical.slice_consistency * 100, "blue")}
                    {renderProgress("Contrast Ratio", (medical.hu_range[1] - medical.hu_range[0]) / 20, "blue", "HU")}
                </div>
            ) : satellite ? (
                <div className="space-y-4">
                    <div className="text-[10px] font-bold text-emerald-400 px-2 py-1 bg-emerald-500/10 rounded border border-emerald-500/20 inline-block">SATELLITE CONTEXT</div>
                    {renderProgress("Cloud Cover", satellite.cloud_cover_estimate * 100, "amber")}
                    {renderProgress("Ground Resolution", (1.0 - satellite.gsd_estimate) * 100, "emerald", "m")}
                </div>
            ) : ocr ? (
                <div className="space-y-4">
                    <div className="text-[10px] font-bold text-orange-400 px-2 py-1 bg-orange-500/10 rounded border border-orange-500/20 inline-block">OCR CONTEXT</div>
                    {renderProgress("Text Density", ocr.text_density * 100, "orange")}
                    {renderProgress("Alignment Score", (1.0 - ocr.orientation_variance) * 100, "orange")}
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
```

- [ ] **Step 2: Commit**

```bash
git add ui/components/InsightPanel.tsx
git commit -m "feat: implement domain-specific InsightPanel with metrics visualization"
```
