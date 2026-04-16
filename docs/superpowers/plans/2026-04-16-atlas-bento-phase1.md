# Atlas Bento Dashboard Phase 1: Grid Foundation & KPI Cards

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 12-column bento grid layout and the 4 primary KPI cards for the main dashboard.

**Architecture:** Create a `DashboardGrid` wrapper and `KPICard` component. Update `App.tsx` to use the new grid layout.

**Tech Stack:** React 19, Tailwind CSS, Lucide-React

---

### Task 1: Create KPICard Component

**Files:**
- Create: `ui/components/KPICard.tsx`
- Modify: `ui/index.html` (Ensure Inter font)

- [ ] **Step 1: Create KPICard with metric styling**

```tsx
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
  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5 hover:border-slate-600 transition-all group">
      <div className="flex justify-between items-start mb-2">
        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{label}</span>
        <Icon className={`w-4 h-4 text-slate-500 group-hover:text-${color}-400 transition-colors`} />
      </div>
      {loading ? (
        <div className="h-8 w-3/4 bg-slate-700 animate-pulse rounded mt-1"></div>
      ) : (
        <div className="text-2xl font-bold font-mono text-white">{value}</div>
      )}
      {subtext && <div className={`text-[10px] text-${color}-500/80 mt-1`}>{subtext}</div>}
    </div>
  );
};

export default KPICard;
```

- [ ] **Step 2: Commit**

```bash
git add ui/components/KPICard.tsx
git commit -m "feat: add KPICard component for dashboard metrics"
```

---

### Task 2: Implement DashboardGrid in App.tsx

**Files:**
- Modify: `ui/App.tsx`
- Create: `ui/components/DashboardGrid.tsx`

- [ ] **Step 1: Create DashboardGrid wrapper**

```tsx
import React from 'react';

const DashboardGrid: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-4 animate-in fade-in duration-700">
      {children}
    </div>
  );
};

export default DashboardGrid;
```

- [ ] **Step 2: Update App.tsx renderContent for 'dashboard'**

```tsx
// Inside renderContent() switch 'dashboard' case:
return (
  <DashboardGrid>
    {/* KPI Row (each 3 cols) */}
    <div className="lg:col-span-3">
      <KPICard 
        label="Total Samples" 
        value={state.results?.fingerprint?.sample_count || '0'} 
        icon={Layers} 
        loading={state.status === 'loading'}
      />
    </div>
    <div className="lg:col-span-3">
      <KPICard 
        label="Drift Confidence" 
        value={state.results?.drift?.overall_drift ? `${(100 - state.results.drift.overall_drift * 100).toFixed(1)}%` : '100%'} 
        icon={Zap} 
        color="amber"
        loading={state.status === 'loading'}
      />
    </div>
    <div className="lg:col-span-3">
       <KPICard 
        label="Detected Domain" 
        value={state.results?.fingerprint?.profiles?.semantic?.inferred_type || 'Unknown'} 
        icon={Target} 
        color="indigo"
        loading={state.status === 'loading'}
      />
    </div>
    <div className="lg:col-span-3">
       <KPICard 
        label="Processing" 
        value={state.gpuActive ? 'GPU' : 'CPU'} 
        subtext={state.gpuActive ? 'PyTorch/CUDA' : 'Standard'}
        icon={Cpu} 
        color="purple"
        loading={state.status === 'loading'}
      />
    </div>

    {/* Placeholder for Task 2 content */}
    <div className="lg:col-span-8 bg-slate-900/50 border border-slate-800 rounded-xl h-96 flex items-center justify-center">
        <span className="text-slate-600 italic">Semantic Map Loading...</span>
    </div>
    <div className="lg:col-span-4 bg-slate-900/50 border border-slate-800 rounded-xl h-96 flex items-center justify-center">
        <span className="text-slate-600 italic">Insights Loading...</span>
    </div>
  </DashboardGrid>
);
```

- [ ] **Step 3: Commit**

```bash
git add ui/components/DashboardGrid.tsx ui/App.tsx
git commit -m "feat: implement 12-column bento grid and KPI cards in main dashboard"
```
