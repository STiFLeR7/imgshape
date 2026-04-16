# Atlas Bento Dashboard Phase 3: Control Drawer & App Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the slide-out Control Drawer and integrate all bento components into a unified, high-performance App structure.

**Architecture:** Create `ControlDrawer` component. Refactor `Sidebar` into an icon-only navigation and a slide-out settings panel.

**Tech Stack:** React 19, Tailwind CSS, Framer Motion (or simple CSS transitions)

---

### Task 1: Create ControlDrawer Component

**Files:**
- Create: `ui/components/ControlDrawer.tsx`

- [ ] **Step 1: Implement ControlDrawer with dataset selection and v4 config**

```tsx
import React from 'react';
import { X, Upload, Settings } from 'lucide-react';
// Import types...

interface ControlDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  // ... state and handlers from App.tsx ...
}

const ControlDrawer: React.FC<ControlDrawerProps> = ({ isOpen, onClose, ...props }) => {
  return (
    <div className={`fixed inset-y-0 left-0 w-80 bg-slate-900 border-r border-slate-800 shadow-2xl z-[100] transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="h-full flex flex-col p-6 overflow-y-auto custom-scrollbar">
            <div className="flex justify-between items-center mb-8">
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Settings className="w-5 h-5 text-slate-400" />
                    Configuration
                </h2>
                <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-lg text-slate-400"><X className="w-5 h-5" /></button>
            </div>
            
            {/* Move Dataset Selection, Task Type, Deployment, Priority from Sidebar here */}
            {/* ... */}
        </div>
    </div>
  );
};

export default ControlDrawer;
```

---

### Task 2: Refactor App.tsx and Sidebar.tsx

**Files:**
- Modify: `ui/App.tsx`
- Modify: `ui/components/Sidebar.tsx`

- [ ] **Step 1: Simplify Sidebar to icon-only navigation**
- [ ] **Step 2: Add "Analyze" floating action button or primary header button**
- [ ] **Step 3: Integrate ControlDrawer into App.tsx**
- [ ] **Step 4: Verify full bento layout with all components passing data correctly**
- [ ] **Step 5: Final CSS polish (Slate/Emerald theme enforcement)**

```bash
git add ui/App.tsx ui/components/Sidebar.tsx ui/components/ControlDrawer.tsx
git commit -m "feat: complete dashboard refactor with ControlDrawer and bento integration"
```
