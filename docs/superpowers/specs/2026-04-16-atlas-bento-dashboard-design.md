# Design Spec: Atlas Bento Dashboard (UX Refactor)

## 1. Overview
The Atlas Bento Dashboard is a complete UX overhaul of the imgshape frontend. It replaces the legacy "Input-Left / Output-Right" split with a modern, modular grid system (Bento) that maximizes information density while maintaining a clean, professional aesthetic.

## 2. Visual Direction
- **Theme:** High-contrast Dark Mode (Slate/Zinc).
- **Typography:** Inter/SF Pro (Sans-serif) for body, JetBrains Mono for metrics.
- **Accents:** 
  - Emerald (#10b981): Healthy data, stable drift.
  - Amber (#f59e0b): Warning, significant drift detected.
  - Indigo (#6366f1): Intelligence, embedding extraction.

## 3. Architecture & Components

### 3.1. Main Layout (`DashboardLayout`)
A 12-column responsive grid container that wraps the bento modules.

### 3.2. Bento Modules
- **KPICard:** Small modular cards (3 cols each) for high-level stats (Count, Speed, Drift %, Domain).
- **SemanticMap:** Large interactive scatter plot (8 cols) using Recharts to visualize DINOv2 embeddings.
- **InsightPanel:** Vertical panel (4 cols) with domain-specific progress bars (Medical/Satellite/OCR).
- **DecisionBoard:** Bottom-weighted card (full width or 6 cols) showing the explainable AI reasoning and the "Apply" button.

### 3.3. Interaction Flow
1. **Drawer Upload:** A slide-out drawer on the left for dataset selection/upload to keep the main canvas clean.
2. **Global Analysis:** A floating action button or primary header button triggers the `Atlas` loop.
3. **Card-Level Detail:** Hovering over charts or metrics provides contextual tooltips and raw data snippets.

## 4. Technical Stack
- **Framework:** React 19 (TypeScript).
- **Styling:** Tailwind CSS (Grid/Flexbox).
- **Charts:** Recharts.
- **Icons:** Lucide-React.

## 5. Implementation Strategy
- Refactor `App.tsx` to use the new grid layout.
- Split `ResultsPanel.tsx` into multiple specialized bento components.
- Move "Configuration" into a collapsible `ControlDrawer` component.
