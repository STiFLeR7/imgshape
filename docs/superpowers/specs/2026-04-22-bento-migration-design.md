# Design Spec: imgshape Bento Migration (Atlas Bento v4.2.0)

**Date:** 2026-04-22  
**Status:** Draft  
**Topic:** Frontend UI/UX Realization of STITCH Design

## 1. Objective
Transform the hypothetical "VisionAI" design provided by STITCH into a functional, data-driven React UI for `imgshape` v4.2.0. This involves replacing all "dummy" text with real project data and implementing a "Local-First" mode due to Supabase database inactivity.

## 2. Branding & Content Realization
| Hypothetical (STITCH) | Real (imgshape) |
| :--- | :--- |
| VisionAI | imgshape |
| v2.4.0 | v4.2.0 |
| Vision Intelligence Platform | Data-Centric AI Toolkit |
| Enterprise PaaS | Atlas Bento Engine |
| Semantic Drift Detection | DINOv2 Semantic Drift |

## 3. Component Architecture

### 3.1 `BentoHero.tsx` (New)
*   **Purpose:** The landing/hook section.
*   **Features:** 
    *   Dynamic brand logo (imgshape).
    *   `v4.2.0 NOW STABLE` status badge.
    *   Interactive `pip install imgshape` widget with copy-to-clipboard functionality.

### 3.2 `InstallationMatrix.tsx` (New)
*   **Purpose:** Interactive command generator.
*   **States:**
    *   `Build`: Stable (v4.2.0), Nightly, LTS.
    *   `OS`: Linux, macOS, Windows.
    *   `Package`: Pip, Conda, Docker.
    *   `Compute`: CPU, CUDA 11.8, CUDA 12.1, ROCm 5.6.
*   **Logic:** Updates the output string (e.g., `pip install "imgshape[full]"`).

### 3.3 `DashboardGrid.tsx` (Refactor)
*   **Layout:** 12-column Tailwind grid.
*   **Cards:**
    *   `SemanticDriftCard`: Visual scatter plot powered by `DINOv2` embeddings.
    *   `AuditResultsCard`: List of real checks (Class Balance, Metadata, Resolution).
    *   `PipelineExportCard`: Syntax-highlighted Python/YAML snippets from `AtlasPipeline`.

### 3.4 `CloudStatusIndicator.tsx` (New)
*   **Purpose:** Handle Supabase inactivity.
*   **Behavior:** 
    *   If Supabase connection fails/is inactive (90+ days), show "LOCAL MODE" in the header.
    *   Tooltip: "Cloud database inactive (>90 days). Syncing to local storage."

## 4. Technical Integration
*   **Styling:** Port `cyber-glow` (gradients) and `accent-border` (custom border-left markers) from STITCH.
*   **API:** Bind `AtlasEngine` stats (latency, throughput, uptime) to the Dashboard cards.
*   **Environment:** Use `VITE_API_BASE_URL` from `.env.local`.

## 5. Success Criteria
*   [ ] Brand "VisionAI" is completely removed.
*   [ ] Version `v4.2.0` is consistently displayed.
*   [ ] Installation command is copyable and correct.
*   [ ] UI degrades gracefully to "Local Mode" if Supabase is offline.
