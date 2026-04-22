# Design System: imgshape (Atlas Intelligence Platform)

imgshape is a high-performance Data-Centric AI toolkit and PaaS. The design system balances the technical utility of a PyPi package (like PyTorch) with the premium, scalable feel of a vision-intelligence platform.

## 🏛️ Brand Identity & Philosophy
- **Core Identity:** "The Data Layer for Vision AI."
- **Dual Nature:** 
  - **The Toolkit:** Fast, deterministic, open-source (PyPi).
  - **The Platform:** Collaborative, hosted, enterprise-ready (PaaS).
- **Visual Style:** "Cyber-Premium Bento" — High-density information wrapped in a clean, professional dark-mode aesthetic.

## 🎨 Colors
- **Background (Deep Space):** `#050714` (Darker than before for higher contrast)
- **Surface (Card/Nav):** `#0D1117`
- **Surface Highlight:** `#161B22`
- **Accent (Brand Blue):** `#3B82F6` (Electric Blue)
- **Secondary (PaaS Purple):** `#8B5CF6` (Used for Cloud/Platform features)
- **Action (Emerald):** `#10B981` (Used for "Deploy" and "Run")
- **Status Amber:** `#F59E0B` (For Drift/Warning metrics)
- **Text Primary:** `#F9FAFB`
- **Text Secondary:** `#8B949E`
- **Borders:** `#30363D`

## ⌨️ Typography
- **Headings:** 'Inter', sans-serif (Bold, -0.02em tracking).
- **Body:** 'Inter', sans-serif (Regular/Medium).
- **Technical/Code:** 'JetBrains Mono' (For CLI commands and API snippets).
- **Scale:**
  - Hero H1: 48px to 64px.
  - Section Headers: 32px.
  - Card Titles: 18px.

## 🗺️ Website Structure (STITCH Layout)

### 1. 🚀 The Hero Section (The Hook)
- **Headline:** "Build Better Vision Models with Clean Data."
- **Subheadline:** "Deterministic fingerprinting, semantic drift detection, and automated preprocessing—from local CLI to Cloud PaaS."
- **Primary CTA:** `pip install imgshape` (Click to copy).
- **Secondary CTA:** "Launch Cloud Dashboard" (Accent Border).

### 2. 📦 The Installation Matrix (PyTorch Style)
- **Concept:** An interactive grid for users to select their environment.
- **Selectors:** 
  - **Build:** Stable / Nightly
  - **OS:** Linux / Mac / Windows
  - **Package:** Pip / Docker / Source
  - **Compute:** CPU / CUDA (GPU)
- **Output:** A dynamic CLI command box (e.g., `pip install "imgshape[full]"`).

### 3. 🍱 The Atlas Bento Dashboard (The Product)
- **Concept:** A "Live" preview of the dashboard.
- **Key Modules to Show:**
  - **Semantic Drift:** A visual scatter plot of dataset versions.
  - **Audit Results:** A checklist of data health (Channels, Resolution, Corruption).
  - **Pipeline Export:** A code snippet preview (Python/PyTorch).

### 4. ☁️ PaaS / Platform Section (The Scalability)
- **Headline:** "Atlas Cloud: Vision Intelligence at Scale."
- **Features:** 
  - **Hosted Analysis:** No-setup dataset auditing.
  - **Team Collaboration:** Share fingerprints and audit reports.
  - **API-First:** Integrate data health checks into CI/CD pipelines.

### 5. 🛠️ Technical Ecosystem
- A section dedicated to integrations: **PyTorch**, **TensorFlow**, **Albumentations**, and **HuggingFace**.

## 📐 Layout & Components
- **Bento Grid:** Use a 12-column grid for the "Features" section. 
- **Card Radius:** 16px (Premium roundness).
- **Gradients:** Subtle "Accent-to-Transparent" glows in corners of cards to indicate activity.
- **Motion:** Staggered entry for Bento cards; "Typing" animation for CLI examples.

## 🤝 Community & Trust
- **GitHub Stats:** Display Star count and Fork count.
- **Logos:** Grayscale logos of compatible frameworks (PyTorch, TF, etc.).
