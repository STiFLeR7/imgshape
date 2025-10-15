// app.js — React + Bootstrap + Framer Motion “Liquid Glass” UI frontend
import React, { useState } from "https://cdn.skypack.dev/react";
import ReactDOM from "https://cdn.skypack.dev/react-dom";
import { motion, AnimatePresence } from "https://cdn.skypack.dev/framer-motion";

function Topbar({ onAnalyze }) {
  return (
    <div className="topbar liquid-glass">
      <div className="brand">
        <div className="brand-logo"></div>
        <div>
          <div className="app-name">imgshape</div>
          <div className="version">v3.0.0 • Liquid Glass UI</div>
        </div>
      </div>
      <div className="d-flex gap-2">
        <input id="apiBase" className="form-control" placeholder="API base" />
        <button className="btn btn-primary" onClick={onAnalyze}>Analyze</button>
      </div>
    </div>
  );
}

function Dropzone({ onSelect }) {
  return (
    <div
      className="dropzone"
      onClick={() => document.getElementById("fileInput").click()}
    >
      <input
        id="fileInput"
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => onSelect(e.target.files[0])}
      />
      <div>Drop image or click to choose</div>
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [json, setJson] = useState(null);
  const [progress, setProgress] = useState(0);

  async function analyze() {
    if (!file) {
      setJson({ error: "No file selected" });
      return;
    }
    setProgress(20);
    await new Promise((r) => setTimeout(r, 500));
    setProgress(60);
    await new Promise((r) => setTimeout(r, 700));
    setJson({
      message: "Analysis complete",
      name: file.name,
      sizeKB: Math.round(file.size / 1024),
    });
    setProgress(100);
    setTimeout(() => setProgress(0), 500);
  }

  return (
    <div>
      <Topbar onAnalyze={analyze} />

      <div className="container mt-4 d-flex gap-3">
        <div className="card col-md-4 liquid-glass">
          <Dropzone onSelect={setFile} />
          <div className="mt-3">
            <label className="form-label small text-muted">Preferences</label>
            <input className="form-control" placeholder="resize=640,augment=true" />
          </div>
          <button className="btn btn-primary w-100 mt-3" onClick={analyze}>
            Run Analysis
          </button>
        </div>

        <div className="col">
          <div className="card liquid-glass mb-3">
            <div className="preview-area">
              <AnimatePresence>
                {file ? (
                  <motion.img
                    key={file.name}
                    src={URL.createObjectURL(file)}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                  />
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-muted"
                  >
                    Drop an image to preview
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <div className="progress mt-2" style={{ height: 8 }}>
              <div
                className="progress-bar bg-info"
                style={{ width: `${progress}%`, transition: "width 0.3s" }}
              ></div>
            </div>
          </div>

          <div className="card liquid-glass">
            <div className="fw-bold mb-2">Output JSON</div>
            <pre className="json">
              {json ? JSON.stringify(json, null, 2) : "Awaiting analysis..."}
            </pre>
          </div>
        </div>
      </div>

      <footer>Made with ☕ and Liquid Glass — imgshape</footer>
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
