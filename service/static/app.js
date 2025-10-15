// app.js — React + Bootstrap + Framer Motion “Liquid Glass” UI frontend
import React, { useState } from "https://cdn.skypack.dev/react";
import ReactDOM from "https://cdn.skypack.dev/react-dom";
import { motion, AnimatePresence } from "https://cdn.skypack.dev/framer-motion";

/* ------------------------------------------------------------------
   Helpers
------------------------------------------------------------------ */

// Detect API base dynamically from FastAPI-injected endpoints
function getApiBase() {
  try {
    if (window.IMGSHAPE && window.IMGSHAPE.analyze) {
      const url = window.IMGSHAPE.analyze;
      return url.replace(/\/analyze\/?$/, "");
    }
  } catch (_) {}
  return document.getElementById("apiBase")?.value || "/";
}

async function postAnalyze(file) {
  const apiBase = getApiBase();
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${apiBase}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return await res.json();
}

/* ------------------------------------------------------------------
   Components
------------------------------------------------------------------ */

function Topbar({ onAnalyze }) {
  return (
    <div className="topbar liquid-glass d-flex align-items-center justify-content-between p-2">
      <div className="brand d-flex align-items-center gap-2">
        <img
          src="/static/assets/sample_images/imgshape_lg.png"
          width="40"
          height="40"
          alt="imgshape"
          className="rounded"
        />
        <div>
          <div className="app-name fw-bold">imgshape</div>
          <div className="version small text-muted">v3.1.0 • Liquid Glass</div>
        </div>
      </div>
      <div className="d-flex gap-2">
        <input
          id="apiBase"
          className="form-control form-control-sm"
          placeholder="API base"
          style={{ width: 160 }}
        />
        <button className="btn btn-sm btn-primary" onClick={onAnalyze}>
          Analyze
        </button>
      </div>
    </div>
  );
}

function Dropzone({ onSelect }) {
  return (
    <div
      className="dropzone p-4 text-center border border-info rounded"
      onClick={() => document.getElementById("fileInput").click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) onSelect(e.dataTransfer.files[0]);
      }}
    >
      <input
        id="fileInput"
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => onSelect(e.target.files[0])}
      />
      <div className="text-muted">Drop image or click to choose</div>
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [json, setJson] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle");

  async function analyze() {
    if (!file) {
      setJson({ error: "No file selected" });
      return;
    }
    setStatus("uploading");
    setProgress(25);

    try {
      const result = await postAnalyze(file);
      setProgress(80);
      setStatus("processing");
      await new Promise((r) => setTimeout(r, 400));
      setJson(result);
      setProgress(100);
      setStatus("done");
    } catch (e) {
      console.error(e);
      setJson({ error: e.message || "Failed to analyze image" });
      setStatus("error");
    } finally {
      setTimeout(() => setProgress(0), 1200);
    }
  }

  return (
    <div>
      <Topbar onAnalyze={analyze} />

      <div className="container mt-4 d-flex flex-wrap gap-3 justify-content-center">
        <div className="card col-md-4 col-sm-12 p-3 liquid-glass shadow">
          <Dropzone onSelect={setFile} />
          <div className="mt-3">
            <label className="form-label small text-muted">Preferences</label>
            <input
              className="form-control"
              placeholder="resize=640,augment=true"
            />
          </div>
          <button
            className="btn btn-primary w-100 mt-3"
            onClick={analyze}
            disabled={status === "uploading"}
          >
            {status === "uploading" ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>

        <div className="col-md-7 col-sm-12">
          <div className="card p-3 liquid-glass mb-3 shadow">
            <div className="fw-bold mb-2">Preview</div>
            <div
              className="preview-area text-center"
              style={{ minHeight: 240, position: "relative" }}
            >
              <AnimatePresence>
                {file ? (
                  <motion.img
                    key={file.name}
                    src={URL.createObjectURL(file)}
                    alt="preview"
                    className="img-fluid rounded"
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
            <div className="progress mt-3" style={{ height: 8 }}>
              <div
                className={`progress-bar ${
                  status === "error" ? "bg-danger" : "bg-info"
                }`}
                style={{ width: `${progress}%`, transition: "width 0.4s" }}
              ></div>
            </div>
          </div>

          <div className="card p-3 liquid-glass shadow">
            <div className="fw-bold mb-2">Output JSON</div>
            <pre className="json small text-wrap">
              {json
                ? JSON.stringify(json, null, 2)
                : "Awaiting analysis..."}
            </pre>
          </div>
        </div>
      </div>

      <footer className="text-center text-muted small mt-4 mb-3">
        Made with ☕ and Liquid Glass — imgshape v3.1.0
      </footer>
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
