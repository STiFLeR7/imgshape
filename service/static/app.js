// app.js — imgshape v4.0.0 Atlas UI
// Vanilla JS implementation with v3/v4 version switching

/* ------------------------------------------------------------------
   Helpers & API Functions
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

function getSelectedVersion() {
  const select = document.getElementById("versionSelect");
  return select ? select.value : "v4";
}

// v3 API calls
async function postAnalyze(file, datasetPath, prefs) {
  const apiBase = getApiBase();
  const form = new FormData();
  if (file) form.append("file", file);
  if (datasetPath) form.append("dataset_path", datasetPath);
  if (prefs) form.append("prefs", prefs);
  const res = await fetch(`${apiBase}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return await res.json();
}

async function postRecommend(file, datasetPath, prefs) {
  const apiBase = getApiBase();
  const form = new FormData();
  if (file) form.append("file", file);
  if (datasetPath) form.append("dataset_path", datasetPath);
  if (prefs) form.append("prefs", prefs);
  const res = await fetch(`${apiBase}/recommend`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return await res.json();
}

async function postCompatibility(file, datasetPath, modelName) {
  const apiBase = getApiBase();
  const form = new FormData();
  if (file) form.append("file", file);
  if (datasetPath) form.append("dataset_path", datasetPath);
  if (modelName) form.append("model_name", modelName);
  const res = await fetch(`${apiBase}/compatibility`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return await res.json();
}

// v4 API calls
async function postV4Fingerprint(file, datasetPath) {
  const apiBase = getApiBase();
  const form = new FormData();
  if (file) form.append("file", file);
  if (datasetPath) form.append("dataset_path", datasetPath);
  const res = await fetch(`${apiBase}/v4/fingerprint`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return await res.json();
}

async function postV4Analyze(file, datasetPath, task, deployment, priority, maxSize) {
  const apiBase = getApiBase();
  const form = new FormData();
  if (file) form.append("file", file);
  if (datasetPath) form.append("dataset_path", datasetPath);
  if (task) form.append("task", task);
  if (deployment) form.append("deployment", deployment);
  if (priority) form.append("priority", priority);
  if (maxSize) form.append("max_model_size_mb", maxSize);
  const res = await fetch(`${apiBase}/v4/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return await res.json();
}

async function getV4Info() {
  const apiBase = getApiBase();
  const res = await fetch(`${apiBase}/v4/info`);
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return await res.json();
}

/* ------------------------------------------------------------------
   UI State & Event Handlers
------------------------------------------------------------------ */

let currentFile = null;
let currentResult = null;
let progressTimer = null;

function log(msg, type = "info") {
  const logArea = document.getElementById("logArea");
  if (!logArea) return;
  const time = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.className = `log-entry ${type}`;
  line.innerHTML = `<span class="time">${time}</span> <span class="msg">${msg}</span>`;
  logArea.appendChild(line);
  logArea.scrollTop = logArea.scrollHeight;
}

function setStatus(status) {
  const badge = document.getElementById("statusBadge");
  if (!badge) return;
  badge.textContent = status;
  badge.className = `badge ${
    status === "error" ? "error" : status === "done" ? "success" : "muted"
  }`;
}

function setProgress(percent) {
  const bar = document.getElementById("progress");
  if (!bar) return;
  const inner = bar.querySelector(".progress-inner");
  if (inner) {
    inner.style.width = `${percent}%`;
    bar.style.display = percent > 0 ? "block" : "none";
  }
}

function displayJSON(data, pretty = true) {
  const output = document.getElementById("jsonOutput");
  if (!output) return;
  currentResult = data;
  if (pretty) {
    output.textContent = JSON.stringify(data, null, 2);
  } else {
    output.textContent = JSON.stringify(data);
  }
}

function displayV4Fingerprint(data) {
  log("Received v4 fingerprint", "info");
  displayJSON(data, true);
  
  // Could add visual panels for each profile here
  log(`Fingerprint class: ${data.class || "unknown"}`, "success");
  log(`Profiles: Spatial, Signal, Distribution, Quality, Semantic`, "info");
}

function displayV4Analysis(data) {
  log("Received v4 analysis with decisions", "info");
  displayJSON(data, true);
  
  if (data.decisions && data.decisions.length > 0) {
    log(`${data.decisions.length} decisions made`, "success");
    data.decisions.forEach((d, i) => {
      log(`  ${i + 1}. ${d.domain}: ${d.decision.value} (confidence: ${d.decision.confidence})`, "info");
    });
  }
  
  if (data.artifacts) {
    log(`Artifacts generated: ${Object.keys(data.artifacts).length}`, "success");
  }
}

function displayV3Result(data) {
  log("Received v3 result", "info");
  displayJSON(data, true);
}

/* ------------------------------------------------------------------
   File Handling
------------------------------------------------------------------ */

function handleFileSelect(file) {
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    log("File too large (max 10MB)", "error");
    return;
  }
  
  currentFile = file;
  log(`File selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, "info");
  
  // Update preview
  const imgPreview = document.getElementById("imgPreview");
  const previewSmall = document.getElementById("previewSmall");
  if (imgPreview && previewSmall) {
    const url = URL.createObjectURL(file);
    imgPreview.src = url;
    imgPreview.style.display = "block";
    previewSmall.style.backgroundImage = `url(${url})`;
    previewSmall.style.display = "block";
    
    // Show metadata
    const img = new Image();
    img.onload = () => {
      const meta = document.getElementById("imgMeta");
      if (meta) {
        meta.innerHTML = `${img.width} × ${img.height} px • ${file.type}`;
      }
    };
    img.src = url;
  }
}

function setupDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  
  if (!dropzone || !fileInput) return;
  
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });
  
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
      handleFileSelect(e.target.files[0]);
    }
  });
}

/* ------------------------------------------------------------------
   Main Actions
------------------------------------------------------------------ */

async function handleAnalyze() {
  const version = getSelectedVersion();
  const datasetPath = document.getElementById("pathInput")?.value.trim();
  
  if (!currentFile && !datasetPath) {
    log("Please select a file or enter dataset path", "error");
    return;
  }
  
  setStatus("analyzing");
  setProgress(20);
  
  try {
    if (version === "v4") {
      log("Running v4 analysis...", "info");
      const task = document.getElementById("taskInput")?.value || "classification";
      const deployment = document.getElementById("deploymentInput")?.value || "cloud";
      const priority = document.getElementById("priorityInput")?.value || "balanced";
      const maxSize = document.getElementById("maxSizeInput")?.value;
      
      setProgress(40);
      const result = await postV4Analyze(currentFile, datasetPath, task, deployment, priority, maxSize);
      setProgress(100);
      displayV4Analysis(result);
      setStatus("done");
    } else {
      log("Running v3 analysis...", "info");
      const prefs = document.getElementById("prefsInput")?.value || "";
      
      setProgress(40);
      const result = await postAnalyze(currentFile, datasetPath, prefs);
      setProgress(100);
      displayV3Result(result);
      setStatus("done");
    }
  } catch (err) {
    log(`Error: ${err.message}`, "error");
    setStatus("error");
  } finally {
    setTimeout(() => setProgress(0), 2000);
  }
}

async function handleFingerprint() {
  const datasetPath = document.getElementById("pathInput")?.value.trim();
  
  if (!currentFile && !datasetPath) {
    log("Please select a file or enter dataset path", "error");
    return;
  }
  
  setStatus("fingerprinting");
  setProgress(20);
  
  try {
    log("Extracting v4 fingerprint...", "info");
    setProgress(40);
    const result = await postV4Fingerprint(currentFile, datasetPath);
    setProgress(100);
    displayV4Fingerprint(result);
    setStatus("done");
  } catch (err) {
    log(`Error: ${err.message}`, "error");
    setStatus("error");
  } finally {
    setTimeout(() => setProgress(0), 2000);
  }
}

async function handleRecommend() {
  const datasetPath = document.getElementById("pathInput")?.value.trim();
  const prefs = document.getElementById("prefsInput")?.value || "";
  
  if (!currentFile && !datasetPath) {
    log("Please select a file or enter dataset path", "error");
    return;
  }
  
  setStatus("recommending");
  setProgress(20);
  
  try {
    log("Getting recommendations...", "info");
    setProgress(40);
    const result = await postRecommend(currentFile, datasetPath, prefs);
    setProgress(100);
    displayV3Result(result);
    setStatus("done");
  } catch (err) {
    log(`Error: ${err.message}`, "error");
    setStatus("error");
  } finally {
    setTimeout(() => setProgress(0), 2000);
  }
}

async function handleCompatibility() {
  const datasetPath = document.getElementById("pathInput")?.value.trim();
  const modelName = document.getElementById("modelInput")?.value || "";
  
  if (!currentFile && !datasetPath) {
    log("Please select a file or enter dataset path", "error");
    return;
  }
  
  if (!modelName) {
    log("Please enter a model name", "error");
    return;
  }
  
  setStatus("checking");
  setProgress(20);
  
  try {
    log(`Checking compatibility with ${modelName}...`, "info");
    setProgress(40);
    const result = await postCompatibility(currentFile, datasetPath, modelName);
    setProgress(100);
    displayV3Result(result);
    setStatus("done");
  } catch (err) {
    log(`Error: ${err.message}`, "error");
    setStatus("error");
  } finally {
    setTimeout(() => setProgress(0), 2000);
  }
}

function handleDownload() {
  if (!currentResult) {
    log("No data to download", "error");
    return;
  }
  
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `imgshape_result_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  log("Downloaded result as JSON", "success");
}

function handleCopy() {
  if (!currentResult) {
    log("No data to copy", "error");
    return;
  }
  
  navigator.clipboard.writeText(JSON.stringify(currentResult, null, 2))
    .then(() => log("Copied to clipboard", "success"))
    .catch(() => log("Failed to copy", "error"));
}

function handlePretty() {
  if (currentResult) {
    displayJSON(currentResult, true);
  }
}

function handleRaw() {
  if (currentResult) {
    displayJSON(currentResult, false);
  }
}

/* ------------------------------------------------------------------
   Version Switching
------------------------------------------------------------------ */

function handleVersionChange() {
  const version = getSelectedVersion();
  const v3Options = document.getElementById("v3Options");
  const v4Options = document.getElementById("v4Options");
  const v3Actions = document.getElementById("v3Actions");
  const fingerprintBtn = document.getElementById("fingerprintBtn");
  
  if (version === "v4") {
    if (v3Options) v3Options.style.display = "none";
    if (v4Options) v4Options.style.display = "block";
    if (v3Actions) v3Actions.style.display = "none";
    if (fingerprintBtn) fingerprintBtn.style.display = "inline-block";
    log("Switched to v4 (Atlas) mode", "info");
  } else {
    if (v3Options) v3Options.style.display = "block";
    if (v4Options) v4Options.style.display = "none";
    if (v3Actions) v3Actions.style.display = "flex";
    if (fingerprintBtn) fingerprintBtn.style.display = "none";
    log("Switched to v3 (Legacy) mode", "info");
  }
}

/* ------------------------------------------------------------------
   Initialization
------------------------------------------------------------------ */

document.addEventListener("DOMContentLoaded", () => {
  log("imgshape v4.0.0 Atlas initialized", "success");
  
  // Check v4 availability
  if (window.IMGSHAPE && window.IMGSHAPE.v4Available) {
    log("v4 API available ✓", "success");
  } else {
    log("v4 API not available (v3 only)", "info");
  }
  
  setupDropzone();
  
  // Event listeners
  const analyzeBtn = document.getElementById("analyzeBtn");
  const fingerprintBtn = document.getElementById("fingerprintBtn");
  const recommendBtn = document.getElementById("recommendBtn");
  const compatBtn = document.getElementById("compatBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  const copyBtn = document.getElementById("copyBtn");
  const prettyBtn = document.getElementById("prettyBtn");
  const rawBtn = document.getElementById("rawBtn");
  const versionSelect = document.getElementById("versionSelect");
  
  if (analyzeBtn) analyzeBtn.addEventListener("click", handleAnalyze);
  if (fingerprintBtn) fingerprintBtn.addEventListener("click", handleFingerprint);
  if (recommendBtn) recommendBtn.addEventListener("click", handleRecommend);
  if (compatBtn) compatBtn.addEventListener("click", handleCompatibility);
  if (downloadBtn) downloadBtn.addEventListener("click", handleDownload);
  if (copyBtn) copyBtn.addEventListener("click", handleCopy);
  if (prettyBtn) prettyBtn.addEventListener("click", handlePretty);
  if (rawBtn) rawBtn.addEventListener("click", handleRaw);
  if (versionSelect) {
    versionSelect.addEventListener("change", handleVersionChange);
    handleVersionChange(); // Initialize
  }
  
  log("UI ready", "info");
});
