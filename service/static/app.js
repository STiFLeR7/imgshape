// app.js (ES module) — upgraded UI behaviors for imgshape v2.2.0
// Keeps the same backend endpoints: /analyze, /recommend, /compatibility
// Improvements: drag & drop, apiBase input, progress UI, cancel support, pretty/raw toggle, copy/download, nicer logs.

const $ = sel => document.querySelector(sel);
const fileInput = $('#fileInput');
const pathInput = $('#pathInput');
const prefsInput = $('#prefsInput');
const analyzeBtn = $('#analyzeBtn');
const recommendBtn = $('#recommendBtn');
const compatBtn = $('#compatBtn');
const downloadBtn = $('#downloadBtn');
const copyBtn = $('#copyBtn');
const jsonOutput = $('#jsonOutput');
const imgPreview = $('#imgPreview');
const imgMeta = $('#imgMeta');
const logArea = $('#logArea');
const modelInput = $('#modelInput');
const apiBaseInput = $('#apiBase');
const previewSmall = $('#previewSmall');
const dropzone = $('#dropzone');
const progressInner = document.querySelector('.progress-inner');
const statusBadge = $('#statusBadge');
const prettyBtn = $('#prettyBtn');
const rawBtn = $('#rawBtn');

let lastJson = null;
let xhrController = null;
let currentMode = 'pretty'; // 'pretty' or 'raw'

function log(msg, level='info') {
  const el = document.createElement('div');
  const t = new Date().toLocaleTimeString();
  el.innerHTML = `<span style="color:var(--muted);font-size:12px">[${t}]</span> ${escapeHtml(msg)}`;
  logArea.prepend(el);
}

// safe small escape for logs
function escapeHtml(s){ return String(s).replace(/[&<>"'`]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;", '`':'&#96;'})[c]); }

// pretty-print JSON into the code block with light coloring (simple)
function prettyPrint(obj) {
  try {
    const json = typeof obj === 'string' ? JSON.parse(obj) : obj;
    const text = JSON.stringify(json, null, 2);
    lastJson = text;
    jsonOutput.textContent = text;
    return text;
  } catch (e) {
    jsonOutput.textContent = String(obj);
    lastJson = String(obj);
    return lastJson;
  }
}

function rawPrint(text) {
  jsonOutput.textContent = text;
  lastJson = text;
}

function setStatus(s) {
  statusBadge.textContent = s;
}

// small helper to get api base (default to '/')
function apiBase() {
  const v = (apiBaseInput.value || '').trim();
  if (!v) return '';
  // if user provided a base without protocol, keep it as-is (could be relative)
  return v.endsWith('/') ? v.slice(0,-1) : v;
}

function setProgress(p) {
  progressInner.style.width = `${Math.max(0, Math.min(100, p))}%`;
}

// handle preview and metadata for chosen file
fileInput.addEventListener('change', (ev) => {
  const f = ev.target.files && ev.target.files[0];
  handleFileSelection(f);
});

function handleFileSelection(f) {
  if (!f) {
    imgPreview.src = '';
    previewSmall.style.backgroundImage = '';
    imgMeta.textContent = '';
    return;
  }
  const url = URL.createObjectURL(f);
  imgPreview.src = url;
  previewSmall.style.backgroundImage = `url('${url}')`;
  imgMeta.textContent = `${f.name} • ${Math.round(f.size/1024)} KB • ${f.type || 'image'}`;
}

// drag & drop
['dragenter','dragover'].forEach(ev => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.add('dragover');
  });
});
['dragleave','drop','dragend'].forEach(ev => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.remove('dragover');
  });
});
dropzone.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; handleFileSelection(f); }
});

// small fetch wrapper that supports timeouts / cancellations
async function postImageTo(endpoint, file, extraForm = {}, onProgress) {
  const url = `${apiBase()}${endpoint}`;
  const fd = new FormData();
  if (file) fd.append('file', file, file.name || 'upload.png');
  Object.entries(extraForm).forEach(([k,v]) => { if (v !== undefined && v !== null) fd.append(k, v); });

  // use fetch with AbortController for cancellation; progress with XHR (fallback)
  if (window.fetch && window.AbortController) {
    xhrController = new AbortController();
    setStatus('uploading');
    setProgress(8);
    try {
      const res = await fetch(url, { method: 'POST', body: fd, signal: xhrController.signal });
      setProgress(60);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
      }
      const json = await res.json();
      setProgress(100);
      setStatus('done');
      setTimeout(()=>setProgress(0), 300);
      return json;
    } catch (err) {
      setProgress(0);
      setStatus('idle');
      throw err;
    } finally {
      xhrController = null;
    }
  } else {
    // (older browsers) fallback to XMLHttpRequest to allow progress events
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url);
      xhr.onload = () => {
        try {
          const json = JSON.parse(xhr.responseText);
          resolve(json);
        } catch (e) { resolve(xhr.responseText); }
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) {
          const p = Math.round((ev.loaded/ev.total) * 90);
          setProgress(p);
        }
      };
      xhr.send(fd);
    });
  }
}

// analyze action
analyzeBtn.addEventListener('click', async () => {
  log('Analyze clicked');
  setStatus('sending');
  setProgress(5);
  const file = fileInput.files[0];
  const path = pathInput.value.trim();
  try {
    let result;
    if (file) {
      result = await postImageTo('/analyze', file);
    } else if (path) {
      // if backend supports analyze-by-path, call it; otherwise compatibility fallback
      const res = await fetch(`${apiBase()}/analyze`, { method: 'POST', body: new URLSearchParams({ dataset_path: path }) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      result = await res.json();
    } else {
      throw new Error('Select a file or enter a server-side path');
    }
    prettyPrint(result);
    log('Analyze successful');
  } catch (err) {
    const em = err && err.message ? err.message : String(err);
    log('Analyze failed: ' + em);
    rawPrint(JSON.stringify({ error: em }, null, 2));
  } finally {
    setStatus('idle');
    setProgress(0);
  }
});

// recommend action
recommendBtn.addEventListener('click', async () => {
  log('Recommend clicked');
  setStatus('sending');
  setProgress(5);
  const file = fileInput.files[0];
  const prefs = prefsInput.value.trim();
  try {
    let result;
    if (file) {
      result = await postImageTo('/recommend', file, { prefs });
    } else if (pathInput.value.trim()) {
      const res = await fetch(`${apiBase()}/recommend`, {
        method: 'POST',
        body: new URLSearchParams({ dataset_path: pathInput.value.trim(), prefs })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      result = await res.json();
    } else {
      throw new Error('Select file or dataset path');
    }
    prettyPrint(result);
    log('Recommend successful');
  } catch (err) {
    log('Recommend failed: ' + (err.message || String(err)));
    rawPrint(JSON.stringify({ error: err.message || String(err) }, null, 2));
  } finally {
    setStatus('idle');
    setProgress(0);
  }
});

// compatibility action
compatBtn.addEventListener('click', async () => {
  log('Compatibility check');
  const model = modelInput.value.trim();
  const ds = pathInput.value.trim();
  if (!model || !ds) {
    rawPrint(JSON.stringify({ error: 'Please provide model and dataset_path in sidebar.' }, null, 2));
    return;
  }
  setStatus('sending');
  setProgress(6);
  try {
    const res = await fetch(`${apiBase()}/compatibility`, {
      method: 'POST',
      body: new URLSearchParams({ model, dataset_path: ds })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    prettyPrint(json);
    log('Compatibility done');
  } catch (err) {
    log('Compatibility failed: ' + (err.message || String(err)));
    rawPrint(JSON.stringify({ error: err.message || String(err) }, null, 2));
  } finally {
    setStatus('idle');
    setProgress(0);
  }
});

// download JSON
downloadBtn.addEventListener('click', () => {
  if (!lastJson) { log('No JSON to download'); return; }
  const blob = new Blob([lastJson], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'imgshape_report.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  log('Downloaded JSON report');
});

// copy JSON
copyBtn.addEventListener('click', async () => {
  if (!lastJson) { log('No JSON to copy'); return; }
  try {
    await navigator.clipboard.writeText(lastJson);
    log('Copied JSON to clipboard');
  } catch (e) {
    log('Copy failed: ' + (e.message || String(e)));
  }
});

// pretty/raw toggle
prettyBtn.addEventListener('click', () => { currentMode = 'pretty'; if (lastJson) prettyPrint(lastJson); });
rawBtn.addEventListener('click', () => { currentMode = 'raw'; if (lastJson) rawPrint(lastJson); });

// very small helper to pretty print if possible
function tryPrettyStored() {
  if (!lastJson) return;
  if (currentMode === 'pretty') prettyPrint(lastJson);
  else rawPrint(lastJson);
}

// initial UI state
setStatus('idle');
setProgress(0);

// simple keyboard shortcut: Ctrl+K focus api
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key.toLowerCase() === 'k') { e.preventDefault(); apiBaseInput.focus(); log('Focus API base (Ctrl+K)'); }
});

// small utility to attempt reading image dimensions
imgPreview.addEventListener('load', () => {
  try {
    const w = imgPreview.naturalWidth;
    const h = imgPreview.naturalHeight;
    if (w && h && imgMeta.textContent) imgMeta.textContent += ` • ${w}×${h}px`;
  } catch (e){}
});

// expose a simple abort/cancel by clearing controller
// if you want an explicit cancel button later, wire this to it
window.cancelCurrent = () => {
  if (xhrController) {
    try { xhrController.abort(); log('Request aborted by user'); setStatus('idle'); setProgress(0); }
    catch(e){ log('Abort failed: ' + (e.message || e)); }
  } else log('No active request to cancel');
};

// small helper to load example image via URL (dev convenience)
window.loadExample = (url) => {
  fetch(url).then(r => r.blob()).then(b => {
    const f = new File([b], 'example.png', { type: b.type });
    const data = new DataTransfer();
    data.items.add(f);
    fileInput.files = data.files;
    handleFileSelection(f);
  }).catch(e => log('Example load failed: ' + e.message));
};

// Quick hack: if the page is hosted at a non-root, keep apiBase in sync with the page origin by default
if (!apiBaseInput.value) {
  const originGuess = (location.origin !== 'null') ? location.origin : '';
  apiBaseInput.value = originGuess;
  // allow empty too; user can clear to use relative paths
}

// small helper: if JSON arrives as object, store it and show in preferred mode
function showAndStore(json) {
  try {
    const text = typeof json === 'string' ? json : JSON.stringify(json, null, 2);
    lastJson = text;
    tryPrettyStored();
  } catch (e) {
    lastJson = String(json);
    rawPrint(lastJson);
  }
}

// override prettyPrint to also set lastJson if object passed
const origPretty = prettyPrint;
prettyPrint = (obj) => { const t = origPretty(obj); lastJson = t; return t; };

log('UI ready — drag an image or pick a file. Use API base to point to your GCP endpoint.');
