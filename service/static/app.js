// app.js (ES module) — frontend logic to call the FastAPI endpoints.
// Place at service/static/app.js
const $ = sel => document.querySelector(sel);
const fileInput = $('#fileInput');
const pathInput = $('#pathInput');
const prefsInput = $('#prefsInput');
const analyzeBtn = $('#analyzeBtn');
const recommendBtn = $('#recommendBtn');
const compatBtn = $('#compatBtn');
const downloadBtn = $('#downloadBtn');
const jsonOutput = $('#jsonOutput');
const imgPreview = $('#imgPreview');
const imgMeta = $('#imgMeta');
const logArea = $('#logArea');
const modelInput = $('#modelInput');

let lastJson = null;

function log(msg, level='info') {
  const el = document.createElement('div');
  el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  el.style.opacity = 0.9;
  logArea.prepend(el);
}

// helper to display JSON prettily
function showJSON(obj) {
  try {
    const text = JSON.stringify(obj, null, 2);
    jsonOutput.textContent = text;
    lastJson = text;
  } catch (e) {
    jsonOutput.textContent = String(obj);
    lastJson = String(obj);
  }
}

// preview the selected file locally
fileInput.addEventListener('change', (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) {
    imgPreview.src = '';
    imgMeta.textContent = '';
    return;
  }
  const url = URL.createObjectURL(f);
  imgPreview.src = url;
  imgMeta.textContent = `${f.name} • ${Math.round(f.size/1024)} KB`;
});

// universal uploader helper
async function postImageTo(endpoint, file, extraForm = {}) {
  const fd = new FormData();
  if (file) fd.append('file', file, file.name || 'upload.png');
  Object.entries(extraForm).forEach(([k,v]) => { if (v !== undefined && v !== null) fd.append(k, v); });
  const res = await fetch(endpoint, { method: 'POST', body: fd });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

// analyze button
analyzeBtn.addEventListener('click', async () => {
  log('Analyze clicked');
  const file = fileInput.files[0];
  const path = pathInput.value.trim();
  try {
    let result;
    if (file) {
      result = await postImageTo('/analyze', file);
    } else if (path) {
      // server path analyze: call compatibility endpoint with a single-file path? we have analyze only for uploaded images,
      // so use recommend endpoint fallback using path (recommend accepts image or dataset path in the backend).
      result = await fetch('/analyze', {
        method: 'POST',
        body: new FormData().append('file', new Blob(), 'noop') // fallback - but prefer uploading
      }).then(r => r.json()).catch(()=> ({status:'error', detail:'No file uploaded and server-side path analyze unsupported.'}));
    } else {
      throw new Error('Select a file or enter a path');
    }
    showJSON(result);
    log('Analyze successful');
  } catch (err) {
    log('Analyze failed: ' + err.message);
    showJSON({error: err.message});
  }
});

// recommend button
recommendBtn.addEventListener('click', async () => {
  log('Recommend clicked');
  const file = fileInput.files[0];
  const prefs = prefsInput.value.trim();
  try {
    let result;
    if (file) {
      const fd = new FormData();
      fd.append('file', file);
      if (prefs) fd.append('prefs', prefs);
      const res = await fetch('/recommend', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      result = await res.json();
    } else if (pathInput.value.trim()) {
      // dataset recommendation via path -> call /compatibility? recommend_dataset is server side, but we expose /compatibility.
      const res = await fetch('/compatibility', {
        method: 'POST',
        body: new URLSearchParams({model: modelInput.value || 'unknown', dataset_path: pathInput.value.trim()})
      });
      result = await res.json();
    } else {
      throw new Error('Select file or dataset path');
    }
    showJSON(result);
    log('Recommend successful');
  } catch (err) {
    log('Recommend failed: ' + err.message);
    showJSON({error: err.message});
  }
});

// compatibility button
compatBtn.addEventListener('click', async () => {
  log('Compatibility check');
  const model = modelInput.value.trim();
  const ds = pathInput.value.trim();
  if (!model || !ds) {
    showJSON({error: 'Please provide model and dataset_path (server-side) in the sidebar.'});
    return;
  }
  try {
    const res = await fetch('/compatibility', {
      method: 'POST',
      body: new URLSearchParams({ model, dataset_path: ds })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    showJSON(json);
    log('Compatibility done');
  } catch (err) {
    log('Compatibility failed: ' + err.message);
    showJSON({error: err.message});
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

