// API Base URL config
const API_BASE = window.location.origin.includes('http') 
  ? window.location.origin 
  : 'http://127.0.0.1:5000';

// Global variables
let metricsChart = null;
let batchResults = [];
let currentPage = 1;
const rowsPerPage = 20;
let selectedFile = null;

// DOM Elements
const thresholdInput = document.getElementById('global-threshold');
const thresholdVal = document.getElementById('threshold-val');
const thresholdHint = document.getElementById('threshold-hint');
const modelSelector = document.getElementById('global-model-selector');

// Collapsible PCA Features
const pcaToggle = document.getElementById('pca-toggle');
const pcaContent = document.getElementById('pca-content');
const pcaContainer = document.getElementById('pca-inputs-container');

// Single Predict Form & Result Elements
const singleForm = document.getElementById('single-predict-form');
const singleResultBox = document.getElementById('single-result-box');
const resultBadge = document.getElementById('result-badge');
const resultBadgeIcon = document.getElementById('result-badge-icon');
const resultBadgeText = document.getElementById('result-badge-text');
const resultProbVal = document.getElementById('result-prob-val');
const resultProgressFill = document.getElementById('result-progress-fill');
const resultModelUsed = document.getElementById('result-model-used');
const resultThresholdUsed = document.getElementById('result-threshold-used');

// Upload Zone Elements
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('batch-file-input');
const fileDetails = document.getElementById('upload-file-details');
const fileNameLabel = document.getElementById('file-name-label');
const fileSizeLabel = document.getElementById('file-size-label');
const clearFileBtn = document.getElementById('btn-clear-file');
const analyzeBatchBtn = document.getElementById('btn-analyze-batch');

// Initialize the page
window.addEventListener('DOMContentLoaded', () => {
  generatePcaInputs();
  fetchMetrics();
  updateThresholdDesc(parseFloat(thresholdInput.value));
});

// Generate V1-V28 PCA Inputs
function generatePcaInputs() {
  pcaContainer.innerHTML = '';
  for (let i = 1; i <= 28; i++) {
    const fieldDiv = document.createElement('div');
    fieldDiv.className = 'pca-field';
    
    const label = document.createElement('label');
    label.htmlFor = `input-v${i}`;
    label.textContent = `V${i}`;
    
    const input = document.createElement('input');
    input.type = 'number';
    input.id = `input-v${i}`;
    input.className = 'form-input';
    input.value = '0.00';
    input.step = '0.01';
    
    fieldDiv.appendChild(label);
    fieldDiv.appendChild(input);
    pcaContainer.appendChild(fieldDiv);
  }
}

// Collapsible PCA trigger
pcaToggle.addEventListener('click', () => {
  pcaToggle.classList.toggle('active');
  pcaContent.classList.toggle('open');
});

// Reset PCA values
document.getElementById('btn-reset-pca').addEventListener('click', () => {
  for (let i = 1; i <= 28; i++) {
    document.getElementById(`input-v${i}`).value = '0.00';
  }
});

// Randomize PCA values (simulates a transaction footprint)
document.getElementById('btn-randomize-pca').addEventListener('click', () => {
  for (let i = 1; i <= 28; i++) {
    // Generate values between -3.0 and +3.0 with higher probability around 0
    const u1 = Math.random();
    const u2 = Math.random();
    const standardNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const val = (standardNormal * 1.5).toFixed(4);
    document.getElementById(`input-v${i}`).value = val;
  }
});

// Threshold slider interactivity
thresholdInput.addEventListener('input', (e) => {
  const val = parseFloat(e.target.value);
  thresholdVal.textContent = val.toFixed(2);
  updateThresholdDesc(val);
  updateThresholdMarker(val);
});

function updateThresholdDesc(val) {
  if (val < 0.25) {
    thresholdHint.textContent = "High Sensitivity (Sensitive): catches more fraud, increases false alarms";
    thresholdHint.style.color = "var(--danger)";
  } else if (val > 0.65) {
    thresholdHint.textContent = "Strict Verification (Strict): decreases false alarms, might miss fraud";
    thresholdHint.style.color = "var(--success)";
  } else {
    thresholdHint.textContent = "Standard settings: balanced precision & recall";
    thresholdHint.style.color = "var(--warning)";
  }
}

function updateThresholdMarker(val) {
  const marker = document.getElementById('threshold-marker');
  if (marker) {
    marker.style.left = `${val * 100}%`;
  }
}

// Fetch Metrics Summary
async function fetchMetrics() {
  const tableBody = document.getElementById('metrics-table-body');
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) throw new Error('API server returned status ' + res.status);
    
    const data = await res.json();
    tableBody.innerHTML = '';
    
    data.forEach(metric => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td><strong>${metric.model_name}</strong></td>
        <td class="text-right" style="font-weight: 600;">${metric.f1_fraud.toFixed(4)}</td>
        <td class="text-right" style="font-weight: 600; color: var(--primary);">${metric.roc_auc.toFixed(4)}</td>
      `;
      tableBody.appendChild(row);
    });
    
    renderMetricsChart(data);
  } catch (err) {
    console.error(err);
    tableBody.innerHTML = `
      <tr>
        <td colspan="3" class="text-center text-danger">
          <i class="fa-solid fa-circle-exclamation"></i> Connection to metrics API failed. Run Flask backend first!
        </td>
      </tr>
    `;
  }
}

// Render Chart
function renderMetricsChart(metrics) {
  const canvas = document.getElementById('metricsChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  const labels = metrics.map(m => m.model_name);
  const f1Scores = metrics.map(m => m.f1_fraud);
  const rocAucs = metrics.map(m => m.roc_auc);
  
  if (metricsChart) {
    metricsChart.destroy();
  }
  
  metricsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'F1-Score (Minority Class)',
          data: f1Scores,
          backgroundColor: 'rgba(239, 68, 68, 0.7)',
          borderColor: 'var(--danger)',
          borderWidth: 1.5,
          borderRadius: 6
        },
        {
          label: 'ROC-AUC',
          data: rocAucs,
          backgroundColor: 'rgba(59, 130, 246, 0.7)',
          borderColor: 'var(--primary)',
          borderWidth: 1.5,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
          grid: {
            color: '#1f2937'
          },
          ticks: {
            color: '#9ca3af',
            font: { family: 'Inter', size: 10 }
          }
        },
        x: {
          grid: { display: false },
          ticks: {
            color: '#9ca3af',
            font: { family: 'Inter', size: 10, weight: 600 }
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: '#f3f4f6',
            font: { family: 'Inter', size: 11, weight: 500 }
          }
        }
      }
    }
  });
}

// Single Prediction Form Submit
singleForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const modelChoice = modelSelector.value;
  const threshold = parseFloat(thresholdInput.value);
  
  // Compile payload
  const payload = {
    model: modelChoice,
    threshold: threshold,
    Time: parseFloat(document.getElementById('input-time').value || 0.0),
    Amount: parseFloat(document.getElementById('input-amount').value || 0.0)
  };
  
  // Read V1-V28 inputs
  for (let i = 1; i <= 28; i++) {
    payload[`V${i}`] = parseFloat(document.getElementById(`input-v${i}`).value || 0.0);
  }
  
  const submitBtn = document.getElementById('btn-predict-single');
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running inference...';
  
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || 'Server error occurred during prediction.');
    }
    
    const result = await res.json();
    displaySingleResult(result, threshold);
  } catch (err) {
    alert(`Prediction Error: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fa-solid fa-shield-heart"></i> Analyze Transaction';
  }
});

// Render Single Prediction Output
function displaySingleResult(result, threshold) {
  singleResultBox.classList.remove('hidden');
  
  const probPercent = result.fraud_probability * 100;
  resultProbVal.textContent = `${probPercent.toFixed(2)}%`;
  resultProgressFill.style.width = `${probPercent}%`;
  
  if (result.prediction === 1) {
    resultBadge.className = 'result-badge fraud';
    resultBadgeIcon.className = 'fa-solid fa-triangle-exclamation';
    resultBadgeText.textContent = 'FRAUD ALERT';
    resultProgressFill.style.backgroundColor = 'var(--danger)';
  } else {
    resultBadge.className = 'result-badge legit';
    resultBadgeIcon.className = 'fa-solid fa-circle-check';
    resultBadgeText.textContent = 'LEGITIMATE';
    resultProgressFill.style.backgroundColor = 'var(--success)';
  }
  
  const nameMap = {
    'random_forest': 'Random Forest (Tuned)',
    'logistic_regression': 'Logistic Regression'
  };
  resultModelUsed.textContent = nameMap[result.model_used] || result.model_used;
  resultThresholdUsed.textContent = threshold.toFixed(2);
  
  updateThresholdMarker(threshold);
}

// Drag & Drop File Upload Interactions
uploadZone.addEventListener('click', () => {
  fileInput.click();
});

// Dragover styles
['dragenter', 'dragover'].forEach(name => {
  uploadZone.addEventListener(name, (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  }, false);
});

['dragleave', 'drop'].forEach(name => {
  uploadZone.addEventListener(name, (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
  }, false);
});

uploadZone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 0) {
    validateAndSelectFile(files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    validateAndSelectFile(e.target.files[0]);
  }
});

function validateAndSelectFile(file) {
  if (!file.name.endsWith('.csv')) {
    alert('Invalid file format. Please upload a structured .csv file.');
    return;
  }
  
  selectedFile = file;
  fileNameLabel.textContent = file.name;
  fileSizeLabel.textContent = formatBytes(file.size);
  
  fileDetails.classList.remove('hidden');
  uploadZone.classList.add('hidden');
  analyzeBatchBtn.disabled = false;
}

function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

clearFileBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  fileDetails.classList.add('hidden');
  uploadZone.classList.remove('hidden');
  analyzeBatchBtn.disabled = true;
  document.getElementById('batch-results-area').classList.add('hidden');
});

// Run Batch Diagnostics Click
analyzeBatchBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  
  const modelChoice = modelSelector.value;
  const threshold = parseFloat(thresholdInput.value);
  
  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('model', modelChoice);
  formData.append('threshold', threshold);
  
  analyzeBatchBtn.disabled = true;
  analyzeBatchBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing batch pipeline...';
  
  try {
    const res = await fetch(`${API_BASE}/api/predict-batch`, {
      method: 'POST',
      body: formData
    });
    
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || 'Server error occurred during batch execution.');
    }
    
    const results = await res.json();
    displayBatchResults(results);
  } catch (err) {
    alert(`Batch Diagnostics Failed: ${err.message}`);
  } finally {
    analyzeBatchBtn.disabled = false;
    analyzeBatchBtn.innerHTML = '<i class="fa-solid fa-radar"></i> Run Batch Diagnostics';
  }
});

// Render stats and pagination for batch
function displayBatchResults(data) {
  batchResults = data;
  currentPage = 1;
  
  const total = data.length;
  const fraudCount = data.filter(row => row.Predicted_Class === 1).length;
  const safeCount = total - fraudCount;
  const rate = (fraudCount / total) * 100;
  
  document.getElementById('stat-total-processed').textContent = total.toLocaleString();
  document.getElementById('stat-flagged-fraud').textContent = fraudCount.toLocaleString();
  document.getElementById('stat-flagged-safe').textContent = safeCount.toLocaleString();
  document.getElementById('stat-fraud-rate').textContent = `${rate.toFixed(2)}%`;
  
  document.getElementById('batch-results-area').classList.remove('hidden');
  renderTablePage();
}

function renderTablePage() {
  const tbody = document.getElementById('results-table-body');
  tbody.innerHTML = '';
  
  const total = batchResults.length;
  const totalPages = Math.ceil(total / rowsPerPage) || 1;
  
  const start = (currentPage - 1) * rowsPerPage;
  const end = Math.min(start + rowsPerPage, total);
  
  document.getElementById('btn-prev-page').disabled = (currentPage === 1);
  document.getElementById('btn-next-page').disabled = (currentPage === totalPages);
  document.getElementById('pagination-info-label').textContent = `Page ${currentPage} of ${totalPages} (Txn ${start + 1} - ${end} of ${total})`;
  
  const pageData = batchResults.slice(start, end);
  
  pageData.forEach(row => {
    // Generate PCA driver details
    const pcaList = [];
    for (let i = 1; i <= 28; i++) {
      const val = row[`V${i}`];
      if (val !== undefined) {
        pcaList.push({ name: `V${i}`, val: parseFloat(val) });
      }
    }
    
    // Sort by largest absolute value (representing highest impact in PCA dimensions)
    pcaList.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
    const top3 = pcaList.slice(0, 3).map(p => {
      const prefix = p.val >= 0 ? '+' : '';
      return `<span class="pca-driver-highlight" title="${p.name}">${p.name}</span>(${prefix}${p.val.toFixed(2)})`;
    }).join(', ');
    
    const isFraud = row.Predicted_Class === 1;
    const badgeClass = isFraud ? 'table-badge fraud' : 'table-badge legit';
    const badgeText = isFraud ? 'Fraud Alert' : 'Legitimate';
    const probPercent = (row.Fraud_Probability * 100).toFixed(2);
    
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${Math.round(row.Time)}s</td>
      <td>&euro;${row.Amount.toFixed(2)}</td>
      <td style="font-weight: 600; color: ${isFraud ? 'var(--danger)' : 'var(--text-main)'};">${probPercent}%</td>
      <td><span class="${badgeClass}">${badgeText}</span></td>
      <td><div class="pca-drivers">${top3 || 'None'}</div></td>
    `;
    tbody.appendChild(tr);
  });
}

// Pagination Navigation
document.getElementById('btn-prev-page').addEventListener('click', () => {
  if (currentPage > 1) {
    currentPage--;
    renderTablePage();
  }
});

document.getElementById('btn-next-page').addEventListener('click', () => {
  const totalPages = Math.ceil(batchResults.length / rowsPerPage);
  if (currentPage < totalPages) {
    currentPage++;
    renderTablePage();
  }
});

// CSV Downloads handler
document.getElementById('btn-download-results').addEventListener('click', () => {
  if (!batchResults || batchResults.length === 0) return;
  
  const headers = Object.keys(batchResults[0]);
  const csvRows = [];
  
  csvRows.push(headers.join(','));
  
  batchResults.forEach(row => {
    const values = headers.map(hdr => {
      const val = row[hdr] !== undefined ? row[hdr] : '';
      const escaped = ('' + val).replace(/"/g, '\\"');
      return `"${escaped}"`;
    });
    csvRows.push(values.join(','));
  });
  
  const csvString = csvRows.join('\n');
  const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = `fraud_diagnostics_${new Date().toISOString().slice(0,10)}.csv`;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});
