"use strict";

const metricDefinitions = [
  { key: "accuracy", label: "Accuracy" },
  { key: "balanced_accuracy", label: "Balanced accuracy" },
  { key: "theft_precision", label: "Theft precision" },
  { key: "theft_recall", label: "Theft recall" },
  { key: "theft_f1", label: "Theft F1-score" },
  { key: "roc_auc", label: "ROC-AUC" },
  { key: "average_precision", label: "Average precision" }
];

const featureDescriptions = {
  mean_log_consumption: "Average log daily use",
  std_log_consumption: "Variability of log use",
  median_log_consumption: "Median log daily use",
  p10_log_consumption: "10th percentile of log use",
  p90_log_consumption: "90th percentile of log use",
  log_consumption_cv: "Relative spread of log use",
  observed_zero_day_share: "Share of observed zero-use days",
  missing_day_share: "Share of missing daily readings",
  mean_abs_daily_change_log: "Average absolute change between days",
  sharp_drop_share: "Share of sharp daily drops",
  sharp_rise_share: "Share of sharp daily rises",
  trend_slope_log: "Linear trend across days"
};

let readingColumns = [];
let sampleReadings = [];
let batchResults = [];
let charts = [];
let apiModelReady = false;

function byId(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  const element = byId(id);
  if (element) element.textContent = text == null ? "" : String(text);
}

function formatPercent(value, digits) {
  return (Number(value) * 100).toFixed(digits == null ? 2 : digits) + "%";
}

function formatNumber(value, digits) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits == null ? 0 : digits });
}

async function getJson(url, options) {
  const response = await fetch(url, options || {});
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error("The server returned an unreadable response.");
  }
  if (!response.ok || payload.success === false) {
    throw new Error(payload.error || "The request could not be completed.");
  }
  return payload;
}

function updateStatus(ready, message, metricsReady) {
  const status = byId("modelStatus");
  status.className = "model-status " + (!ready ? "status-offline" : metricsReady ? "" : "status-warning");
  status.replaceChildren();
  const dot = document.createElement("span");
  dot.className = "status-dot";
  const label = document.createElement("span");
  label.textContent = !ready ? "Model unavailable" : metricsReady ? "Model + metrics ready" : "Model ready · metrics missing";
  status.append(dot, label);

  const banner = byId("modelBanner");
  banner.classList.toggle("banner-error", !ready);
  setText("bannerTitle", ready ? "Saved model loaded" : "Model needs attention");
  setText("bannerMessage", message || (ready ? "The fitted Gaussian Naive Bayes model and notebook preprocessing are ready." : "Predictions are disabled until valid model artifacts are available."));
  setText("bannerTag", ready ? "MODEL READY" : "NOT AVAILABLE");
  apiModelReady = ready;
  byId("predictButton").disabled = !ready || readingColumns.length === 0;
  byId("csvButton").disabled = !ready;
}

function showNotice(message) {
  const notice = byId("apiNotice");
  notice.textContent = message;
  notice.classList.remove("hidden");
}

function renderMetrics(payload) {
  const available = Boolean(payload.available && payload.metrics);
  if (!available) {
    setText("evaluationSplit", payload.message || "Evaluation metrics have not been exported");
    metricDefinitions.forEach(function (item) {
      setText("metric-" + item.key, "—");
      setText("detail-" + item.key, "Metrics unavailable");
    });
    byId("metricChart").classList.add("hidden");
    byId("metricChartEmpty").classList.remove("hidden");
    return;
  }
  const metrics = payload.metrics;
  const metadata = payload.metadata || {};
  const metricCount = metricDefinitions.filter(function (item) { return Number.isFinite(Number(metrics[item.key])); }).length;
  setText("evaluationSplit", (metadata.test_records ? formatNumber(metadata.test_records) + " test records" : "Held-out test split") + " · exact scores shown below");

  metricDefinitions.forEach(function (item) {
    const value = Number(metrics[item.key]);
    if (!Number.isFinite(value)) {
      setText("metric-" + item.key, "—");
      setText("detail-" + item.key, "Not supplied by the evaluation");
      return;
    }
    setText("metric-" + item.key, formatPercent(value, 2));
    setText("detail-" + item.key, "Exact score: " + value.toFixed(4));
    const bar = byId("bar-" + item.key);
    if (bar) bar.style.width = Math.max(0, Math.min(100, value * 100)) + "%";
  });
  if (metricCount && window.Chart) renderMetricChart(metrics);
  else {
    byId("metricChart").classList.add("hidden");
    byId("metricChartEmpty").classList.remove("hidden");
    if (!window.Chart && metricCount) setText("metricChartEmpty", "Chart.js could not load in this browser session, so the metric chart is unavailable.");
  }
  renderMetadata(metadata, payload.metadata || {});
}

function renderMetadata(metadata) {
  if (!metadata || !Object.keys(metadata).length) return;
  readingColumns = Array.isArray(metadata.daily_reading_columns) ? metadata.daily_reading_columns : [];
  sampleReadings = Array.isArray(metadata.sample_readings) ? metadata.sample_readings : [];
  const features = Array.isArray(metadata.feature_names) ? metadata.feature_names : [];
  setText("readingCount", readingColumns.length ? readingColumns.length + " daily readings · enter values in the order shown" : "Reading schema unavailable");
  if (readingColumns.length) renderReadingFields();
  if (metadata.dataset_name) setText("modelDescription", "The model was trained on " + metadata.dataset_name + ". It summarizes each customer's daily consumption history, scales the resulting features with the saved training scaler, and estimates the likelihood of the theft-labelled class.");

  const properties = [
    ["Dataset", metadata.dataset_name || "Not recorded"],
    ["Evaluation split", metadata.evaluation_split || "Not recorded"],
    ["Training records", metadata.train_records == null ? "Not recorded" : formatNumber(metadata.train_records)],
    ["Test records", metadata.test_records == null ? "Not recorded" : formatNumber(metadata.test_records)],
    ["Decision threshold", metadata.threshold == null ? "Not recorded" : Number(metadata.threshold).toFixed(6)],
    ["Selected smoothing", metadata.var_smoothing == null ? "Not recorded" : Number(metadata.var_smoothing).toExponential(0)]
  ];
  const propertyWrap = byId("modelProperties");
  propertyWrap.replaceChildren();
  properties.forEach(function (item) {
    const block = document.createElement("div");
    block.className = "property";
    const label = document.createElement("span");
    label.textContent = item[0];
    const value = document.createElement("strong");
    value.textContent = item[1];
    block.append(label, value);
    propertyWrap.append(block);
  });

  const featureWrap = byId("featureList");
  featureWrap.replaceChildren();
  if (!features.length) {
    const empty = document.createElement("span");
    empty.className = "muted";
    empty.textContent = "Feature names were not exported.";
    featureWrap.append(empty);
    return;
  }
  features.forEach(function (feature) {
    const chip = document.createElement("span");
    chip.className = "feature-chip";
    chip.textContent = feature;
    chip.title = featureDescriptions[feature] || feature;
    featureWrap.append(chip);
  });
}

function renderReadingFields() {
  const container = byId("readingFields");
  container.replaceChildren();
  readingColumns.forEach(function (column, index) {
    const wrapper = document.createElement("div");
    wrapper.className = "reading-field";
    const label = document.createElement("label");
    label.htmlFor = "reading-" + index;
    label.textContent = column;
    const input = document.createElement("input");
    input.id = "reading-" + index;
    input.name = column;
    input.type = "number";
    input.min = "0";
    input.step = "any";
    input.required = true;
    input.inputMode = "decimal";
    input.placeholder = "kWh";
    input.autocomplete = "off";
    wrapper.append(label, input);
    container.append(wrapper);
  });
  byId("predictButton").disabled = !apiModelReady;
}

const chartColors = {
  grid: "rgba(145, 170, 194, .1)",
  tick: "#8197ad",
  blue: "#58acff",
  teal: "#4bd7bd",
  amber: "#ffc16e",
  violet: "#a195ff",
  rose: "#ff8495"
};

function registerChart(chart) {
  charts.push(chart);
  return chart;
}

function baseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 450 },
    plugins: {
      legend: { labels: { color: "#9aadc0", usePointStyle: true, pointStyle: "circle", boxWidth: 7, boxHeight: 7, padding: 15, font: { family: "DM Sans", size: 10 } } },
      tooltip: { backgroundColor: "#102238", titleColor: "#edf5fc", bodyColor: "#bed0df", borderColor: "#36516a", borderWidth: 1, padding: 9, displayColors: true }
    },
    scales: {
      x: { grid: { color: chartColors.grid }, ticks: { color: chartColors.tick, font: { family: "DM Sans", size: 9 } }, border: { display: false } },
      y: { beginAtZero: true, max: 1, grid: { color: chartColors.grid }, ticks: { color: chartColors.tick, font: { family: "DM Sans", size: 9 }, callback: function (value) { return Math.round(Number(value) * 100) + "%"; } }, border: { display: false } }
    }
  };
}

function renderMetricChart(metrics) {
  const canvas = byId("metricChart");
  if (!window.Chart || !canvas) return;
  const keys = ["accuracy", "balanced_accuracy", "theft_precision", "theft_recall", "theft_f1"];
  const labels = ["Accuracy", "Balanced accuracy", "Precision", "Recall", "F1-score"];
  const values = keys.map(function (key) { return Number(metrics[key]); });
  const options = baseChartOptions();
  options.plugins.legend.display = false;
  options.scales.y.suggestedMax = 1;
  registerChart(new Chart(canvas, {
    type: "bar",
    data: { labels: labels, datasets: [{ data: values, backgroundColor: [chartColors.blue, chartColors.violet, chartColors.amber, chartColors.teal, chartColors.rose].map(function (color) { return color + "bb"; }), borderRadius: 5, maxBarThickness: 31 }] },
    options: options
  }));
}

function renderEvaluation(payload) {
  if (!payload || !payload.available) {
    ["confusionEmpty", "metricChartEmpty", "rocEmpty", "prEmpty", "classEmpty"].forEach(function (id) { byId(id).classList.remove("hidden"); });
    ["confusionChart", "metricChart", "rocChart", "prChart", "classChart"].forEach(function (id) { byId(id).classList.add("hidden"); });
    return;
  }
  if (!window.Chart) {
    ["confusionChart", "metricChart", "rocChart", "prChart", "classChart"].forEach(function (id) { byId(id).classList.add("hidden"); });
    ["confusionEmpty", "metricChartEmpty", "rocEmpty", "prEmpty", "classEmpty"].forEach(function (id) {
      setText(id, "Chart.js could not load in this browser session. Check your internet connection and refresh to view the saved test-set charts.");
      byId(id).classList.remove("hidden");
    });
    return;
  }
  const matrix = payload.confusion_matrix;
  if (Array.isArray(matrix) && matrix.length === 2) renderConfusion(matrix, payload.confusion_counts || {});
  else {
    byId("confusionChart").classList.add("hidden");
    byId("confusionEmpty").classList.remove("hidden");
  }
  const roc = payload.roc_curve;
  if (roc && Array.isArray(roc.false_positive_rate) && Array.isArray(roc.true_positive_rate)) {
    renderCurve("rocChart", roc.false_positive_rate, roc.true_positive_rate, "False positive rate", "True positive rate", "ROC", chartColors.blue, roc.auc);
  } else {
    byId("rocChart").classList.add("hidden");
    byId("rocEmpty").classList.remove("hidden");
  }
  const pr = payload.precision_recall_curve;
  if (pr && Array.isArray(pr.recall) && Array.isArray(pr.precision)) {
    renderCurve("prChart", pr.recall, pr.precision, "Theft recall", "Precision", "Precision–recall", chartColors.teal, pr.average_precision, payload.class_counts ? payload.class_counts.theft_labelled / payload.class_counts.total : null);
  } else {
    byId("prChart").classList.add("hidden");
    byId("prEmpty").classList.remove("hidden");
  }
  const classCounts = payload.class_counts;
  if (classCounts) renderClassDistribution(classCounts);
  else {
    byId("classChart").classList.add("hidden");
    byId("classEmpty").classList.remove("hidden");
  }
}

function renderConfusion(matrix, counts) {
  const canvas = byId("confusionChart");
  const options = baseChartOptions();
  options.scales.x.stacked = true;
  options.scales.y.stacked = true;
  options.scales.y.max = undefined;
  options.scales.y.ticks.callback = function (value) { return formatNumber(value); };
  options.plugins.tooltip.callbacks = { label: function (context) { return " " + context.dataset.label + ": " + formatNumber(context.raw); } };
  registerChart(new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["Actual normal", "Actual theft-labelled"],
      datasets: [
        { label: "Predicted normal", data: [matrix[0][0], matrix[1][0]], backgroundColor: "#58acffbd", borderRadius: 4, maxBarThickness: 47 },
        { label: "Predicted theft-labelled", data: [matrix[0][1], matrix[1][1]], backgroundColor: "#ffbd68c7", borderRadius: 4, maxBarThickness: 47 }
      ]
    },
    options: options
  }));
  const pairs = [["True negative", counts.true_negative], ["False positive", counts.false_positive], ["False negative", counts.false_negative], ["True positive", counts.true_positive]];
  const summary = byId("matrixSummary");
  summary.replaceChildren();
  pairs.forEach(function (pair) {
    const item = document.createElement("div");
    item.className = "matrix-cell";
    const label = document.createElement("span");
    label.textContent = pair[0];
    const value = document.createElement("strong");
    value.textContent = pair[1] == null ? "—" : formatNumber(pair[1]);
    item.append(label, value);
    summary.append(item);
  });
  summary.classList.remove("hidden");
}

function renderCurve(canvasId, xs, ys, xLabel, yLabel, legendLabel, color, score, baseline) {
  const canvas = byId(canvasId);
  const data = xs.map(function (x, index) { return { x: Number(x), y: Number(ys[index]) }; });
  const datasets = [{ label: legendLabel + (score == null ? "" : " · " + Number(score).toFixed(4)), data: data, borderColor: color, backgroundColor: color + "22", pointRadius: 0, borderWidth: 2, fill: false, tension: 0, parsing: false }];
  if (canvasId === "rocChart") datasets.push({ label: "Chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "#657b91", borderDash: [4, 4], pointRadius: 0, borderWidth: 1, parsing: false });
  if (baseline != null) datasets.push({ label: "Test-set theft share", data: [{ x: 0, y: baseline }, { x: 1, y: baseline }], borderColor: "#657b91", borderDash: [4, 4], pointRadius: 0, borderWidth: 1, parsing: false });
  const options = baseChartOptions();
  options.scales.x.type = "linear";
  options.scales.x.min = 0;
  options.scales.x.max = 1;
  options.scales.x.title = { display: true, text: xLabel, color: "#7d94aa", font: { family: "DM Sans", size: 9 } };
  options.scales.y.min = 0;
  options.scales.y.max = 1;
  options.scales.y.title = { display: true, text: yLabel, color: "#7d94aa", font: { family: "DM Sans", size: 9 } };
  options.plugins.tooltip.callbacks = { label: function (context) { return " " + context.dataset.label + ": " + Number(context.parsed.y).toFixed(3); } };
  registerChart(new Chart(canvas, { type: "line", data: { datasets: datasets }, options: options }));
}

function renderClassDistribution(counts) {
  registerChart(new Chart(byId("classChart"), {
    type: "doughnut",
    data: { labels: ["Normal", "Theft-labelled"], datasets: [{ data: [counts.normal, counts.theft_labelled], backgroundColor: [chartColors.blue, chartColors.amber], borderColor: "#0e1c2b", borderWidth: 4, hoverOffset: 4 }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: "70%", plugins: { legend: { display: false }, tooltip: { backgroundColor: "#102238", titleColor: "#edf5fc", bodyColor: "#bed0df", borderColor: "#36516a", borderWidth: 1, callbacks: { label: function (context) { return " " + context.label + ": " + formatNumber(context.raw); } } } } }
  }));
  const legend = byId("classLegend");
  legend.replaceChildren();
  [["Normal", counts.normal, chartColors.blue], ["Theft-labelled", counts.theft_labelled, chartColors.amber]].forEach(function (item) {
    const row = document.createElement("div");
    row.className = "legend-item";
    const dot = document.createElement("span");
    dot.className = "legend-dot";
    dot.style.backgroundColor = item[2];
    const label = document.createElement("span");
    label.innerHTML = "";
    label.textContent = item[0];
    const value = document.createElement("strong");
    value.textContent = formatNumber(item[1]);
    row.append(dot, label, value);
    legend.append(row);
  });
  const total = Number(counts.total || Number(counts.normal) + Number(counts.theft_labelled));
  const note = document.createElement("small");
  note.textContent = "Total: " + formatNumber(total) + " customers";
  legend.append(note);
}

function showManualError(message) {
  setText("manualError", message);
  byId("manualError").classList.remove("hidden");
  byId("predictionResult").classList.add("hidden");
}

function clearManualError() {
  byId("manualError").classList.add("hidden");
}

async function submitManual(event) {
  event.preventDefault();
  clearManualError();
  const button = byId("predictButton");
  const values = Array.from(byId("readingFields").querySelectorAll("input")).map(function (input) { return Number(input.value); });
  if (values.length !== readingColumns.length || values.some(function (value) { return !Number.isFinite(value) || value < 0; })) {
    showManualError("Enter a finite, non-negative value for every daily reading.");
    return;
  }
  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Scoring readings…";
  try {
    const result = await getJson("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ readings: values }) });
    const card = byId("predictionResult");
    card.classList.toggle("is-theft", result.predicted_class === 1);
    card.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = result.prediction === "Theft-labelled" ? "Flagged for review" : "Predicted normal";
    const score = document.createElement("div");
    score.className = "score-line";
    const scoreLabel = document.createElement("span");
    scoreLabel.textContent = "Theft probability estimate";
    const scoreValue = document.createElement("b");
    scoreValue.textContent = formatPercent(result.theft_probability, 2) + " · score " + Number(result.theft_probability).toFixed(4);
    score.append(scoreLabel, scoreValue);
    const threshold = document.createElement("div");
    threshold.className = "score-line";
    const thresholdLabel = document.createElement("span");
    thresholdLabel.textContent = "Saved decision threshold";
    const thresholdValue = document.createElement("b");
    thresholdValue.textContent = Number(result.threshold).toFixed(6);
    threshold.append(thresholdLabel, thresholdValue);
    const explanation = document.createElement("p");
    explanation.textContent = "This is a model prediction to support review, not proof of electricity theft.";
    card.append(heading, score, threshold, explanation);
    card.classList.remove("hidden");
  } catch (error) {
    showManualError(error.message);
  } finally {
    button.disabled = !apiModelReady;
    button.querySelector("span:first-child").textContent = "Run prediction";
  }
}

function escapeCsv(value) {
  const stringValue = String(value == null ? "" : value);
  return '"' + stringValue.replace(/"/g, '""') + '"';
}

async function submitCsv(event) {
  event.preventDefault();
  byId("csvError").classList.add("hidden");
  const file = byId("csvFile").files[0];
  if (!file) {
    setText("csvError", "Choose a CSV file first.");
    byId("csvError").classList.remove("hidden");
    return;
  }
  const button = byId("csvButton");
  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Processing CSV…";
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await getJson("/api/predict-csv", { method: "POST", body: form });
    batchResults = response.results || [];
    setText("csvSummary", "");
    const summary = byId("csvSummary");
    [["Processed", response.processed, ""], ["Normal", response.normal_count, ""], ["Theft-labelled", response.theft_labelled_count, "theft"]].forEach(function (item) {
      const chip = document.createElement("span");
      chip.className = "summary-chip " + item[2];
      const amount = document.createElement("b");
      amount.textContent = formatNumber(item[1]);
      chip.append(amount, document.createTextNode(" " + item[0]));
      summary.append(chip);
    });
    renderBatchRows("");
    byId("csvResults").classList.remove("hidden");
  } catch (error) {
    setText("csvError", error.message);
    byId("csvError").classList.remove("hidden");
  } finally {
    button.disabled = !apiModelReady;
    button.querySelector("span:first-child").textContent = "Predict all rows";
  }
}

function renderBatchRows(query) {
  const normalized = String(query || "").trim().toLowerCase();
  const matches = batchResults.filter(function (result) {
    return [result.record, result.prediction, result.theft_probability, result.threshold].join(" ").toLowerCase().includes(normalized);
  });
  const visible = matches.slice(0, 200);
  const tbody = byId("resultRows");
  tbody.replaceChildren();
  visible.forEach(function (result) {
    const row = document.createElement("tr");
    const record = document.createElement("td");
    record.textContent = formatNumber(result.record);
    const prediction = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "prediction-badge" + (result.predicted_class === 1 ? " theft" : "");
    badge.textContent = result.prediction;
    prediction.append(badge);
    const score = document.createElement("td");
    score.textContent = formatPercent(result.theft_probability, 2);
    const threshold = document.createElement("td");
    threshold.textContent = Number(result.threshold).toFixed(4);
    row.append(record, prediction, score, threshold);
    tbody.append(row);
  });
  setText("tableFoot", "Showing " + formatNumber(visible.length) + " of " + formatNumber(matches.length) + " matching rows" + (matches.length > visible.length ? " · download the CSV to keep all results" : ""));
}

function downloadResults() {
  const header = ["record", "prediction", "predicted_class", "theft_probability", "threshold"];
  const lines = [header.map(escapeCsv).join(",")];
  batchResults.forEach(function (row) {
    lines.push([row.record, row.prediction, row.predicted_class, row.theft_probability, row.threshold].map(escapeCsv).join(","));
  });
  const blob = new Blob([lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "electricity_theft_predictions.csv";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function setNavigation() {
  const links = Array.from(document.querySelectorAll(".nav-link"));
  links.forEach(function (link) {
    link.addEventListener("click", function () {
      links.forEach(function (item) { item.classList.remove("active"); });
      link.classList.add("active");
    });
  });
  const sections = links.map(function (link) { return document.querySelector(link.getAttribute("href")); }).filter(Boolean);
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        const link = document.querySelector('.nav-link[href="#' + entry.target.id + '"]');
        if (!link) return;
        links.forEach(function (item) { item.classList.toggle("active", item === link); });
      });
    }, { rootMargin: "-25% 0px -65% 0px" });
    sections.forEach(function (section) { observer.observe(section); });
  }
}

async function initializeDashboard() {
  try {
    const responses = await Promise.all([getJson("/api/health"), getJson("/api/metrics"), getJson("/api/evaluation")]);
    const health = responses[0];
    const metrics = responses[1];
    const evaluation = responses[2];
    updateStatus(Boolean(health.model_loaded), health.message, Boolean(health.metrics_available));
    renderMetrics(metrics);
    renderEvaluation(evaluation);
    if (metrics.metadata) renderMetadata(metrics.metadata);
    if (!health.model_loaded) showNotice(health.message || "The saved model could not be loaded.");
  } catch (error) {
    updateStatus(false, "Could not connect to the Flask API. Start the local server and refresh this page.");
    showNotice("Could not connect to the backend. Open the app from http://127.0.0.1:5000 and confirm that /api/health is available.");
    ["confusionEmpty", "metricChartEmpty", "rocEmpty", "prEmpty", "classEmpty"].forEach(function (id) { byId(id).classList.remove("hidden"); });
    ["confusionChart", "metricChart", "rocChart", "prChart", "classChart"].forEach(function (id) { byId(id).classList.add("hidden"); });
    console.error("Dashboard initialization failed", error);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  byId("manualForm").addEventListener("submit", submitManual);
  byId("csvForm").addEventListener("submit", submitCsv);
  byId("sampleButton").addEventListener("click", function () {
    if (!sampleReadings.length || sampleReadings.length !== readingColumns.length) {
      showManualError("Training median sample values are not available in the saved artifacts.");
      return;
    }
    sampleReadings.forEach(function (value, index) { byId("reading-" + index).value = Number(value).toFixed(4); });
    clearManualError();
    byId("predictionResult").classList.add("hidden");
  });
  byId("csvFile").addEventListener("change", function () {
    const file = byId("csvFile").files[0];
    setText("fileName", file ? file.name : "Choose a CSV file");
    byId("csvButton").disabled = !apiModelReady || !file;
  });
  byId("resultSearch").addEventListener("input", function (event) { renderBatchRows(event.target.value); });
  byId("downloadButton").addEventListener("click", downloadResults);
  setNavigation();
  initializeDashboard();
});
