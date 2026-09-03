/**
 * DeepSea eDNA AI Explorer – frontend logic
 * Handles file upload, analysis request, and rendering of results (metrics, plot, taxonomy).
 */

(function () {
  "use strict";

  const runBtn = document.getElementById("run-btn");
  const fileInput = document.getElementById("fasta");
  const fileZone = document.getElementById("file-zone");
  const filenameEl = document.getElementById("filename");
  const statusEl = document.getElementById("status");
  const metricsEl = document.getElementById("metrics");
  const downloadJsonBtn = document.getElementById("download-json");
  const downloadCsvBtn = document.getElementById("download-csv");
  const taxonomyCard = document.getElementById("taxonomy-card");
  const taxonomyTableBody = document.querySelector("#taxonomy-table tbody");
  const whatSection = document.getElementById("what-section");
  const whatHeader = document.getElementById("what-header");

  let lastResult = null;

  // --- File zone: show filename and drag-and-drop ---
  if (fileInput && fileZone && filenameEl) {
    fileInput.addEventListener("change", function () {
      const file = fileInput.files[0];
      fileZone.classList.toggle("has-file", !!file);
      filenameEl.textContent = file ? file.name : "";
    });

    fileZone.addEventListener("dragover", function (e) {
      e.preventDefault();
      fileZone.classList.add("dragover");
    });
    fileZone.addEventListener("dragleave", function () {
      fileZone.classList.remove("dragover");
    });
    fileZone.addEventListener("drop", function (e) {
      e.preventDefault();
      fileZone.classList.remove("dragover");
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file && /\.(fa|fasta|fna)$/i.test(file.name)) {
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        fileZone.classList.add("has-file");
        filenameEl.textContent = file.name;
      }
    });
  }

  // --- "What we're doing" toggle ---
  if (whatSection && whatHeader) {
    whatHeader.addEventListener("click", function () {
      whatSection.classList.toggle("open");
    });
  }

  function setStatus(msg, isError, isSuccess) {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.className = "status" + (isError ? " error" : isSuccess ? " success" : "");
  }

  function renderMetrics(data) {
    if (!metricsEl) return;
    metricsEl.innerHTML = "";
    const bio = data.biodiversity || {};
    const items = [
      ["Sequences", data.n_sequences],
      ["Clusters (richness)", data.num_clusters],
      ["Noise reads", data.noise_count],
      ["Shannon H′", (bio.shannon != null ? bio.shannon : 0).toFixed(3)],
      ["Simpson 1−D", (bio.simpson_diversity != null ? bio.simpson_diversity : 0).toFixed(3)],
    ];
    items.forEach(function (pair) {
      const label = pair[0];
      const value = pair[1];
      const div = document.createElement("div");
      div.className = "metric-pill";
      div.innerHTML =
        "<div class=\"metric-label\">" + label + "</div><div class=\"metric-value\">" + value + "</div>";
      metricsEl.appendChild(div);
    });
    metricsEl.style.display = "grid";
  }

  function renderTaxonomy(data) {
    const tax = data.cluster_taxonomy || {};
    if (!taxonomyTableBody || !taxonomyCard) return;
    taxonomyTableBody.innerHTML = "";
    const ids = Object.keys(tax).sort(function (a, b) { return Number(a) - Number(b); });
    if (ids.length === 0) {
      taxonomyCard.style.display = "none";
      return;
    }
    ids.forEach(function (cid) {
      const row = tax[cid];
      const tr = document.createElement("tr");
      const ident = row.best_identity != null ? Number(row.best_identity).toFixed(1) : "—";
      const nov = row.novelty_score != null ? Number(row.novelty_score).toFixed(2) : "—";
      const desc = row.best_description != null ? String(row.best_description) : "—";
      tr.innerHTML =
        "<td>" + cid + "</td>" +
        "<td>" + (row.size != null ? row.size : "—") + "</td>" +
        "<td>" + ident + "%</td>" +
        "<td>" + nov + "</td>" +
        "<td>" + desc + "</td>";
      taxonomyTableBody.appendChild(tr);
    });
    taxonomyCard.style.display = "block";
  }

  function renderPlot(data) {
    const embedding = data.embedding;
    const labels = data.labels;
    if (!embedding || embedding.length === 0 || typeof Plotly === "undefined") {
      if (typeof Plotly !== "undefined") Plotly.purge("plot");
      return;
    }
    const x = embedding.map(function (p) { return p[0]; });
    const y = embedding.map(function (p) { return p[1]; });
    const trace = {
      x: x,
      y: y,
      mode: "markers",
      type: "scattergl",
      marker: {
        size: 8,
        color: labels,
        colorscale: [[0, "#0c4a6e"], [0.35, "#0ea5e9"], [0.7, "#38bdf8"], [1, "#67e8f9"]],
        showscale: true,
        colorbar: { title: "Cluster", tickfont: { color: "#94a3b8" }, outlinewidth: 0 },
        line: { width: 0 },
      },
      text: labels.map(function (l, i) { return "Seq " + i + " · Cluster " + l; }),
      hoverinfo: "text",
    };
    const layout = {
      margin: { l: 12, r: 12, t: 12, b: 12 },
      paper_bgcolor: "rgba(3,7,18,0.6)",
      plot_bgcolor: "rgba(3,7,18,0.6)",
      font: { family: "Outfit, system-ui, sans-serif", color: "#94a3b8", size: 11 },
      xaxis: {
        zeroline: false,
        showgrid: true,
        gridcolor: "rgba(56,189,248,0.12)",
        color: "#94a3b8",
      },
      yaxis: {
        zeroline: false,
        showgrid: true,
        gridcolor: "rgba(56,189,248,0.12)",
        color: "#94a3b8",
      },
      colorway: ["#38bdf8", "#22d3ee", "#2dd4bf", "#34d399", "#a78bfa", "#f472b6"],
    };
    Plotly.newPlot("plot", [trace], layout, { responsive: true, displaylogo: false });
  }

  function downloadBlob(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function toCsv(data) {
    const rows = [];
    rows.push(["cluster_id", "size", "best_identity", "novelty_score", "best_description"].join(","));
    const tax = data.cluster_taxonomy || {};
    Object.keys(tax).sort(function (a, b) { return Number(a) - Number(b); }).forEach(function (cid) {
      const row = tax[cid];
      rows.push([
        cid,
        row.size,
        row.best_identity,
        row.novelty_score,
        '"' + (row.best_description || "").replace(/"/g, '""') + '"',
      ].join(","));
    });
    return rows.join("\n");
  }

  if (runBtn && fileInput) {
    runBtn.addEventListener("click", function () {
      const file = fileInput.files[0];
      if (!file) {
        setStatus("Please choose a FASTA file first.", true);
        return;
      }
      runBtn.disabled = true;
      setStatus("Uploading file and running clustering…");

      var form = new FormData();
      form.append("file", file);

      fetch("/api/analyze", { method: "POST", body: form })
        .then(function (res) {
          if (!res.ok) {
            return res.json().catch(function () { return {}; }).then(function (err) {
              throw new Error(err.detail || "Server error while analyzing file.");
            });
          }
          return res.json();
        })
        .then(function (data) {
          lastResult = data;
          setStatus("Analysis complete.", false, true);
          renderMetrics(data);
          renderPlot(data);
          renderTaxonomy(data);
        })
        .catch(function (err) {
          console.error(err);
          setStatus(err.message || "Unexpected error occurred.", true);
        })
        .finally(function () {
          runBtn.disabled = false;
        });
    });
  }

  if (downloadJsonBtn) {
    downloadJsonBtn.addEventListener("click", function () {
      if (!lastResult) {
        setStatus("Run an analysis before downloading results.", true);
        return;
      }
      downloadBlob(JSON.stringify(lastResult, null, 2), "edna_summary.json", "application/json");
    });
  }

  if (downloadCsvBtn) {
    downloadCsvBtn.addEventListener("click", function () {
      if (!lastResult) {
        setStatus("Run an analysis before downloading results.", true);
        return;
      }
      downloadBlob(toCsv(lastResult), "edna_clusters.csv", "text/csv");
    });
  }
})();
