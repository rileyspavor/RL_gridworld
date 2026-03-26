const state = {
  overview: null,
  runs: [],
  combos: [],
  configs: [],
  results: null,
  queue: {},
  selectedRunId: null,
  preview: null,
  bestRunDetail: null,
  currentSort: {
    by: "coverage",
    order: "desc",
  },
};

const filterIds = [
  "filter-family",
  "filter-algorithm",
  "filter-observation",
  "filter-reward",
  "filter-map-suite",
  "filter-status",
  "filter-q",
];

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  }
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return String(value);
  }
  return Number.isInteger(parsed) ? String(parsed) : parsed.toFixed(digits);
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return "—";
  }
  return `${(parsed * 100).toFixed(digits)}%`;
}

function formatDeltaPercent(value, digits = 1) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return "—";
  }
  const sign = parsed > 0 ? "+" : "";
  return `${sign}${(parsed * 100).toFixed(digits)} pts`;
}

function formatTimesteps(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return String(value);
  }
  return parsed.toLocaleString();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return response.json();
}

function setSelectOptions(selectId, values, includeAny = true, anyLabel = "(any)") {
  const select = document.getElementById(selectId);
  const previous = select.value;
  const options = [];
  if (includeAny) {
    options.push(["", anyLabel]);
  }
  for (const value of values || []) {
    options.push([String(value), String(value)]);
  }
  select.innerHTML = options.map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
  if (options.some(([value]) => value === previous)) {
    select.value = previous;
  }
}

function getFilters() {
  return {
    family: document.getElementById("filter-family").value,
    algorithm: document.getElementById("filter-algorithm").value,
    observation: document.getElementById("filter-observation").value,
    reward: document.getElementById("filter-reward").value,
    map_suite: document.getElementById("filter-map-suite").value,
    status: document.getElementById("filter-status").value,
    q: document.getElementById("filter-q").value,
  };
}

function toQueryString(payload) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(payload)) {
    if (value === null || value === undefined) {
      continue;
    }
    const text = String(value).trim();
    if (text === "") {
      continue;
    }
    params.set(key, text);
  }
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

function qualityClass(value) {
  if (value === null || value === undefined || value === "") {
    return "neutral";
  }
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return "neutral";
  }
  if (parsed >= 0.8) return "excellent";
  if (parsed >= 0.55) return "good";
  if (parsed >= 0.3) return "watch";
  return "bad";
}

function coverageChip(value) {
  const klass = qualityClass(value);
  return `<span class="value-chip ${klass}">${escapeHtml(formatPercent(value))}</span>`;
}

function familyLabel(family) {
  return family ? String(family).replaceAll("_", " ") : "—";
}

function focusLabel(run) {
  return [run.observation, run.reward].filter(Boolean).join(" / ") || "—";
}

function mapMetric(run, mapName, key = "mean_coverage") {
  const rows = run?.primary_map_eval?.rows || [];
  const row = rows.find((item) => item.map_name === mapName);
  if (row) {
    return row[key];
  }
  const fallbackKeys = {
    just_go: "just_go_coverage",
    safe: "safe_coverage",
    maze: "maze_coverage",
    chokepoint: "chokepoint_coverage",
    sneaky_enemies: "sneaky_coverage",
  };
  const fallbackKey = fallbackKeys[mapName];
  return fallbackKey ? run?.[fallbackKey] ?? null : null;
}

function normalizeSortKey(raw) {
  const key = String(raw || "").trim().toLowerCase();
  const aliases = {
    "": "coverage",
    overall: "coverage",
    overall_coverage: "coverage",
    sneaky_enemies: "sneaky",
    map: "map_mean",
    mean_map: "map_mean",
    steps: "timesteps",
  };
  return aliases[key] || key;
}

function bestRunId() {
  const featured = state.overview?.overview?.featured || {};
  return featured.current_best_model?.run_id || featured.pure_coverage_winner?.run_id || null;
}

function renderActionAssetLinks(containerId, assets, emptyMessage, limit = 6) {
  const container = document.getElementById(containerId);
  if (!container) {
    return;
  }

  const items = (assets || []).slice(0, limit);
  if (!items.length) {
    container.innerHTML = `<span class="table-subtext">${escapeHtml(emptyMessage)}</span>`;
    return;
  }

  container.innerHTML = items
    .map(
      (asset) =>
        `<a href="/artifacts/${encodeURI(asset.path)}" target="_blank" rel="noopener noreferrer">${escapeHtml(asset.label || asset.name || asset.path)}</a>`,
    )
    .join("");
}

function runCoverageValue(run) {
  const parsed = Number(run?.metrics?.mean_coverage);
  return Number.isFinite(parsed) ? parsed : -1;
}

function topComparisonRuns(limit = 4) {
  const featured = state.overview?.overview?.featured || {};
  const selected = [];
  const seen = new Set();

  const add = (run) => {
    if (!run) {
      return;
    }
    const key = run.run_id || run.run_name;
    if (!key || seen.has(key)) {
      return;
    }
    seen.add(key);
    selected.push(run);
  };

  add(featured.current_best_model || featured.pure_coverage_winner || featured.balanced_agent);

  for (const run of [...state.runs].sort((a, b) => runCoverageValue(b) - runCoverageValue(a))) {
    add(run);
    if (selected.length >= limit) {
      break;
    }
  }

  if (selected.length < limit) {
    for (const run of [featured.pure_coverage_winner, featured.balanced_agent, featured.sneaky_specialist, featured.baseline_reference]) {
      add(run);
      if (selected.length >= limit) {
        break;
      }
    }
  }

  return selected.slice(0, limit);
}

function renderTopModelComparison() {
  const container = document.getElementById("top-model-comparison");
  if (!container) {
    return;
  }

  const leaders = topComparisonRuns(4);
  if (!leaders.length) {
    container.innerHTML = '<div class="table-subtext">No completed runs yet.</div>';
    return;
  }

  const championId = bestRunId();
  container.innerHTML = `
    <table class="compact-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Coverage</th>
          <th>just_go</th>
          <th>safe</th>
          <th>maze</th>
          <th>chokepoint</th>
          <th>sneaky</th>
          <th>Replay</th>
        </tr>
      </thead>
      <tbody>
        ${leaders
          .map((run) => {
            const isChampion = championId && championId === run.run_id;
            const replayLink = run.top_action_asset
              ? `<a class="quick-link" href="/artifacts/${encodeURI(run.top_action_asset.path)}" target="_blank" rel="noopener noreferrer">Replay</a>`
              : "—";
            const runLabel = run.run_name || run.run_id || "Unnamed run";
            const runControl = run.run_id
              ? `<button class="link-button" type="button" data-run-id="${escapeHtml(run.run_id)}">${escapeHtml(runLabel)}</button>`
              : `<span>${escapeHtml(runLabel)}</span>`;
            return `
              <tr class="${isChampion ? "best-row" : ""}">
                <td>
                  <div class="model-name">${runControl}${isChampion ? '<span class="best-tag">best</span>' : ""}</div>
                  <div class="table-subtext">${escapeHtml(familyLabel(run.family))}</div>
                </td>
                <td>${coverageChip(run.metrics?.mean_coverage)}</td>
                <td>${coverageChip(mapMetric(run, "just_go"))}</td>
                <td>${coverageChip(mapMetric(run, "safe"))}</td>
                <td>${coverageChip(mapMetric(run, "maze"))}</td>
                <td>${coverageChip(mapMetric(run, "chokepoint"))}</td>
                <td>${coverageChip(mapMetric(run, "sneaky_enemies"))}</td>
                <td>${replayLink}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;

  for (const button of container.querySelectorAll("button[data-run-id]")) {
    button.addEventListener("click", () => loadRunDetail(button.dataset.runId));
  }
}

function toggleTableColumn(className, visible) {
  for (const cell of document.querySelectorAll(`#runs-table .${className}`)) {
    cell.classList.toggle("show-col", visible);
  }
}

function applyTableColumnVisibility() {
  toggleTableColumn("col-family", Boolean(document.getElementById("show-family-col")?.checked));
  toggleTableColumn("col-map", Boolean(document.getElementById("show-map-col")?.checked));
  toggleTableColumn("col-runtime", Boolean(document.getElementById("show-runtime-col")?.checked));
}

function renderOverview() {
  const overviewPayload = state.overview?.overview;
  const cards = document.getElementById("overview-cards");
  const notes = document.getElementById("overview-notes");
  const meta = document.getElementById("repo-meta");
  const bestCard = document.getElementById("best-model-card");

  if (!overviewPayload) {
    if (cards) cards.innerHTML = "";
    if (notes) notes.textContent = "";
    if (bestCard) bestCard.innerHTML = "";
    if (meta) meta.textContent = "";
    renderActionAssetLinks("best-action-assets", [], "No sanity assets found for the current best model.");
    renderTopModelComparison();
    return;
  }

  const counts = overviewPayload.counts || {};
  const featured = overviewPayload.featured || {};
  const best = featured.current_best_model || featured.pure_coverage_winner || featured.balanced_agent;
  const bestCombo = featured.best_combo;

  if (bestCard) {
    if (!best) {
      bestCard.innerHTML = '<div class="table-subtext">No completed runs yet.</div>';
    } else {
      const leadAsset =
        state.bestRunDetail?.run_id && state.bestRunDetail.run_id === best.run_id
          ? (state.bestRunDetail.action_assets || [])[0]
          : best.top_action_asset;
      bestCard.innerHTML = `
        <div class="winner-card compact">
          <span class="best-badge">Best by coverage</span>
          <div class="best-title">${escapeHtml(best.run_name || "")}</div>
          <div class="best-subtitle">${escapeHtml(familyLabel(best.family))} • ${escapeHtml(focusLabel(best))}${best.map_suite ? ` • ${escapeHtml(best.map_suite)}` : ""}</div>
          <div class="best-metrics">
            <div class="best-metric"><span class="mini-label">Coverage</span><span class="mini-value">${formatPercent(best.metrics?.mean_coverage)}</span></div>
            <div class="best-metric"><span class="mini-label">Map mean</span><span class="mini-value">${formatPercent(best.primary_map_eval?.mean_coverage)}</span></div>
            <div class="best-metric"><span class="mini-label">Sneaky</span><span class="mini-value">${formatPercent(mapMetric(best, "sneaky_enemies"))}</span></div>
            <div class="best-metric"><span class="mini-label">Chokepoint</span><span class="mini-value">${formatPercent(mapMetric(best, "chokepoint"))}</span></div>
            <div class="best-metric"><span class="mini-label">Timesteps</span><span class="mini-value">${formatTimesteps(best.total_timesteps)}</span></div>
          </div>
          <div class="button-row">
            ${best.run_id ? `<button class="button" type="button" data-run-id="${escapeHtml(best.run_id)}">Open details</button>` : ""}
            ${leadAsset ? `<a class="button ghost" href="/artifacts/${encodeURI(leadAsset.path)}" target="_blank" rel="noopener noreferrer">Open top replay</a>` : ""}
          </div>
        </div>
      `;
    }
  }

  if (cards) {
    cards.innerHTML = [
      ["Completed", counts.runs_completed],
      ["Total runs", counts.runs_total],
      ["Combinations", counts.combinations_total],
      ["Queued", counts.queue_total],
    ]
      .map(
        ([label, value]) =>
          `<div class="card"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(formatNumber(value, 0))}</div></div>`,
      )
      .join("");
  }

  if (notes) {
    const comboText = bestCombo
      ? `${bestCombo.algorithm}/${bestCombo.observation}/${bestCombo.reward} leads combinations at ${formatPercent(bestCombo.mean_coverage)} coverage.`
      : "No combination summary yet.";
    notes.textContent = best
      ? `${best.run_name} leads at ${formatPercent(best.metrics?.mean_coverage)} coverage. ${comboText}`
      : comboText;
  }

  const featuredAssets =
    state.bestRunDetail?.run_id && best?.run_id && state.bestRunDetail.run_id === best.run_id
      ? state.bestRunDetail.action_assets || []
      : best?.top_action_asset
        ? [best.top_action_asset]
        : [];

  renderActionAssetLinks(
    "best-action-assets",
    featuredAssets,
    "No replay assets found for the current best model.",
    5,
  );

  renderTopModelComparison();

  if (meta) {
    meta.textContent = `Repo: ${state.overview.repo_root} | Queue: ${state.overview.queue_file}`;
  }

  const bestButton = bestCard?.querySelector("button[data-run-id]");
  if (bestButton) {
    bestButton.addEventListener("click", () => loadRunDetail(bestButton.dataset.runId));
  }
}

function renderExperimentChanges() {
  const changes = state.overview?.overview?.experiment_deltas || {};
  const reference = changes.reference;
  const referenceEl = document.getElementById("delta-reference");
  const improvementsEl = document.getElementById("improvement-list");
  const regressionsEl = document.getElementById("regression-list");

  if (!referenceEl || !improvementsEl || !regressionsEl) {
    return;
  }

  referenceEl.textContent = reference
    ? `Baseline: ${reference.run_name} (${formatPercent(reference.primary_map_eval?.mean_coverage)} map mean, ${formatPercent(mapMetric(reference, "sneaky_enemies"))} sneaky).`
    : "Baseline unavailable.";

  const renderDeltaList = (items, tone) => {
    if (!items?.length) {
      return `<div class="insight-item">No ${tone === "positive" ? "clear improvements" : "clear regressions"} beyond threshold.</div>`;
    }
    return items
      .map(
        (item) => `
          <div class="insight-item">
            <div class="insight-top">
              <div>
                <strong>${escapeHtml(item.run_name || "")}</strong><br />
                <span class="table-subtext">${escapeHtml(familyLabel(item.family))} • ${escapeHtml(focusLabel(item))}</span>
              </div>
              <div class="delta ${tone}">${formatDeltaPercent(item.delta_map_mean_coverage)}</div>
            </div>
            <div class="table-subtext">
              Map mean ${formatPercent(item.primary_map_eval?.mean_coverage)}
              ${item.delta_sneaky_coverage !== null && item.delta_sneaky_coverage !== undefined ? ` • sneaky ${formatDeltaPercent(item.delta_sneaky_coverage)}` : ""}
            </div>
          </div>
        `,
      )
      .join("");
  };

  improvementsEl.innerHTML = renderDeltaList(changes.improvements || [], "positive");
  regressionsEl.innerHTML = renderDeltaList(changes.regressions || [], "negative");
}

function renderSortHeaders() {
  const activeKey = normalizeSortKey(state.currentSort.by);
  const headers = document.querySelectorAll("#runs-table th.sort-header");
  for (const header of headers) {
    if (!header.dataset.baseLabel) {
      header.dataset.baseLabel = (header.textContent || "").trim();
    }
    header.textContent = header.dataset.baseLabel;
    header.classList.remove("active");

    const key = normalizeSortKey(header.dataset.sortKey || "");
    if (key !== activeKey) {
      continue;
    }
    header.classList.add("active");
    const arrow = document.createElement("span");
    arrow.className = "sort-arrow";
    arrow.textContent = state.currentSort.order === "asc" ? "▲" : "▼";
    header.appendChild(arrow);
  }
}

function renderRuns() {
  const tbody = document.querySelector("#runs-table tbody");
  tbody.innerHTML = "";

  if (!state.runs.length) {
    tbody.innerHTML = '<tr><td colspan="13">No runs match current filters.</td></tr>';
    renderSortHeaders();
    applyTableColumnVisibility();
    return;
  }

  const championId = bestRunId();

  state.runs.forEach((run, index) => {
    const metrics = run.metrics || {};
    const row = document.createElement("tr");
    row.classList.add("clickable");

    const isChampion = championId && championId === run.run_id;
    const isSelected = state.selectedRunId === run.run_id;
    if (isChampion) {
      row.classList.add("best-row");
    }
    if (isSelected) {
      row.classList.add("selected-row");
    }

    const topAction = run.top_action_asset;
    const replayLink = topAction
      ? `<a class="quick-link" href="/artifacts/${encodeURI(topAction.path)}" target="_blank" rel="noopener noreferrer">${escapeHtml(topAction.label || "Open")}</a>`
      : "—";

    row.innerHTML = `
      <td><span class="rank-pill ${isChampion ? "best" : ""}">${isChampion ? "#1" : `#${index + 1}`}</span></td>
      <td>
        <div class="model-name">${escapeHtml(run.run_name || run.run_id)}${isChampion ? '<span class="best-tag">★ best</span>' : ""}</div>
        <div class="table-subtext">${escapeHtml(focusLabel(run))}</div>
      </td>
      <td>${coverageChip(metrics.mean_coverage)}</td>
      <td>${coverageChip(run.primary_map_eval?.mean_coverage)}</td>
      <td>${coverageChip(mapMetric(run, "sneaky_enemies"))}</td>
      <td>${replayLink}</td>
      <td class="col-family"><span class="family-pill">${escapeHtml(familyLabel(run.family))}</span></td>
      <td class="col-map">${coverageChip(mapMetric(run, "just_go"))}</td>
      <td class="col-map">${coverageChip(mapMetric(run, "safe"))}</td>
      <td class="col-map">${coverageChip(mapMetric(run, "maze"))}</td>
      <td class="col-map">${coverageChip(mapMetric(run, "chokepoint"))}</td>
      <td class="col-runtime">${formatPercent(metrics.success_rate)}</td>
      <td class="col-runtime">${formatTimesteps(run.total_timesteps)}</td>
    `;

    const link = row.querySelector("a.quick-link");
    if (link) {
      link.addEventListener("click", (event) => {
        event.stopPropagation();
      });
    }

    row.addEventListener("click", () => loadRunDetail(run.run_id));
    tbody.appendChild(row);
  });

  renderSortHeaders();
  applyTableColumnVisibility();
}

function drawCurve(points) {
  const canvas = document.getElementById("curve-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!points || points.length < 2) {
    ctx.fillStyle = "#99acc6";
    ctx.font = "14px sans-serif";
    ctx.fillText("No learning curve points in evaluations.json", 12, 30);
    return;
  }

  const validPoints = points.filter((item) => item.timesteps !== null && item.timesteps !== undefined && item.mean_coverage !== null && item.mean_coverage !== undefined);
  if (validPoints.length < 2) {
    ctx.fillStyle = "#99acc6";
    ctx.font = "14px sans-serif";
    ctx.fillText("Learning curve exists but does not contain enough numeric coverage points", 12, 30);
    return;
  }

  const xValues = validPoints.map((item) => Number(item.timesteps));
  const yValues = validPoints.map((item) => Number(item.mean_coverage));
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const xPadding = 48;
  const yPadding = 24;
  const width = canvas.width - xPadding * 2;
  const height = canvas.height - yPadding * 2;

  ctx.strokeStyle = "#273140";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(xPadding, yPadding);
  ctx.lineTo(xPadding, canvas.height - yPadding);
  ctx.lineTo(canvas.width - xPadding, canvas.height - yPadding);
  ctx.stroke();

  const xScale = (value) => {
    if (maxX === minX) return xPadding;
    return xPadding + ((value - minX) / (maxX - minX)) * width;
  };
  const yScale = (value) => {
    if (maxY === minY) return canvas.height - yPadding;
    return canvas.height - yPadding - ((value - minY) / (maxY - minY)) * height;
  };

  ctx.strokeStyle = "#57a7ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  validPoints.forEach((point, index) => {
    const x = xScale(Number(point.timesteps));
    const y = yScale(Number(point.mean_coverage));
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  ctx.fillStyle = "#dbe5f2";
  for (const point of validPoints) {
    const x = xScale(Number(point.timesteps));
    const y = yScale(Number(point.mean_coverage));
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#99acc6";
  ctx.font = "12px sans-serif";
  ctx.fillText(`t=${formatTimesteps(minX)}`, xPadding, canvas.height - 6);
  ctx.fillText(`t=${formatTimesteps(maxX)}`, canvas.width - xPadding - 90, canvas.height - 6);
  ctx.fillText(`cov=${formatNumber(maxY)}`, 6, yPadding + 4);
  ctx.fillText(`cov=${formatNumber(minY)}`, 6, canvas.height - yPadding);
}

function primaryMapSummaryTable(primaryMapEval) {
  const rows = primaryMapEval?.rows || [];
  if (!rows.length) {
    return "No per-map summary found for this run.";
  }
  return `
    <div class="table-subtext">Source: ${escapeHtml(primaryMapEval.source_name || primaryMapEval.source_dir || "map_eval")}</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Map</th>
            <th>Coverage</th>
            <th>Success</th>
            <th>Death</th>
            <th>Timeout</th>
            <th>Mean reward</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) => `
                <tr>
                  <td><strong>${escapeHtml(row.map_name || "")}</strong></td>
                  <td>${coverageChip(row.mean_coverage)}</td>
                  <td>${formatPercent(row.success_rate)}</td>
                  <td>${formatPercent(row.death_rate)}</td>
                  <td>${formatPercent(row.timeout_rate)}</td>
                  <td>${formatNumber(row.mean_reward)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderRunSummary(run) {
  const container = document.getElementById("run-summary");
  const rows = run.primary_map_eval?.rows || [];
  if (!rows.length) {
    container.innerHTML = '<div class="insight-item">No primary per-map evaluation is available for this run yet.</div>';
    return;
  }

  const strongest = [...rows].sort((a, b) => Number(b.mean_coverage || 0) - Number(a.mean_coverage || 0))[0];
  const weakest = [...rows].sort((a, b) => Number(a.mean_coverage || 0) - Number(b.mean_coverage || 0))[0];
  const sneaky = rows.find((row) => row.map_name === "sneaky_enemies");
  const topAction = (run.action_assets || [])[0];
  const topReplay = topAction
    ? `<a class="quick-link" href="/artifacts/${encodeURI(topAction.path)}" target="_blank" rel="noopener noreferrer">${escapeHtml(topAction.label || topAction.name || "Open replay")}</a>`
    : "—";

  container.innerHTML = [
    {
      label: "Evaluation source",
      value: run.primary_map_eval?.source_name || run.primary_map_eval?.source_dir || "map_eval",
    },
    {
      label: "Strongest map",
      value: strongest ? `${strongest.map_name} (${formatPercent(strongest.mean_coverage)})` : "—",
    },
    {
      label: "Weakest map",
      value: weakest ? `${weakest.map_name} (${formatPercent(weakest.mean_coverage)})` : "—",
    },
    {
      label: "Sneaky coverage",
      value: sneaky ? formatPercent(sneaky.mean_coverage) : "—",
    },
    {
      label: "Top replay",
      value: topReplay,
      isHtml: true,
    },
  ]
    .map(
      (item) => `
        <div class="insight-item">
          <div class="mini-label">${escapeHtml(item.label)}</div>
          <div class="mini-value">${item.isHtml ? item.value : escapeHtml(item.value)}</div>
        </div>
      `,
    )
    .join("");
}

function renderRunDetail(run) {
  const empty = document.getElementById("run-detail-empty");
  const detail = document.getElementById("run-detail");
  const detailPanel = document.getElementById("run-detail-panel");
  if (!run) {
    empty.classList.remove("hidden");
    detail.classList.add("hidden");
    return;
  }

  empty.classList.add("hidden");
  detail.classList.remove("hidden");
  if (detailPanel && detailPanel.tagName === "DETAILS") {
    detailPanel.open = true;
  }

  const metrics = run.metrics || {};
  const metricGrid = document.getElementById("run-metrics");
  metricGrid.innerHTML = [
    ["Run", run.run_name],
    ["Family", familyLabel(run.family)],
    ["Map Suite", run.map_suite],
    ["Focus", focusLabel(run)],
    ["Overall Coverage", formatPercent(metrics.mean_coverage)],
    ["Map Mean", formatPercent(run.primary_map_eval?.mean_coverage)],
    ["Sneaky Coverage", formatPercent(mapMetric(run, "sneaky_enemies"))],
    ["Chokepoint Coverage", formatPercent(mapMetric(run, "chokepoint"))],
    ["Success Rate", formatPercent(metrics.success_rate)],
    ["Death Rate", formatPercent(metrics.death_rate)],
    ["Mean Reward", formatNumber(metrics.mean_reward)],
    ["Timesteps", formatTimesteps(run.total_timesteps)],
  ]
    .map(
      ([name, value]) =>
        `<div class="metric"><div class="name">${escapeHtml(name)}</div><div class="value">${escapeHtml(value ?? "—")}</div></div>`,
    )
    .join("");

  renderRunSummary(run);
  renderActionAssetLinks("run-action-assets", run.action_assets || [], "No sanity playback/replay assets for this model.", 10);
  drawCurve(run.evaluation_points || []);
  document.getElementById("primary-map-summary").innerHTML = primaryMapSummaryTable(run.primary_map_eval);

  const mapSummary = document.getElementById("map-eval-summary");
  mapSummary.innerHTML = "";
  for (const item of run.map_eval_summaries || []) {
    const rows = item.rows || [];
    if (!rows.length) {
      continue;
    }
    const section = document.createElement("div");
    section.className = "subpanel";
    section.innerHTML = `
      <h4>${escapeHtml(item.source_name || item.map_eval_dir || "map_eval")}</h4>
      ${primaryMapSummaryTable(item)}
    `;
    mapSummary.appendChild(section);
  }
  if (!mapSummary.innerHTML) {
    mapSummary.textContent = "No additional per-map summaries found for this run.";
  }

  const artifacts = document.getElementById("run-artifacts");
  artifacts.innerHTML = (run.artifacts || [])
    .map(
      (artifact) =>
        `<a href="/artifacts/${encodeURI(artifact.path)}" target="_blank" rel="noopener noreferrer">${escapeHtml(artifact.path)}</a>`,
    )
    .join("");

  const actionImagePaths = (run.action_assets || [])
    .map((asset) => asset.path)
    .filter((path) => /\.(png|gif)$/i.test(path));
  const mapImagePaths = Array.from(new Set([...(run.map_eval_images || []), ...actionImagePaths]));

  const mapImages = document.getElementById("map-images");
  mapImages.innerHTML = mapImagePaths
    .map(
      (path) => `
        <a href="/artifacts/${encodeURI(path)}" target="_blank" rel="noopener noreferrer">
          <img src="/artifacts/${encodeURI(path)}" alt="${escapeHtml(path)}" />
        </a>
      `,
    )
    .join("");
}

function renderCombinations() {
  const tbody = document.querySelector("#combo-table tbody");
  tbody.innerHTML = "";
  for (const item of state.combos) {
    const row = document.createElement("tr");
    const width = Math.max(0, Math.min(100, Number(item.mean_coverage || 0) * 100));
    row.innerHTML = `
      <td><strong>${escapeHtml(item.algorithm)}</strong><br><span class="table-subtext">${escapeHtml(item.observation)} / ${escapeHtml(item.reward)}</span></td>
      <td>${escapeHtml(String(item.run_count || 0))}</td>
      <td class="bar-cell"><span class="bar" style="width: ${width}%;"></span>${formatPercent(item.mean_coverage)}</td>
      <td>${formatPercent(item.success_rate)}</td>
      <td>${formatNumber(item.mean_reward)}</td>
      <td>${formatPercent(item.death_rate)}</td>
    `;
    tbody.appendChild(row);
  }
}

function renderResults() {
  const resultLinks = document.getElementById("results-links");
  const sweepPlots = document.getElementById("sweep-plots");
  const leaderboard = document.getElementById("sweep-leaderboard");
  const inboxPreview = document.getElementById("inbox-preview");

  if (!state.results) {
    resultLinks.innerHTML = "";
    sweepPlots.innerHTML = "";
    leaderboard.innerHTML = "";
    inboxPreview.textContent = "";
    return;
  }

  const links = [];
  for (const path of state.results.results_documents || []) {
    links.push(path);
  }
  for (const path of state.results.sweep?.documents || []) {
    links.push(path);
  }
  resultLinks.innerHTML = Array.from(new Set(links))
    .map(
      (path) =>
        `<a href="/artifacts/${encodeURI(path)}" target="_blank" rel="noopener noreferrer">${escapeHtml(path)}</a>`,
    )
    .join("");

  sweepPlots.innerHTML = (state.results.sweep?.plots || [])
    .map(
      (path) => `
        <a href="/artifacts/${encodeURI(path)}" target="_blank" rel="noopener noreferrer">
          <img src="/artifacts/${encodeURI(path)}" alt="${escapeHtml(path)}" />
        </a>
      `,
    )
    .join("");

  const rows = state.results.sweep?.leaderboard || [];
  if (rows.length) {
    leaderboard.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Obs</th>
            <th>Reward</th>
            <th>Coverage</th>
            <th>Success</th>
            <th>Mean Reward</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .slice(0, 12)
            .map(
              (row) => `
                <tr>
                  <td>${escapeHtml(row.run_name || "")}</td>
                  <td>${escapeHtml(row.observation || "")}</td>
                  <td>${escapeHtml(row.reward || "")}</td>
                  <td>${formatPercent(row.mean_coverage)}</td>
                  <td>${formatPercent(row.success_rate)}</td>
                  <td>${formatNumber(row.mean_reward)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    `;
  } else {
    leaderboard.innerHTML = "No sweep leaderboard found in results/observation_reward_sweep.";
  }

  const inbox = state.results.experiment_inbox || {};
  inboxPreview.textContent = inbox.preview || "No inbox preview available.";
}

function plannerSpecFromForm() {
  return {
    base_config: document.getElementById("planner-base-config").value,
    experiment_name: document.getElementById("planner-experiment-name").value,
    algorithm: document.getElementById("planner-algorithm").value,
    observation: document.getElementById("planner-observation").value,
    reward: document.getElementById("planner-reward").value,
    map_suite: document.getElementById("planner-map-suite").value,
    total_timesteps: document.getElementById("planner-total-timesteps").value,
    seed: document.getElementById("planner-seed").value,
    output_dir: document.getElementById("planner-output-dir").value,
    notes: document.getElementById("planner-notes").value,
  };
}

function renderPreview() {
  const commandEl = document.getElementById("command-preview");
  const configEl = document.getElementById("config-preview");
  if (!state.preview) {
    commandEl.textContent = "Click 'Generate Preview' to build a queued command/config preview.";
    configEl.textContent = "";
    return;
  }
  commandEl.textContent = state.preview.command_preview || "";
  const previewPayload = {
    changed_fields: state.preview.changed_fields,
    requires_materialized_config: state.preview.requires_materialized_config,
    suggested_config_path: state.preview.suggested_config_path,
    config_overrides: state.preview.config_overrides,
  };
  configEl.textContent = JSON.stringify(previewPayload, null, 2);
}

function renderQueue() {
  const queueFile = document.getElementById("queue-file-path");
  const tbody = document.querySelector("#queue-table tbody");
  queueFile.textContent = state.queue.queue_file ? `Queue file: ${state.queue.queue_file}` : "Queue file unavailable.";
  const items = state.queue.items || [];
  tbody.innerHTML = items
    .map(
      (item) => `
        <tr>
          <td>${escapeHtml(item.id || "")}</td>
          <td>${escapeHtml(item.created_at || "")}</td>
          <td>${escapeHtml(item.base_config || "")}</td>
          <td>${escapeHtml(item.status || "")}</td>
          <td><code>${escapeHtml(item.preview?.command_preview || "")}</code></td>
        </tr>
      `,
    )
    .join("");
}

function setSort(by, order = null, toggleIfSame = false) {
  const normalized = normalizeSortKey(by);
  if (toggleIfSame && normalized === state.currentSort.by) {
    state.currentSort.order = state.currentSort.order === "asc" ? "desc" : "asc";
  } else {
    state.currentSort.by = normalized;
    state.currentSort.order = order || (normalized === "family" ? "asc" : "desc");
  }

  const sortBy = document.getElementById("sort-by");
  const sortDir = document.getElementById("sort-dir");
  if (sortBy && sortBy.value !== state.currentSort.by) {
    sortBy.value = state.currentSort.by;
  }
  if (sortDir && sortDir.value !== state.currentSort.order) {
    sortDir.value = state.currentSort.order;
  }

  renderSortHeaders();
}

async function loadRunDetail(runId) {
  state.selectedRunId = runId;
  renderRuns();
  const detail = await fetchJson(`/api/runs/${encodeURIComponent(runId)}`);
  if (runId === bestRunId()) {
    state.bestRunDetail = detail;
    renderOverview();
  }
  renderRunDetail(detail);
}

async function refreshBestRunDetail() {
  const runId = bestRunId();
  if (!runId) {
    state.bestRunDetail = null;
    return;
  }
  try {
    state.bestRunDetail = await fetchJson(`/api/runs/${encodeURIComponent(runId)}`);
  } catch (error) {
    state.bestRunDetail = null;
  }
}

async function reloadRunsAndCombos() {
  const filters = getFilters();
  const runsQuery = toQueryString({
    ...filters,
    sort: state.currentSort.by,
    order: state.currentSort.order,
  });
  const combosQuery = toQueryString(filters);

  const [runsResponse, combosResponse] = await Promise.all([
    fetchJson(`/api/runs${runsQuery}`),
    fetchJson(`/api/combinations${combosQuery}`),
  ]);

  state.runs = runsResponse.items || [];
  state.combos = combosResponse.items || [];
  renderRuns();
  renderCombinations();
  renderTopModelComparison();
}

async function refreshAll() {
  const [overview, configs, results, queue] = await Promise.all([
    fetchJson("/api/overview"),
    fetchJson("/api/configs"),
    fetchJson("/api/results"),
    fetchJson("/api/queue"),
  ]);
  state.overview = overview;
  state.configs = configs.items || [];
  state.results = results;
  state.queue = queue;

  setSelectOptions("filter-family", overview.options.family);
  setSelectOptions("filter-algorithm", overview.options.algorithm);
  setSelectOptions("filter-observation", overview.options.observation);
  setSelectOptions("filter-reward", overview.options.reward);
  setSelectOptions("filter-map-suite", overview.options.map_suite);
  setSelectOptions("filter-status", ["completed", "configured", "partial"], true, "(any)");

  setSelectOptions("planner-base-config", state.configs.map((item) => item.path), false);
  setSelectOptions("planner-algorithm", overview.options.algorithm, true, "(inherit base)");
  setSelectOptions("planner-observation", overview.options.observation, true, "(inherit base)");
  setSelectOptions("planner-reward", overview.options.reward, true, "(inherit base)");
  setSelectOptions("planner-map-suite", overview.options.map_suite, true, "(inherit base)");

  setSort(state.currentSort.by, state.currentSort.order);

  renderExperimentChanges();
  renderResults();
  renderQueue();
  renderPreview();

  await reloadRunsAndCombos();
  await refreshBestRunDetail();
  renderOverview();

  if (state.selectedRunId && state.runs.some((run) => run.run_id === state.selectedRunId)) {
    await loadRunDetail(state.selectedRunId);
  } else {
    state.selectedRunId = null;
    renderRunDetail(null);
    renderRuns();
  }
}

async function generatePreview() {
  const payload = plannerSpecFromForm();
  state.preview = await fetchJson("/api/queue/preview", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderPreview();
}

async function queueExperiment() {
  const payload = plannerSpecFromForm();
  const entry = await fetchJson("/api/queue", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.preview = entry.preview;
  state.queue = await fetchJson("/api/queue");
  renderPreview();
  renderQueue();
}

function wireEvents() {
  document.getElementById("refresh-btn").addEventListener("click", () => {
    refreshAll().catch((error) => alert(error.message));
  });

  for (const id of filterIds) {
    const eventName = id === "filter-q" ? "input" : "change";
    document.getElementById(id).addEventListener(eventName, () => {
      reloadRunsAndCombos().catch((error) => alert(error.message));
    });
  }

  document.getElementById("sort-by").addEventListener("change", (event) => {
    const sortBy = event.target.value;
    setSort(sortBy);
    reloadRunsAndCombos().catch((error) => alert(error.message));
  });

  document.getElementById("sort-dir").addEventListener("change", (event) => {
    const order = String(event.target.value || "desc").toLowerCase() === "asc" ? "asc" : "desc";
    setSort(state.currentSort.by, order);
    reloadRunsAndCombos().catch((error) => alert(error.message));
  });

  for (const header of document.querySelectorAll("#runs-table th.sort-header")) {
    header.addEventListener("click", () => {
      setSort(header.dataset.sortKey || "coverage", null, true);
      reloadRunsAndCombos().catch((error) => alert(error.message));
    });
  }

  for (const id of ["show-family-col", "show-map-col", "show-runtime-col"]) {
    const checkbox = document.getElementById(id);
    if (!checkbox) {
      continue;
    }
    checkbox.addEventListener("change", () => {
      applyTableColumnVisibility();
    });
  }

  document.getElementById("preview-btn").addEventListener("click", () => {
    generatePreview().catch((error) => alert(error.message));
  });

  document.getElementById("queue-btn").addEventListener("click", () => {
    queueExperiment().catch((error) => alert(error.message));
  });
}

wireEvents();
refreshAll().catch((error) => {
  alert(error.message);
});
