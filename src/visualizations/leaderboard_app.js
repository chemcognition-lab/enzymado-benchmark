/**
 * LeaderboardApp — owns the page's state (filters, active families, theme,
 * current view) and wires the DOM controls to LeaderboardCharts' stateless
 * renderers. Call LeaderboardApp.init(DATA) once the page and the data
 * payload are both ready.
 */
(function (global) {
  "use strict";

  var FAMILY_LABEL = { linear: "Linear", mlp: "MLP", xgboost: "XGBoost" };
  var SPLIT_LABEL = { random: "Random split", loco: "Similarity-clustered (LOCO)" };
  var MODE_LABEL = { no_attention: "Mean-pooled", attention: "Attention-pooled" };
  var METRIC_LABEL = {
    mae: "MAE", rmse: "RMSE", r2: "R²", kendall_tau: "Kendall's τ",
    accuracy: "Accuracy", precision: "Precision", recall: "Recall", f1: "F1",
  };
  var SPLIT_ORDER = ["random", "loco"];
  var MODE_ORDER = ["no_attention", "attention"];

  function init(DATA) {
    var Charts = global.LeaderboardCharts;

    var els = {
      root: document.documentElement,
      vizRoot: document.querySelector(".viz-root"),
      taskSelect: document.getElementById("task-select"),
      splitSelect: document.getElementById("split-select"),
      modeSelect: document.getElementById("mode-select"),
      chartTitle: document.getElementById("chart-title"),
      chartSubtitle: document.getElementById("chart-subtitle"),
      chartNotes: document.getElementById("chart-notes"),
      legend: document.getElementById("legend"),
      svg: document.getElementById("chart"),
      tooltip: document.getElementById("tooltip"),
      chartWrap: document.getElementById("chart-wrap"),
      emptyState: document.getElementById("empty-state"),
      tableWrap: document.getElementById("table-wrap"),
      viewChartBtn: document.getElementById("view-chart"),
      viewTableBtn: document.getElementById("view-table"),
      themeToggle: document.getElementById("theme-toggle"),
      exportSvgBtn: document.getElementById("export-svg"),
      exportPngBtn: document.getElementById("export-png"),
    };

    var state = {
      activeFamilies: {},
      view: "chart",
    };
    DATA.model_families.forEach(function (f) {
      state.activeFamilies[f] = true;
    });

    // ---- controls -----------------------------------------------------------

    function populateTaskSelect() {
      var groups = { properties: "Benchmark properties", real_case: "Real-case" };
      var optgroups = {};
      DATA.task_order.forEach(function (task) {
        var meta = DATA.tasks[task];
        var groupKey = groups[meta.scope] ? meta.scope : "properties";
        if (!optgroups[groupKey]) {
          var og = document.createElement("optgroup");
          og.label = groups[groupKey];
          els.taskSelect.appendChild(og);
          optgroups[groupKey] = og;
        }
        var opt = document.createElement("option");
        opt.value = task;
        opt.textContent = task;
        optgroups[groupKey].appendChild(opt);
      });
    }

    function availableCombosForTask(task) {
      var combos = {};
      DATA.records.forEach(function (r) {
        if (r.task !== task) return;
        combos[r.split + "|" + r.mode] = { split: r.split, mode: r.mode };
      });
      return combos;
    }

    function uniquePreferring(arr, preferredOrder) {
      var seen = {};
      var out = [];
      arr.forEach(function (v) {
        if (!seen[v]) {
          seen[v] = true;
          out.push(v);
        }
      });
      out.sort(function (a, b) {
        var ia = preferredOrder.indexOf(a), ib = preferredOrder.indexOf(b);
        if (ia === -1) ia = preferredOrder.length;
        if (ib === -1) ib = preferredOrder.length;
        return ia !== ib ? ia - ib : a.localeCompare(b);
      });
      return out;
    }

    function fillSelect(select, values, labelMap) {
      var prior = select.value;
      select.innerHTML = "";
      values.forEach(function (v) {
        var opt = document.createElement("option");
        opt.value = v;
        opt.textContent = labelMap[v] || v;
        select.appendChild(opt);
      });
      if (values.indexOf(prior) !== -1) select.value = prior;
    }

    function populateSplitAndMode() {
      var combos = availableCombosForTask(els.taskSelect.value);
      var splits = uniquePreferring(Object.keys(combos).map(function (k) { return combos[k].split; }), SPLIT_ORDER);
      var modes = uniquePreferring(Object.keys(combos).map(function (k) { return combos[k].mode; }), MODE_ORDER);
      fillSelect(els.splitSelect, splits, SPLIT_LABEL);
      fillSelect(els.modeSelect, modes, MODE_LABEL);
    }

    // ---- data shaping ---------------------------------------------------------

    function currentRows() {
      var task = els.taskSelect.value, split = els.splitSelect.value, mode = els.modeSelect.value;
      return DATA.records.filter(function (r) {
        return r.task === task && r.split === split && r.mode === mode;
      });
    }

    function groupByEmbedding(rows) {
      var byEmbedding = {};
      rows.forEach(function (r) {
        if (!byEmbedding[r.embedding]) byEmbedding[r.embedding] = {};
        byEmbedding[r.embedding][r.family] = r;
      });
      return byEmbedding;
    }

    function rankedEmbeddings(byEmbedding, higherIsBetter) {
      var entries = Object.keys(byEmbedding).map(function (emb) {
        var families = byEmbedding[emb];
        var vals = DATA.model_families
          .filter(function (f) { return families[f]; })
          .map(function (f) { return families[f].value; });
        var mean = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
        return { embedding: emb, families: families, rankValue: mean };
      });
      entries.sort(function (a, b) {
        return higherIsBetter ? b.rankValue - a.rankValue : a.rankValue - b.rankValue;
      });
      return entries;
    }

    // ---- rendering --------------------------------------------------------

    function exportFilename(ext) {
      var safe = function (s) { return String(s).replace(/[^a-z0-9]+/gi, "-"); };
      return [
        "leaderboard", safe(els.taskSelect.value), safe(els.splitSelect.value), safe(els.modeSelect.value),
      ].join("-") + "." + ext;
    }

    function render() {
      var task = els.taskSelect.value;
      if (!task) return;
      var meta = DATA.tasks[task];
      var byEmbedding = groupByEmbedding(currentRows());
      var ranked = rankedEmbeddings(byEmbedding, meta.higher_is_better);
      var theme = Charts.resolveTheme(els.vizRoot);

      els.chartTitle.textContent = task;
      els.chartSubtitle.textContent =
        METRIC_LABEL[meta.metric] + " on the test split · " +
        (meta.higher_is_better ? "higher is better" : "lower is better") +
        " · " + meta.task_type;
      els.chartNotes.textContent = ranked.length
        ? "Ranked by mean " + METRIC_LABEL[meta.metric] + " across the model families that completed for each embedding."
        : "";

      Charts.renderLegend(els.legend, {
        families: DATA.model_families,
        activeFamilies: state.activeFamilies,
        familyLabel: FAMILY_LABEL,
        theme: theme,
        onToggle: function (family) {
          state.activeFamilies[family] = !state.activeFamilies[family];
          render();
        },
      });

      els.emptyState.hidden = ranked.length > 0;
      els.svg.style.display = ranked.length ? "block" : "none";
      var hasData = ranked.length > 0;
      els.exportSvgBtn.disabled = !hasData;
      els.exportPngBtn.disabled = !hasData;

      if (state.view === "chart") {
        Charts.renderBarChart(els.svg, {
          ranked: ranked, meta: meta, families: DATA.model_families,
          activeFamilies: state.activeFamilies, familyLabel: FAMILY_LABEL,
          theme: theme, chartWrap: els.chartWrap, tooltip: els.tooltip,
        });
      } else {
        Charts.renderTable(els.tableWrap, {
          ranked: ranked, meta: meta, families: DATA.model_families,
          familyLabel: FAMILY_LABEL, metricLabel: METRIC_LABEL[meta.metric],
        });
      }
    }

    // ---- wiring -------------------------------------------------------------

    function setView(view) {
      state.view = view;
      els.viewChartBtn.setAttribute("aria-pressed", String(view === "chart"));
      els.viewTableBtn.setAttribute("aria-pressed", String(view === "table"));
      els.chartWrap.hidden = view !== "chart";
      els.tableWrap.hidden = view !== "table";
      render();
    }

    els.viewChartBtn.addEventListener("click", function () { setView("chart"); });
    els.viewTableBtn.addEventListener("click", function () { setView("table"); });

    els.taskSelect.addEventListener("change", function () { populateSplitAndMode(); render(); });
    els.splitSelect.addEventListener("change", render);
    els.modeSelect.addEventListener("change", render);
    window.addEventListener("resize", function () {
      if (state.view === "chart") render();
    });

    els.themeToggle.addEventListener("click", function () {
      var isDark = els.root.getAttribute("data-theme") === "dark";
      els.root.setAttribute("data-theme", isDark ? "light" : "dark");
      els.themeToggle.setAttribute("aria-pressed", String(!isDark));
      render();
    });

    els.exportSvgBtn.addEventListener("click", function () {
      Charts.downloadSvg(els.svg, exportFilename("svg"));
    });
    els.exportPngBtn.addEventListener("click", function () {
      Charts.downloadPng(els.svg, exportFilename("png"));
    });

    populateTaskSelect();
    populateSplitAndMode();
    render();
  }

  global.LeaderboardApp = { init: init };
})(window);
