/**
 * LeaderboardCharts — stateless rendering primitives for the leaderboard chart.
 *
 * Nothing in this file reads or writes app state (filters, active families,
 * current view). Every function takes its inputs as arguments and draws into
 * the DOM node it's given, so it can be unit-exercised or reused by a
 * different app shell without change. leaderboard_app.js owns the state and
 * calls into these.
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif";

  // ---- theme -----------------------------------------------------------
  //
  // Colors are resolved to concrete values (not `var(--x)` references) once
  // per render, from whichever CSS custom properties are active on `el` at
  // that moment (light/dark). Every mark is then drawn with a plain hex/rgb
  // attribute. That's what makes the chart's own SVG markup exportable as a
  // standalone file: it never depends on the page's stylesheet to render
  // correctly, so an exported .svg (or a PNG rasterized from it) looks
  // identical outside this page.

  function resolveTheme(el) {
    var cs = getComputedStyle(el);
    function v(name) {
      return cs.getPropertyValue(name).trim();
    }
    return {
      surface: v("--surface-1"),
      textPrimary: v("--text-primary"),
      textSecondary: v("--text-secondary"),
      textMuted: v("--text-muted"),
      gridline: v("--gridline"),
      baseline: v("--baseline"),
      border: v("--border"),
      hoverWash: v("--hover-wash"),
      families: {
        linear: v("--series-linear"),
        mlp: v("--series-mlp"),
        xgboost: v("--series-xgboost"),
      },
    };
  }

  function formatValue(value) {
    return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(3);
  }

  // ---- legend ------------------------------------------------------------

  function renderLegend(container, opts) {
    var families = opts.families;
    var activeFamilies = opts.activeFamilies;
    var familyLabel = opts.familyLabel;
    var theme = opts.theme;
    var onToggle = opts.onToggle;

    container.innerHTML = "";
    families.forEach(function (family) {
      var isActive = !!activeFamilies[family];
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "legend-chip";
      btn.setAttribute("aria-pressed", String(isActive));
      btn.style.setProperty("--chip-color", theme.families[family]);

      var swatch = document.createElement("span");
      swatch.className = "swatch";
      btn.appendChild(swatch);

      var label = document.createElement("span");
      label.textContent = familyLabel[family] || family;
      btn.appendChild(label);

      btn.addEventListener("click", function () {
        onToggle(family);
      });
      container.appendChild(btn);
    });
  }

  // ---- bar chart -----------------------------------------------------------

  function roundedBarPath(x, y, w, h, r, roundTop) {
    r = Math.min(r, w / 2, h);
    if (roundTop) {
      return (
        "M" + x + "," + (y + h) +
        " L" + x + "," + (y + r) +
        " Q" + x + "," + y + " " + (x + r) + "," + y +
        " L" + (x + w - r) + "," + y +
        " Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) +
        " L" + (x + w) + "," + (y + h) + " Z"
      );
    }
    return (
      "M" + x + "," + y +
      " L" + (x + w) + "," + y +
      " L" + (x + w) + "," + (y + h - r) +
      " Q" + (x + w) + "," + (y + h) + " " + (x + w - r) + "," + (y + h) +
      " L" + (x + r) + "," + (y + h) +
      " Q" + x + "," + (y + h) + " " + x + "," + (y + h - r) +
      " Z"
    );
  }

  function textEl(attrs, content) {
    var t = document.createElementNS(SVGNS, "text");
    t.setAttribute("font-family", FONT_FAMILY);
    for (var key in attrs) t.setAttribute(key, attrs[key]);
    if (content != null) t.textContent = content;
    return t;
  }

  function attachTooltip(node, tooltipEl, chartWrapEl, info) {
    function show() {
      var rectBox = chartWrapEl.getBoundingClientRect();
      var nodeBox = node.getBoundingClientRect();
      tooltipEl.innerHTML = "";

      var key = document.createElement("span");
      key.className = "tt-key";
      key.style.background = info.color;
      tooltipEl.appendChild(key);

      var metaEl = document.createElement("span");
      metaEl.className = "tt-meta";
      metaEl.appendChild(document.createTextNode(info.embedding + " · " + info.familyLabel));
      tooltipEl.appendChild(metaEl);
      tooltipEl.appendChild(document.createElement("br"));

      var val = document.createElement("span");
      val.className = "tt-value";
      val.textContent = formatValue(info.value);
      tooltipEl.appendChild(val);

      var n = document.createElement("span");
      n.className = "tt-meta";
      n.textContent = " (n=" + info.n + ")";
      tooltipEl.appendChild(n);

      tooltipEl.style.left = nodeBox.left - rectBox.left + nodeBox.width / 2 + "px";
      tooltipEl.style.top = nodeBox.top - rectBox.top + "px";
      tooltipEl.classList.add("visible");
    }
    function hide() {
      tooltipEl.classList.remove("visible");
    }
    node.addEventListener("pointerenter", show);
    node.addEventListener("pointermove", show);
    node.addEventListener("pointerleave", hide);
    node.addEventListener("focus", show);
    node.addEventListener("blur", hide);
  }

  /**
   * Draws the grouped bar chart into `svg`.
   * opts: { ranked, meta, families, activeFamilies, familyLabel, theme,
   *         chartWrap, tooltip }
   */
  function renderBarChart(svg, opts) {
    var ranked = opts.ranked;
    var meta = opts.meta;
    var theme = opts.theme;
    var familyLabel = opts.familyLabel;
    var visibleFamilies = opts.families.filter(function (f) {
      return opts.activeFamilies[f];
    });

    svg.innerHTML = "";
    if (!ranked.length || !visibleFamilies.length) return;

    var width = Math.max(opts.chartWrap.clientWidth, 320);
    var marginTop = 28, marginRight = 16, marginBottom = 96, marginLeft = 52;
    var barBand = visibleFamilies.length;
    var groupGap = 22;
    var groupWidth = Math.max(40, Math.min(140, (width - marginLeft - marginRight) / ranked.length - groupGap));
    var innerWidth = ranked.length * (groupWidth + groupGap);
    width = Math.max(width, innerWidth + marginLeft + marginRight);
    var height = 380;
    var plotH = height - marginTop - marginBottom;

    var allValues = [];
    ranked.forEach(function (e) {
      visibleFamilies.forEach(function (f) {
        if (e.families[f]) allValues.push(e.families[f].value);
      });
    });
    var dataMin = Math.min.apply(null, allValues.concat([0]));
    var dataMax = Math.max.apply(null, allValues.concat([0]));
    var pad = (dataMax - dataMin) * 0.12 || 1;
    var yMax = dataMax + pad;
    var yMin = dataMin - (dataMin < 0 ? pad : 0);
    var yScale = function (v) {
      return marginTop + plotH * (1 - (v - yMin) / (yMax - yMin));
    };
    var baselineY = yScale(0);

    svg.setAttribute("viewBox", "0 0 " + width + " " + height);

    // background — needed so an exported standalone SVG/PNG has a surface
    // rather than a transparent hole where the page used to show through.
    var bg = document.createElementNS(SVGNS, "rect");
    bg.setAttribute("x", 0);
    bg.setAttribute("y", 0);
    bg.setAttribute("width", width);
    bg.setAttribute("height", height);
    bg.setAttribute("fill", theme.surface);
    svg.appendChild(bg);

    // gridlines + y ticks
    var ticks = 5;
    for (var t = 0; t <= ticks; t++) {
      var v = yMin + (t / ticks) * (yMax - yMin);
      var y = yScale(v);
      var line = document.createElementNS(SVGNS, "line");
      line.setAttribute("x1", marginLeft);
      line.setAttribute("x2", width - marginRight);
      line.setAttribute("y1", y);
      line.setAttribute("y2", y);
      line.setAttribute("stroke", theme.gridline);
      line.setAttribute("stroke-width", "1");
      svg.appendChild(line);

      svg.appendChild(
        textEl(
          { x: marginLeft - 8, y: y + 3.5, "text-anchor": "end", "font-size": "11", fill: theme.textMuted },
          formatValue(v)
        )
      );
    }

    // baseline
    var base = document.createElementNS(SVGNS, "line");
    base.setAttribute("x1", marginLeft);
    base.setAttribute("x2", width - marginRight);
    base.setAttribute("y1", baselineY);
    base.setAttribute("y2", baselineY);
    base.setAttribute("stroke", theme.baseline);
    base.setAttribute("stroke-width", "1");
    svg.appendChild(base);

    var barWidth = Math.min(24, (groupWidth - (barBand - 1) * 2) / barBand);

    ranked.forEach(function (entry, i) {
      var groupX = marginLeft + i * (groupWidth + groupGap);
      var usedWidth = barWidth * visibleFamilies.length + 2 * (visibleFamilies.length - 1);
      var startX = groupX + (groupWidth - usedWidth) / 2;

      visibleFamilies.forEach(function (family, j) {
        var rec = entry.families[family];
        if (!rec) return;
        var x = startX + j * (barWidth + 2);
        var y0 = yScale(0), y1 = yScale(rec.value);
        var top = Math.min(y0, y1), h = Math.max(1, Math.abs(y1 - y0));
        var radius = Math.min(4, h);

        var bar = document.createElementNS(SVGNS, "path");
        bar.setAttribute("d", roundedBarPath(x, top, barWidth, h, radius, rec.value >= 0));
        bar.setAttribute("class", "bar");
        bar.setAttribute("fill", theme.families[family]);
        bar.setAttribute("tabindex", "0");
        bar.setAttribute("role", "img");
        bar.setAttribute("aria-label", entry.embedding + ", " + familyLabel[family] + ": " + formatValue(rec.value));
        svg.appendChild(bar);

        attachTooltip(bar, opts.tooltip, opts.chartWrap, {
          embedding: entry.embedding,
          familyLabel: familyLabel[family],
          color: theme.families[family],
          value: rec.value,
          n: rec.n,
        });
      });

      // direct label on the leading (best-ranked) group only — see
      // marks-and-anatomy: label selectively, never every bar.
      if (i === 0) {
        var bestFamily = visibleFamilies
          .filter(function (f) {
            return entry.families[f];
          })
          .sort(function (a, b) {
            return meta.higher_is_better
              ? entry.families[b].value - entry.families[a].value
              : entry.families[a].value - entry.families[b].value;
          })[0];
        if (bestFamily) {
          var rec2 = entry.families[bestFamily];
          var lx = startX + visibleFamilies.indexOf(bestFamily) * (barWidth + 2) + barWidth / 2;
          var ly = yScale(rec2.value) + (rec2.value >= 0 ? -6 : 14);
          svg.appendChild(
            textEl(
              { x: lx, y: ly, "text-anchor": "middle", "font-size": "11", "font-weight": "700", fill: theme.textPrimary },
              formatValue(rec2.value)
            )
          );
        }
      }

      var lx2 = groupX + groupWidth / 2;
      var ly2 = height - marginBottom + 14;
      svg.appendChild(
        textEl(
          {
            x: lx2, y: ly2, "text-anchor": "end", "font-size": "11", fill: theme.textMuted,
            transform: "rotate(-38 " + lx2 + " " + ly2 + ")",
          },
          entry.embedding
        )
      );
    });
  }

  // ---- table view --------------------------------------------------------

  function renderTable(container, opts) {
    var ranked = opts.ranked;
    var meta = opts.meta;
    var families = opts.families;
    var familyLabel = opts.familyLabel;
    var metricLabel = opts.metricLabel;

    container.innerHTML = "";
    if (!ranked.length) return;

    var table = document.createElement("table");
    table.className = "leaderboard-table";

    var caption = document.createElement("caption");
    caption.textContent = "Full ranked values — " + metricLabel + ", test split.";
    table.appendChild(caption);

    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");
    ["Rank", "Embedding model"]
      .concat(families.map(function (f) { return familyLabel[f] || f; }))
      .forEach(function (h) {
        var th = document.createElement("th");
        th.textContent = h;
        headRow.appendChild(th);
      });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement("tbody");
    ranked.forEach(function (entry, i) {
      var row = document.createElement("tr");
      var rank = document.createElement("td");
      rank.textContent = String(i + 1);
      row.appendChild(rank);

      var name = document.createElement("td");
      name.textContent = entry.embedding;
      row.appendChild(name);

      families.forEach(function (f) {
        var td = document.createElement("td");
        td.textContent = entry.families[f] ? formatValue(entry.families[f].value) : "—";
        row.appendChild(td);
      });
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    container.appendChild(table);
  }

  // ---- export --------------------------------------------------------------
  //
  // Because renderBarChart already bakes concrete colors/fonts into every
  // element (see resolveTheme above), serializing the live <svg> node is
  // enough to get a standalone file — no separate "inline the computed
  // styles" pass is needed at export time.

  function svgMarkup(svgEl) {
    var clone = svgEl.cloneNode(true);
    clone.setAttribute("xmlns", SVGNS);
    var vb = svgEl.viewBox.baseVal;
    var w = (vb && vb.width) || svgEl.clientWidth;
    var h = (vb && vb.height) || svgEl.clientHeight;
    clone.setAttribute("width", w);
    clone.setAttribute("height", h);
    return { markup: new XMLSerializer().serializeToString(clone), width: w, height: h };
  }

  function triggerDownload(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 1000);
  }

  function downloadSvg(svgEl, filename) {
    var out = svgMarkup(svgEl);
    triggerDownload(new Blob([out.markup], { type: "image/svg+xml;charset=utf-8" }), filename);
  }

  function downloadPng(svgEl, filename, scale) {
    scale = scale || 2;
    var out = svgMarkup(svgEl);
    var url = URL.createObjectURL(new Blob([out.markup], { type: "image/svg+xml;charset=utf-8" }));
    var img = new Image();
    img.onload = function () {
      var canvas = document.createElement("canvas");
      canvas.width = out.width * scale;
      canvas.height = out.height * scale;
      var ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0, out.width, out.height);
      URL.revokeObjectURL(url);
      canvas.toBlob(function (pngBlob) {
        triggerDownload(pngBlob, filename);
      }, "image/png");
    };
    img.src = url;
  }

  global.LeaderboardCharts = {
    resolveTheme: resolveTheme,
    formatValue: formatValue,
    renderLegend: renderLegend,
    renderBarChart: renderBarChart,
    renderTable: renderTable,
    downloadSvg: downloadSvg,
    downloadPng: downloadPng,
  };
})(window);
