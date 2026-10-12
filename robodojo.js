(() => {
  const section = document.querySelector("#robodojo");
  const results = document.querySelector("#robodojo-results");
  if (!section || !results) return;
  const { tasks, models, heatmapModels, heatmapTasks, reference: source } = benchmarkData;
  const ours = models.find(model => model.id === "ours");
  const totalTrials = tasks.reduce((sum, task) => sum + task.trials, 0);
  const totalSuccesses = tasks.reduce((sum, task) => sum + task.successes, 0);
  const columns = heatmapModels.map(id => models.find(model => model.id === id));
  const rows = heatmapTasks.map(id => tasks.find(task => task.id === id));
  let metric = "score";
  const labels = {
    en: {
      title: "Mean Score Across Ten Tasks", successTitle: "Mean Success Rate Across Ten Tasks",
      subtitle: "RoboDojo-Sim · 10 tasks · one-shot GPT-Policy",
      score: "Mean score", success: "Success rate", ours: "Ours",
      chart: "Results by model", metric: "Chart metric", axisScore: "Mean score (0–100)",
      axisSuccess: "Success rate (%)", source: "Reference report",
      note: "GPT-Policy: ten-task mean, one demonstration per task. Baselines retain the reference report’s values and aggregation.",
      details: "GPT-Policy · Per-task results", download: "Download CSV",
      task: "Task", successes: "Successes", total: "Mean (10 tasks)",
      heatScore: "Score heatmap", heatSuccess: "Success-rate heatmap",
      heatScoreTitle: "Per-task Score (0–100)", heatSuccessTitle: "Per-task success rate (%)",
      tableCaption: "RoboDojo per-task results.", overall: "Overall",
      heatDescription: "Darker colors indicate higher scores or success rates. Baseline cells and overall values follow the reference report; GPT-Policy averages tasks equally.",
      heatRegion: "Per-task heatmap; scroll horizontally on small screens", missing: "Not reported", scale: "Color scale (0–100)",
    },
    "zh-CN": {
      title: "十项任务的平均得分", successTitle: "十项任务的平均成功率",
      subtitle: "RoboDojo-Sim · 10 项任务 · GPT-Policy 单次示范",
      score: "平均得分", success: "成功率", ours: "我们的方法",
      chart: "各模型评测结果", metric: "图表指标", axisScore: "平均得分（0–100）",
      axisSuccess: "成功率（%）", source: "参考报告",
      note: "GPT-Policy 每项任务使用一次示范，十项任务等权平均；基线数值与汇总口径沿用参考报告。",
      details: "GPT-Policy · 各任务明细", download: "下载 CSV",
      task: "任务", successes: "成功数", total: "平均（10 项任务）",
      heatScore: "Score 热力表", heatSuccess: "成功率热力表",
      heatScoreTitle: "逐任务 Score（0–100）", heatSuccessTitle: "逐任务成功率（%）",
      tableCaption: "RoboDojo 各任务结果。", overall: "总体",
      heatDescription: "颜色越深，得分或成功率越高。基线逐任务与总体数值沿用参考报告；GPT-Policy 按任务等权平均。",
      heatRegion: "逐任务热力图；小屏可横向滑动", missing: "未报告", scale: "色阶（0–100）",
    },
  };
  const format = (value, selectedMetric, fixed = false) => value == null ? "—" :
    `${fixed ? value.toFixed(2) : Number(value.toFixed(2))}${selectedMetric === "success" ? "%" : ""}`;
  const modelName = name => name.replace("π₀.₅", '<span class="benchmark-pi">π<sub>0.5</sub></span>');
  const icons = model => model.logos.map(logo => `<img src="${logo}" alt="" width="24" height="24" />`).join("");
  // Table 1 color function adapted directly from GPT-as-Policy's X9 function:
  // https://anonymous-report-421.github.io/public-website/?view=1 (2026-10-11).
  function heatStyle(value) {
    if (value == null) return "";
    const ratio = Number(value) / 100;
    const rgb = [247 - 214 * ratio, 250 - 159 * ratio, 255 - 78 * ratio].map(Math.round);
    const linear = rgb.map(v => v / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
    const luminance = linear[0] * .2126 + linear[1] * .7152 + linear[2] * .0722;
    return `background-color:rgb(${rgb.join(",")});color:${1.05 / (luminance + .05) >= 4.5 ? "#fff" : "#000"}`;
  }
  // Preserve the reference's column order, markup, legend, and metric switch.
  // GPT-Policy is the only appended column; no special cell outline or logos.
  function heatmap(t, chinese) {
    const title = metric === "score" ? t.heatScoreTitle : t.heatSuccessTitle;
    return `<div class="benchmark-heatmap rr-report" id="robodojo-heatmap" lang="${chinese ? "zh-CN" : "en"}">
      <div class="rr-chart-switch" role="group" aria-label="${t.metric}" data-control="heatmap">
        <button type="button" data-metric="score" aria-pressed="${metric === "score"}">${t.heatScore}</button>
        <button type="button" data-metric="success" aria-pressed="${metric === "success"}">${t.heatSuccess}</button>
      </div>
      <h4>${title}</h4>
      <div class="rr-heat-legend" aria-label="${t.scale}"><span>${metric === "score" ? "Score" : t.axisSuccess}</span>${[0,20,40,60,80,100].map(value => `<span style="${heatStyle(value)}">${value}</span>`).join("")}</div>
      <div class="rr-table-scroll" role="region" aria-label="${t.heatRegion}" tabindex="0">
        <table class="rr-table rr-heat-table" aria-describedby="heatmap-description">
          <caption class="benchmark-sr-only">${title}</caption>
          <thead><tr><th scope="col">${t.task}</th>${columns.map(model => `<th scope="col" data-model="${model.id}">${modelName(model.header).replace("(Ours)", `(${t.ours})`)}</th>`).join("")}</tr></thead>
          <tbody>${rows.map(task => `<tr data-task="${task.id}"><th scope="row"><button type="button" data-sim-task="${task.id}">${task[chinese ? "zh" : "en"]}</button></th>${columns.map(model => {
            const value = task.values[model.id][metric];
            return `<td data-model="${model.id}" data-value="${value}" style="${heatStyle(value)}">${format(value, metric, true)}</td>`;
          }).join("")}</tr>`).join("")}</tbody>
          <tfoot><tr><th scope="row">${t.overall}</th>${columns.map(model => `<td data-model="${model.id}" data-value="${model[metric]}" style="${heatStyle(model[metric])}">${format(model[metric], metric, true)}</td>`).join("")}</tr></tfoot>
        </table>
      </div>
      <div class="rr-prose"><p id="heatmap-description"><b>${t.tableCaption}</b> ${modelName(t.heatDescription)}</p></div>
    </div>`;
  }
  function controls(t, location) {
    return `<div class="benchmark-controls" role="group" aria-label="${t.metric}" data-control="${location}">
      <button type="button" data-metric="score" aria-pressed="${metric === "score"}">${t.score}</button>
      <button type="button" data-metric="success" aria-pressed="${metric === "success"}">${t.success}</button>
    </div>`;
  }
  function render() {
    const chinese = document.documentElement.lang === "zh-CN";
    const t = labels[chinese ? "zh-CN" : "en"];
    const detailsOpen = results.querySelector(".benchmark-details")?.open;
    const scrollLeft = results.querySelector(".rr-table-scroll")?.scrollLeft || 0;
    const ranked = models.filter(model => model[metric] != null).sort((a, b) => b[metric] - a[metric]);
    section.setAttribute("aria-label", `RoboDojo · ${metric === "score" ? t.title : t.successTitle}`);
    section.innerHTML = `
      <header class="benchmark-header">
        <a class="benchmark-brand" href="https://robodojo-benchmark.com/" target="_blank" rel="noopener noreferrer"><img src="assets/benchmarks/logos/robodojo.png" alt="RoboDojo" width="1403" height="299" /></a>
        <div class="benchmark-heading"><h2>${metric === "score" ? t.title : t.successTitle}</h2><p class="benchmark-subtitle">${t.subtitle}</p></div>
      </header>
      <div class="benchmark-toolbar">${controls(t, "bars")}</div>
      <div class="benchmark-plot" role="group" aria-label="${t.chart}">
        <ol class="benchmark-bars">${ranked.map(model => `<li class="benchmark-row ${model.kind}" data-model="${model.id}">
          <span class="benchmark-model"><span class="benchmark-icons">${icons(model)}</span><span>${modelName(model.name)}${model.kind === "ours" ? ` <span class="benchmark-ours-label">(${t.ours})</span>` : ""}</span></span>
          <span class="benchmark-track"><span class="benchmark-bar" aria-hidden="true" style="--bar-value:${model[metric]}%"></span><span class="benchmark-value" style="left:calc(${model[metric]}% + 8px)">${format(model[metric], metric)}</span></span><span aria-hidden="true"></span>
        </li>`).join("")}</ol>
        <div class="benchmark-axis" aria-hidden="true"><span></span><div>${[0, 20, 40, 60, 80, 100].map(value => `<span>${value}${metric === "success" ? "%" : ""}</span>`).join("")}</div><span></span></div>
        <p class="benchmark-axis-label">${metric === "score" ? t.axisScore : t.axisSuccess}</p>
      </div>
      <div class="benchmark-caption"><p>${t.note} <a href="${source}" target="_blank" rel="noopener noreferrer">${t.source} ↗</a></p></div>`;
    results.innerHTML = `${heatmap(t, chinese)}
      <details class="benchmark-details"${detailsOpen ? " open" : ""}>
        <summary>${t.details}<span aria-hidden="true">+</span></summary>
        <div class="benchmark-table-scroll" role="region" aria-label="${t.details}" tabindex="0">
          <table><caption>${t.subtitle}</caption><thead><tr><th scope="col">${t.task}</th><th scope="col">${t.score}<br>(0–100)</th><th scope="col">${t.successes}</th><th scope="col">${t.success}</th></tr></thead>
            <tbody>${tasks.map(task => `<tr><th scope="row">${task[chinese ? "zh" : "en"]}</th><td>${task.values.ours.score.toFixed(2)}</td><td>${task.successes}/${task.trials}</td><td>${format(task.values.ours.success, "success")}</td></tr>`).join("")}</tbody>
            <tfoot><tr><th scope="row">${t.total}</th><td>${ours.score.toFixed(2)}</td><td>${totalSuccesses}/${totalTrials}</td><td>${format(ours.success, "success")}</td></tr></tfoot>
          </table>
        </div><a class="benchmark-download" href="assets/benchmarks/robodojo-tasks.csv" download>${t.download} ↓</a>
      </details>`;
    results.querySelector(".rr-table-scroll").scrollLeft = scrollLeft;
  }
  function switchMetric(event) {
    const button = event.target.closest("button[data-metric]");
    if (!button || button.dataset.metric === metric) return;
    const location = button.closest("[data-control]").dataset.control;
    // Preserve the active control's position when the table height changes.
    const previousTop = button.getBoundingClientRect().top;
    metric = button.dataset.metric;
    render();
    const root = location === "heatmap" ? results : section;
    const active = root.querySelector(`[data-control="${location}"] button[data-metric="${metric}"]`);
    window.scrollBy({ top: active.getBoundingClientRect().top - previousTop, behavior: "instant" });
    active.focus({ preventScroll: true });
  }
  section.addEventListener("click", switchMetric);
  results.addEventListener("click", switchMetric);
  document.addEventListener("languagechange", render);
  render();
})();
