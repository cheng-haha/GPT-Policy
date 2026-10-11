(() => {
  const section = document.querySelector("#robodojo");
  if (!section) return;
  const { tasks, models, heatmapModels, reference: source } = benchmarkData;
  const ours = models.find(model => model.id === "ours");
  const totalTrials = tasks.reduce((sum, task) => sum + task.trials, 0);
  const totalSuccesses = tasks.reduce((sum, task) => sum + task.successes, 0);
  const columns = heatmapModels.map(id => models.find(model => model.id === id));
  let metric = "score";
  const labels = {
    en: {
      title: "Mean Score Across Ten Tasks", successTitle: "Mean Success Rate Across Ten Tasks",
      subtitle: "RoboDojo-Sim · 10 tasks · one-shot GPT-Policy",
      score: "Mean score", success: "Success rate", ours: "Ours",
      chart: "Results by model", metric: "Chart metric", axisScore: "Mean score (0–100)",
      axisSuccess: "Success rate (%)", source: "Reference report", paper: "Paper · Tables 2 & 5",
      note: "GPT-Policy: 63.00 mean score and 52% success. Paper results take precedence; additional official-model results come from the reference report. Baselines retain their published protocols.",
      definition: "Score measures partial completion; success rate measures completed episodes. Means weight the ten tasks equally. RoboICL success rates are not reported.",
      details: "GPT-Policy · Per-task results", download: "Download CSV",
      task: "Task", successes: "Successes", total: "Mean (10 tasks)",
      heatTitle: "Per-task comparison", tableCaption: "Table 1. Per-task score and success rate.",
      heatDescription: "Switch between metrics; darker colors indicate higher values. Official references include the reference report’s overall top four models and π₀.₅, alongside the paper’s agent baselines.",
      heatNote: "— Not reported. † RoboICL uses its zero-shot score for language-guided classification. Official-model success rates outside the paper are from the reference report. π₀.₅ uses the paper’s standard/random averaging; Direct uses a task-averaged score of 38.50.",
      heatRegion: "Per-task heatmap; scroll horizontally on small screens", missing: "Not reported", scale: "Color scale (0–100)",
    },
    "zh-CN": {
      title: "十项任务的平均得分", successTitle: "十项任务的平均成功率",
      subtitle: "RoboDojo-Sim · 10 项任务 · GPT-Policy 单次示范",
      score: "平均得分", success: "成功率", ours: "我们的方法",
      chart: "各模型评测结果", metric: "图表指标", axisScore: "平均得分（0–100）",
      axisSuccess: "成功率（%）", source: "参考报告", paper: "论文 · 表 2 与表 5",
      note: "GPT-Policy：平均得分 63.00，成功率 52%。优先采用论文数值，补充的官方模型数据来自参考报告。各基线保留其原始评测方案。",
      definition: "得分衡量部分完成程度，成功率衡量完整完成任务的比例。十项任务等权平均；RoboICL 未报告成功率。",
      details: "GPT-Policy · 各任务明细", download: "下载 CSV",
      task: "任务", successes: "成功数", total: "平均（10 项任务）",
      heatTitle: "逐任务对比", tableCaption: "表 1. 各任务得分与成功率。",
      heatDescription: "点击切换指标；颜色越深，数值越高。官方参考包括参考报告总榜前四模型和 π₀.₅，并列展示论文中的智能体基线。",
      heatNote: "— 表示未报告。† RoboICL 的语言分类沿用零样本得分。论文之外的官方模型成功率来自参考报告。π₀.₅ 按论文对标准/随机场景取平均；Direct 按任务等权平均为 38.50 分。",
      heatRegion: "逐任务热力图；小屏可横向滑动", missing: "未报告", scale: "色阶（0–100）",
    },
  };
  const format = (value, selectedMetric, fixed = false) => value == null ? "—" :
    `${fixed ? value.toFixed(2) : Number(value.toFixed(2))}${selectedMetric === "success" ? "%" : ""}`;
  const modelName = name => name.replace("π₀.₅", '<span class="benchmark-pi">π<sub>0.5</sub></span>');
  const icons = model => model.logos.map(logo => `<img src="${logo}" alt="" width="24" height="24" />`).join("");
  // Match the reference heatmap's fixed 0–100 blue scale; choose readable text.
  function heatStyle(value) {
    if (value == null) return "";
    const ratio = Math.max(0, Math.min(100, value)) / 100;
    const rgb = [247 - 214 * ratio, 250 - 159 * ratio, 255 - 78 * ratio].map(Math.round);
    const linear = rgb.map(v => v / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
    const luminance = linear[0] * .2126 + linear[1] * .7152 + linear[2] * .0722;
    return `background-color:rgb(${rgb.join(",")});color:${1.05 / (luminance + .05) >= 4.5 ? "#fff" : "#000"}`;
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
    const detailsOpen = section.querySelector(".benchmark-details")?.open;
    const scrollLeft = section.querySelector(".benchmark-heat-scroll")?.scrollLeft || 0;
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
      <div class="benchmark-caption"><p>${t.note} <a href="paper.pdf?v=20261011#page=8">${t.paper} ↗</a> · <a href="${source}" target="_blank" rel="noopener noreferrer">${t.source} ↗</a></p><p>${t.definition}</p></div>
      <div class="benchmark-heatmap" id="robodojo-heatmap">
        <div class="benchmark-heat-heading"><h3>${t.heatTitle}</h3>${controls(t, "heatmap")}</div>
        <p class="benchmark-heat-description" id="heatmap-description"><b>${t.tableCaption}</b> ${t.heatDescription}</p>
        <div class="benchmark-heat-scroll benchmark-table-scroll" role="region" aria-label="${t.heatRegion}" tabindex="0">
          <table class="benchmark-heat-table" aria-describedby="heatmap-description heatmap-note">
            <caption class="benchmark-sr-only">${t.heatTitle} · ${metric === "score" ? t.score : t.success}</caption>
            <thead><tr><th scope="col">${t.task}</th>${columns.map(model => `<th scope="col" class="${model.kind === "ours" ? "heat-ours" : ""}" title="${model.name}"><span class="heat-model-icons ${model.kind}">${icons(model)}</span><span>${modelName(model.header).replace("(Ours)", `(${t.ours})`)}</span></th>`).join("")}</tr></thead>
            <tbody>${tasks.map(task => `<tr data-task="${task.id}"><th scope="row">${task[chinese ? "zh" : "en"]}</th>${columns.map(model => {
              const value = task.values[model.id][metric];
              const footnote = model.id === "roboicl-one" && task.id === "classify_objects_by_language" && metric === "score";
              return `<td data-model="${model.id}" class="${model.kind === "ours" ? "heat-ours" : ""}${value == null ? " heat-missing" : ""}" style="${heatStyle(value)}" title="${model.name}: ${value == null ? t.missing : format(value, metric, true)}">${format(value, metric, true)}${footnote ? "<sup>†</sup>" : ""}</td>`;
            }).join("")}</tr>`).join("")}</tbody>
            <tfoot><tr><th scope="row">${t.total}</th>${columns.map(model => `<td data-model="${model.id}" class="${model.kind === "ours" ? "heat-ours" : ""}${model[metric] == null ? " heat-missing" : ""}" style="${heatStyle(model[metric])}">${format(model[metric], metric, true)}</td>`).join("")}</tr></tfoot>
          </table>
        </div>
        <div class="benchmark-heat-footer"><a class="benchmark-download" href="assets/benchmarks/robodojo-comparison.csv" download>${t.download} ↓</a><div class="benchmark-heat-legend" aria-label="${t.scale}">${[0,20,40,60,80,100].map(value => `<span style="${heatStyle(value)}">${value}${metric === "success" ? "%" : ""}</span>`).join("")}</div></div>
        <p class="benchmark-heat-note" id="heatmap-note">${t.heatNote}</p>
      </div>
      <details class="benchmark-details"${detailsOpen ? " open" : ""}>
        <summary>${t.details}<span aria-hidden="true">+</span></summary>
        <div class="benchmark-table-scroll" role="region" aria-label="${t.details}" tabindex="0">
          <table><caption>${t.subtitle}</caption><thead><tr><th scope="col">${t.task}</th><th scope="col">${t.score}<br>(0–100)</th><th scope="col">${t.successes}</th><th scope="col">${t.success}</th></tr></thead>
            <tbody>${tasks.map(task => `<tr><th scope="row">${task[chinese ? "zh" : "en"]}</th><td>${task.values.ours.score.toFixed(2)}</td><td>${task.successes}/${task.trials}</td><td>${format(task.values.ours.success, "success")}</td></tr>`).join("")}</tbody>
            <tfoot><tr><th scope="row">${t.total}</th><td>${ours.score.toFixed(2)}</td><td>${totalSuccesses}/${totalTrials}</td><td>${format(ours.success, "success")}</td></tr></tfoot>
          </table>
        </div><a class="benchmark-download" href="assets/benchmarks/robodojo-tasks.csv" download>${t.download} ↓</a>
      </details>`;
    section.querySelector(".benchmark-heat-scroll").scrollLeft = scrollLeft;
  }
  section.addEventListener("click", event => {
    const button = event.target.closest("button[data-metric]");
    if (!button || button.dataset.metric === metric) return;
    const location = button.closest("[data-control]").dataset.control;
    // The score chart has extra RoboICL rows; preserve the active control's position.
    const previousTop = button.getBoundingClientRect().top;
    metric = button.dataset.metric;
    render();
    const active = section.querySelector(`[data-control="${location}"] button[data-metric="${metric}"]`);
    window.scrollBy({ top: active.getBoundingClientRect().top - previousTop, behavior: "instant" });
    active.focus({ preventScroll: true });
  });
  document.addEventListener("languagechange", render);
  render();
})();
