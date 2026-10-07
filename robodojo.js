(() => {
  const section = document.querySelector("#robodojo");
  if (!section) return;

  // GPT-Policy values transcribed from the supplied evaluation table.
  // Each tuple: English label, Chinese label, mean score, successes.
  const tasks = [
    ["Classify objects", "物品分类", 100, 5],
    ["Classify by language", "语言分类", 100, 5],
    ["Organize the table", "整理桌面", 75, 1],
    ["Imitate a sorting sequence", "顺序模仿", 100, 5],
    ["Arrange the largest number", "排列最大数", 100, 5],
    ["Pack objects into a box", "物品装箱", 50, 2],
    ["Build a tower", "搭塔", 64, 3],
    ["Make Kong", "Make Kong", 80, 4],
    ["Fold clothes", "折衣服", 100, 5],
    ["Put bottles in a bin", "瓶子入垃圾桶", 100, 5],
  ];
  const trialsPerTask = 5;
  const totalTrials = tasks.length * trialsPerTask;
  const totals = tasks.reduce((sum, task) => {
    task.slice(2).forEach((value, index) => { sum[index] += value; });
    return sum;
  }, [0, 0]);
  const ours = {
    name: "GPT-Policy", score: totals[0] / tasks.length,
    success: totals[1] / totalTrials * 100, kind: "ours", logos: ["assets/morphi-logo.png", "assets/benchmarks/logos/openai.ico"],
  };

  // Published references, NOT reruns of our evaluation. See assets/benchmarks/README.md.
  const references = [
    { name: "π₀.₅ + GPT 6 Astra", score: 62.60, success: 48, kind: "hybrid", logos: ["pi.png", "openai.ico"] },
    { name: "Galaxea G0.5", score: 38.26, success: 30.95, logos: ["galaxea.png"] },
    { name: "GPT 6 Astra (direct)", score: 37.81, success: 26, kind: "direct", logos: ["openai.ico"] },
    { name: "Xiaomi R1", score: 33.97, success: 25.51, logos: ["xiaomi.png"] },
    { name: "OpenWAM-α", score: 32.73, success: 23.07, logos: ["openwam.png"] },
    { name: "DM0.5", score: 31.82, success: 22.04, logos: ["dm.png"] },
    { name: "Meituan R0", score: 28.33, success: 17.88, logos: ["meituan.png"] },
    { name: "Spatial Forcing", score: 24.96, success: 17.15, logos: ["spatial.png"] },
    { name: "π₀.₅", score: 24.43, success: 15.67, logos: ["pi.png"], kind: "pi" },
    { name: "StarVLA-PI", score: 22.36, success: 16.17, logos: ["starvla.png"] },
    { name: "InternVLA A1.5", score: 21.74, success: 15.03, logos: ["internvla.png"] },
    { name: "Hy-VLA 0.5", score: 21.16, success: 13.94, logos: ["hunyuan.svg"] },
  ];
  const source = "https://anonymous-report-421.github.io/public-website/?view=1";
  let metric = "score";

  const labels = {
    en: {
      title: "Mean Score Across Ten Tasks", successTitle: "Mean Success Rate Across Ten Tasks",
      subtitle: "RoboDojo-Sim · 10 tasks",
      score: "Mean score", success: "Success rate", ours: "Ours",
      chart: "Results by model", metric: "Chart metric", axisScore: "Mean score (0–100)",
      axisSuccess: "Success rate (%)", source: "Published reference results",
      note: "GPT-Policy results are from our evaluation. Other methods are reported values from the reference study’s ten-task subset; evaluation settings may differ.",
      definition: "Score measures partial completion; success rate measures completed episodes.",
      details: "GPT-Policy · Per-task results", download: "Download CSV",
      task: "Task", successes: "Successes", total: "Overall",
    },
    "zh-CN": {
      title: "十项任务的平均得分", successTitle: "十项任务的平均成功率",
      subtitle: "RoboDojo-Sim · 10 项任务",
      score: "平均得分", success: "成功率", ours: "我们的方法",
      chart: "各模型评测结果", metric: "图表指标", axisScore: "平均得分（0–100）",
      axisSuccess: "成功率（%）", source: "公开报告中的参考结果",
      note: "GPT-Policy 数据来自我们的评测。其他方法的数据引自参考报告的十任务子集，评测设置可能不同。",
      definition: "得分衡量部分完成程度；成功率衡量完整完成任务的比例。",
      details: "GPT-Policy · 各任务明细", download: "下载 CSV",
      task: "任务", successes: "成功数", total: "合计 / 平均",
    },
  };
  const format = (value, selectedMetric) => selectedMetric === "score"
    ? String(Number(value.toFixed(2)))
    : `${Number(value.toFixed(2))}%`;

  function render() {
    const chinese = document.documentElement.lang === "zh-CN";
    const t = labels[chinese ? "zh-CN" : "en"];
    const detailsOpen = section.querySelector(".benchmark-details")?.open;
    const models = [ours, ...references].sort((a, b) => b[metric] - a[metric]);
    section.setAttribute("aria-label", `RoboDojo · ${metric === "score" ? t.title : t.successTitle}`);
    section.innerHTML = `
      <header class="benchmark-header">
        <a class="benchmark-brand" href="https://robodojo-benchmark.com/" target="_blank" rel="noopener noreferrer"><img src="assets/benchmarks/logos/robodojo.png" alt="RoboDojo" width="1403" height="299" /></a>
        <div class="benchmark-heading">
          <h2>${metric === "score" ? t.title : t.successTitle}</h2>
          <p class="benchmark-subtitle">${t.subtitle}</p>
        </div>
      </header>
      <div class="benchmark-toolbar">
        <div class="benchmark-controls" role="group" aria-label="${t.metric}">
          <button type="button" data-metric="score" aria-pressed="${metric === "score"}">${t.score}</button>
          <button type="button" data-metric="success" aria-pressed="${metric === "success"}">${t.success}</button>
        </div>
      </div>
      <div class="benchmark-plot" role="group" aria-label="${t.chart}">
        <ol class="benchmark-bars">
          ${models.map(model => `<li class="benchmark-row ${model.kind || "reference"}">
            <span class="benchmark-model"><span class="benchmark-icons">${model.logos.map(logo => `<img src="${model.kind === "ours" ? logo : `assets/benchmarks/logos/${logo}`}" alt="" width="24" height="24" />`).join("")}</span><span>${model.name.replace("π₀.₅", '<span class="benchmark-pi">π<sub>0.5</sub></span>')}${model.kind === "ours" ? ` <span class="benchmark-ours-label">(${t.ours})</span>` : ""}</span></span>
            <span class="benchmark-track"><span class="benchmark-bar" aria-hidden="true" style="--bar-value: ${model[metric]}%"></span><span class="benchmark-value" style="left: calc(${model[metric]}% + 8px)">${format(model[metric], metric)}</span></span>
            <span aria-hidden="true"></span>
          </li>`).join("")}
        </ol>
        <div class="benchmark-axis" aria-hidden="true"><span></span><div>${[0, 20, 40, 60, 80, 100].map(value => `<span>${value}${metric === "success" ? "%" : ""}</span>`).join("")}</div><span></span></div>
        <p class="benchmark-axis-label">${metric === "score" ? t.axisScore : t.axisSuccess}</p>
      </div>
      <div class="benchmark-caption"><p>${t.note} <a href="${source}" target="_blank" rel="noopener noreferrer">${t.source} ↗</a></p><p>${t.definition}</p></div>
      <details class="benchmark-details"${detailsOpen ? " open" : ""}>
        <summary>${t.details}<span aria-hidden="true">+</span></summary>
        <div class="benchmark-table-scroll" role="region" aria-label="${t.details}" tabindex="0">
          <table>
            <caption>${t.subtitle}</caption>
            <thead><tr><th scope="col">${t.task}</th><th scope="col">${t.score}<br>(0–100)</th><th scope="col">${t.successes}</th><th scope="col">${t.success}</th></tr></thead>
            <tbody>${tasks.map(task => `<tr><th scope="row">${task[chinese ? 1 : 0]}</th><td>${task[2].toFixed(2)}</td><td>${task[3]}/${trialsPerTask}</td><td>${format(task[3] / trialsPerTask * 100, "success")}</td></tr>`).join("")}</tbody>
            <tfoot><tr><th scope="row">${t.total}</th><td>${ours.score.toFixed(2)}</td><td>${totals[1]}/${totalTrials}</td><td>${format(ours.success, "success")}</td></tr></tfoot>
          </table>
        </div>
        <a class="benchmark-download" href="assets/benchmarks/robodojo-tasks.csv" download>${t.download} ↓</a>
      </details>`;
  }

  section.addEventListener("click", event => {
    const button = event.target.closest("button[data-metric]");
    if (!button || button.dataset.metric === metric) return;
    metric = button.dataset.metric;
    render();
    section.querySelector(`button[data-metric="${metric}"]`).focus({ preventScroll: true });
  });
  document.addEventListener("languagechange", render);
  render();
})();
