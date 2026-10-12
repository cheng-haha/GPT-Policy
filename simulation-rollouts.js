(() => {
  const root = document.querySelector('#simulation-rollouts');
  const data = window.gptPolicySimulation;
  if (!root || !data) return;
  const base = 'assets/robodojo-rollouts/';
  const labels = {
    en: {
      title: 'Simulation rollouts', intro: 'Choose a task and layout to inspect the recorded execution.',
      task: 'Task', layout: 'Layout', standard: 'Standard', random: 'Random', success: 'Success', failure: 'Not completed',
      recorded: 'Recorded execution', initial: 'Initial observation', complete: 'Episode ended',
      decision: 'Decision', control: 'Control step', left: 'Left wrist', overhead: 'Overhead', right: 'Right wrist',
      leftEEF: 'Left EEF', rightEEF: 'Right EEF', gripper: 'Gripper', score: 'Final score', liveScore: 'Native score',
      play: 'Play', pause: 'Pause', restart: 'Restart', speed: 'Speed', seek: 'Video position',
      loading: 'Loading execution records…', error: 'Video unavailable. Select another layout or reload the page.',
      traceError: 'Execution records unavailable.', target: 'Requested targets · position (m), quaternion (xyzw), gripper',
      call: 'Tool call', previous: 'Previous tool call', next: 'Next tool call', noteTitle: 'Note', noNote: 'No recorded note.',
      noAdvance: 'No control-step advance', point: 'Located point', noPosition: 'No valid 3D position', residual: 'Residual', localization: 'Localization inputs and result',
      note: 'All 50 selected episodes are available, including unsuccessful runs. Playback uses one frame per control step at 25 fps and excludes model waiting time.',
      protocol: 'Evaluation protocol', video: 'Download video', oneShot: 'One-shot · seed',
    },
    'zh-CN': {
      title: '仿真执行视频', intro: '选择任务和场景，查看对应的执行过程。',
      task: '任务', layout: '场景', standard: '标准', random: '随机', success: '成功', failure: '未完成',
      recorded: '执行记录', initial: '初始观测', complete: '执行结束',
      decision: '决策', control: '控制步', left: '左腕视角', overhead: '俯视角', right: '右腕视角',
      leftEEF: '左臂末端', rightEEF: '右臂末端', gripper: '夹爪', score: '最终得分', liveScore: '原生得分',
      play: '播放', pause: '暂停', restart: '重播', speed: '倍速', seek: '视频进度',
      loading: '正在加载执行记录…', error: '视频暂不可用，请切换场景或刷新页面。',
      traceError: '执行记录暂不可用。', target: '指令目标 · 位置（米）、四元数（xyzw）、夹爪',
      call: '工具调用', previous: '上一次工具调用', next: '下一次工具调用', noteTitle: 'Note', noNote: '无记录的 note。',
      noAdvance: '不推进控制步', point: '定位坐标', noPosition: '未得到有效三维坐标', residual: '残差', localization: '定位输入与返回结果',
      note: '展示全部 50 条选定试验，包括未成功试验。每个控制步对应一帧，以 25 fps 播放，不包含模型等待时间。',
      protocol: '评测方案', video: '下载视频', oneShot: '单次示范 · 种子',
    },
  };
  let language = document.documentElement.lang === 'zh-CN' ? 'zh-CN' : 'en';
  const t = () => labels[language];
  const name = task => task[language === 'zh-CN' ? 'zh' : 'en'];
  const params = new URLSearchParams(location.search);
  let task = data.tasks.find(item => item.id === params.get('task')) || data.tasks[1];
  let episode = task.cases.find(item => item.layout === Number(params.get('layout') || 0)
    && item.variant === (params.get('variant') || 'standard')) || task.cases[0];
  let speed = [0.5, 1, 2, 4].includes(Number(params.get('speed'))) ? Number(params.get('speed')) : 1;
  let trace = null, traceFailed = false, selection = 0, userPaused = true, visible = false;
  let selectedCall = null, displayedCall = null;
  const tracePromises = new Map();
  const key = item => `${item.variant}:${item.layout}`;
  const time = value => `${Math.floor(value / 60)}:${Math.floor(value % 60).toString().padStart(2, '0')}`;
  const fixed = value => Number(value).toFixed(3);

  root.innerHTML = `
    <h4 id="simulation-rollouts-title" data-label="title"></h4>
    <p class="sim-intro" data-label="intro"></p>
    <div class="sim-player">
      <div class="sim-selectors">
        <label><span data-label="task"></span><select id="sim-task"></select></label>
        <label><span data-label="layout"></span><select id="sim-layout"></select></label>
        <div class="sim-result"><span id="sim-outcome"></span><strong id="sim-final-score"></strong></div>
      </div>
      <div class="sim-execution">
        <div class="sim-execution-head"><span data-label="recorded"></span><span id="sim-step"></span></div>
        <div class="sim-call-nav">
          <label for="sim-call" data-label="call"></label><select id="sim-call"></select>
          <button type="button" id="sim-prev-call">←</button><button type="button" id="sim-next-call">→</button>
          <span id="sim-call-status"></span>
        </div>
        <div class="sim-action"><strong id="sim-tool"></strong><span id="sim-live-score"></span></div>
        <p class="sim-note-text"><span data-label="noteTitle"></span><span id="sim-note-text"></span></p>
        <div class="sim-targets" id="sim-targets"></div>
        <details class="sim-action-details"><summary id="sim-details-title"></summary><pre id="sim-action-json"></pre></details>
      </div>
      <div class="sim-camera-scroll" tabindex="0">
        <div class="sim-camera-strip"><span data-label="left"></span><span data-label="overhead"></span><span data-label="right"></span></div>
        <video id="sim-video" muted playsinline preload="metadata" aria-labelledby="simulation-rollouts-title"></video>
      </div>
      <p class="sim-error" id="sim-error" role="status" hidden></p>
      <div class="sim-transport">
        <button type="button" id="sim-play"></button><button type="button" id="sim-restart" data-label="restart"></button>
        <input id="sim-seek" type="range" min="0" value="0" step="0.04" />
        <output id="sim-time"></output>
        <div class="sim-speeds" role="group">${[0.5, 1, 2, 4].map(rate => `<button type="button" data-speed="${rate}">${rate}×</button>`).join('')}</div>
      </div>
      <div class="sim-footer"><span id="sim-protocol"></span><a id="sim-download" download data-label="video"></a></div>
    </div>
    <p class="sim-note"><span data-label="note"></span> <a href="${base}EVALUATION.md" data-label="protocol"></a>.</p>`;
  const $ = selector => root.querySelector(selector);
  const video = $('#sim-video');
  const taskSelect = $('#sim-task');
  const layoutSelect = $('#sim-layout');
  const play = $('#sim-play');
  const seek = $('#sim-seek');
  const callSelect = $('#sim-call');

  function writeUrl() {
    const url = new URL(location.href);
    url.searchParams.set('task', task.id);
    url.searchParams.set('layout', episode.layout);
    url.searchParams.set('variant', episode.variant);
    url.searchParams.set('speed', speed);
    try { history.replaceState(history.state, '', url); } catch (_) { /* file URLs can restrict history */ }
  }
  function updatePlay() {
    play.textContent = video.paused ? t().play : t().pause;
    play.setAttribute('aria-label', play.textContent);
  }
  function updateSpeed() {
    video.playbackRate = speed;
    root.querySelectorAll('[data-speed]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.speed) === speed)));
  }
  function populate() {
    taskSelect.replaceChildren(...data.tasks.map(item => new Option(name(item), item.id)));
    taskSelect.value = task.id;
    layoutSelect.replaceChildren(...task.cases.map(item => new Option(
      `${t()[item.variant]} ${item.layout} · ${item.success ? t().success : t().failure}`, key(item))));
    layoutSelect.value = key(episode);
    $('#sim-outcome').textContent = episode.success ? t().success : t().failure;
    $('#sim-outcome').dataset.success = episode.success;
    $('#sim-final-score').textContent = `${t().score} ${Number(episode.score_100.toFixed(2))}/100`;
    $('#sim-protocol').textContent = `${t().oneShot} ${episode.seed} · ${t()[episode.variant]} ${episode.layout}`;
    $('#sim-download').href = base + episode.video;
    seek.max = episode.duration;
  }
  function updateTrace(mediaTime) {
    const position = typeof mediaTime === 'number' ? mediaTime : video.currentTime;
    const step = Math.min(episode.control_steps, Math.floor((position + 0.00001) * data.fps));
    const action = trace?.actions.findLast(item => item.step <= step);
    const call = selectedCall !== null ? trace?.calls[selectedCall]
      : trace?.calls.find(item => item.decision === action?.decision) || trace?.calls[0];
    displayedCall = call ? trace.calls.indexOf(call) : null;
    callSelect.value = displayedCall === null ? '' : String(displayedCall);
    $('#sim-prev-call').disabled = displayedCall === null || displayedCall === 0;
    $('#sim-next-call').disabled = displayedCall === null || displayedCall === trace.calls.length - 1;
    $('#sim-call-status').textContent = call?.controls === 0 ? t().noAdvance : '';
    const score = trace?.scores.findLast(item => item[0] <= step)?.[1];
    $('#sim-step').textContent = `${t().control} ${step} / ${episode.control_steps}${call ? ` · ${t().decision} ${call.decision} / ${episode.decisions}` : ''}`;
    $('#sim-tool').textContent = !trace ? (traceFailed ? t().traceError : t().loading) : call ? call.tool : t().initial;
    $('#sim-note-text').textContent = call?.note || (trace ? t().noNote : '…');
    $('#sim-live-score').textContent = `${t().liveScore} ${score == null ? '—' : Number(score.toFixed(2))}/100`;
    const targetAction = action?.decision === call?.decision ? action
      : trace?.actions.find(item => item.decision === call?.decision);
    const localization = call?.tool === 'locate_point';
    $('#sim-targets').replaceChildren(...(localization ? [(() => {
      const el = document.createElement('span');
      const result = call.localization;
      el.textContent = result?.metric_position_available && result.position
        ? `${t().point} (${result.position.map(fixed).join(', ')}) m`
        : t().noPosition;
      if (result?.residual_m != null) el.textContent += ` · ${t().residual} ${(result.residual_m * 1000).toFixed(2)} mm`;
      el.className = 'sim-localization';
      return el;
    })()] : ['left', 'right'].map(arm => {
      const el = document.createElement('span');
      const target = targetAction?.targets[arm];
      const xyz = target?.pose_xyzquat?.slice(0, 3).map(fixed).join(', ');
      el.textContent = `${t()[`${arm}EEF`]}  ${xyz ? `(${xyz}) m` : '—'}${target?.gripper != null ? ` · ${t().gripper} ${target.gripper}` : ''}`;
      return el;
    })));
    $('#sim-details-title').textContent = localization ? t().localization : t().target;
    $('#sim-action-json').textContent = localization
      ? JSON.stringify({points: call.points, result: call.localization}, null, 2)
      : targetAction ? JSON.stringify(targetAction.targets, null, 2) : '—';
    seek.value = position;
    seek.setAttribute('aria-valuetext', `${time(position)} / ${time(episode.duration)}`);
    $('#sim-time').textContent = `${time(position)} / ${time(episode.duration)}`;
  }
  function loadTrace(item) {
    if (window.gptPolicySimulationTraces?.[item.episode_id]) return Promise.resolve(window.gptPolicySimulationTraces[item.episode_id]);
    if (!tracePromises.has(item.episode_id)) {
      const promise = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = base + item.trace + '?v=tool-notes-20261011';
        script.onload = () => {
          script.remove();
          const loaded = window.gptPolicySimulationTraces?.[item.episode_id];
          if (loaded) resolve(loaded); else reject(new Error('Missing execution trace'));
        };
        script.onerror = () => { script.remove(); reject(new Error('Execution trace unavailable')); };
        document.head.append(script);
      }).catch(error => { tracePromises.delete(item.episode_id); throw error; });
      tracePromises.set(item.episode_id, promise);
    }
    return tracePromises.get(item.episode_id);
  }
  function selectEpisode(write = true) {
    const token = ++selection;
    trace = null;
    selectedCall = null;
    displayedCall = null;
    callSelect.replaceChildren();
    callSelect.disabled = true;
    traceFailed = false;
    video.pause();
    video.poster = base + episode.poster;
    video.src = base + episode.video;
    seek.disabled = true;
    $('#sim-error').hidden = true;
    video.load();
    populate();
    updateSpeed();
    updateTrace();
    loadTrace(episode).then(value => {
      if (token !== selection) return;
      trace = value;
      callSelect.replaceChildren(...trace.calls.map((call, index) => new Option(`${call.decision} · ${call.tool}`, String(index))));
      callSelect.disabled = false;
      updateTrace();
    }).catch(() => { if (token === selection) { traceFailed = true; updateTrace(); } });
    if (write) writeUrl();
  }
  function localize() {
    language = document.documentElement.lang === 'zh-CN' ? 'zh-CN' : 'en';
    root.querySelectorAll('[data-label]').forEach(el => { el.textContent = t()[el.dataset.label]; });
    taskSelect.setAttribute('aria-label', t().task);
    layoutSelect.setAttribute('aria-label', t().layout);
    seek.setAttribute('aria-label', t().seek);
    $('#sim-prev-call').setAttribute('aria-label', t().previous);
    $('#sim-next-call').setAttribute('aria-label', t().next);
    $('#sim-prev-call').title = t().previous;
    $('#sim-next-call').title = t().next;
    $('.sim-speeds').setAttribute('aria-label', t().speed);
    $('.sim-camera-scroll').setAttribute('aria-label', `${t().left} · ${t().overhead} · ${t().right}`);
    $('#sim-error').textContent = t().error;
    populate(); updatePlay(); updateTrace();
  }
  function begin() { selectedCall = null; video.play().catch(updatePlay); }
  function inspectCall(index) {
    if (!trace || !trace.calls[index]) return;
    userPaused = true;
    video.pause();
    selectedCall = index;
    if (video.readyState > 0) video.currentTime = trace.calls[index].step / data.fps;
    updateTrace();
  }
  callSelect.addEventListener('change', () => inspectCall(Number(callSelect.value)));
  $('#sim-prev-call').addEventListener('click', () => inspectCall(displayedCall - 1));
  $('#sim-next-call').addEventListener('click', () => inspectCall(displayedCall + 1));
  play.addEventListener('click', () => {
    userPaused = !video.paused;
    if (userPaused) video.pause(); else begin();
  });
  $('#sim-restart').addEventListener('click', () => { selectedCall = null; video.currentTime = 0; userPaused = false; updateTrace(); begin(); });
  seek.addEventListener('input', () => { selectedCall = null; if (video.readyState > 0) video.currentTime = Number(seek.value); updateTrace(); });
  root.querySelectorAll('[data-speed]').forEach(button => button.addEventListener('click', () => {
    speed = Number(button.dataset.speed); updateSpeed(); writeUrl();
  }));
  taskSelect.addEventListener('change', () => { task = data.tasks.find(item => item.id === taskSelect.value); episode = task.cases[0]; selectEpisode(); });
  layoutSelect.addEventListener('change', () => { episode = task.cases.find(item => key(item) === layoutSelect.value); selectEpisode(); });
  video.addEventListener('canplay', () => { if (!userPaused && visible) begin(); });
  video.addEventListener('loadedmetadata', () => {
    seek.disabled = false;
    if (selectedCall !== null && trace?.calls[selectedCall]) video.currentTime = trace.calls[selectedCall].step / data.fps;
  });
  ['timeupdate', 'seeking', 'seeked', 'loadedmetadata'].forEach(event => video.addEventListener(event, updateTrace));
  if (video.requestVideoFrameCallback) {
    const updateFrame = (_, metadata) => {
      updateTrace(metadata.mediaTime);
      video.requestVideoFrameCallback(updateFrame);
    };
    video.requestVideoFrameCallback(updateFrame);
  }
  ['play', 'pause', 'ended'].forEach(event => video.addEventListener(event, updatePlay));
  video.addEventListener('error', () => { $('#sim-error').hidden = false; });
  document.addEventListener('languagechange', localize);
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-sim-task]');
    if (!button) return;
    const next = data.tasks.find(item => item.id === button.dataset.simTask);
    if (!next) return;
    task = next; episode = task.cases[0]; selectEpisode();
    root.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    if (!visible) video.pause();
    else if (!userPaused && video.readyState >= 2) begin();
  }, {threshold: 0.1}).observe($('.sim-player'));
  selectEpisode(false);
  localize();
})();
