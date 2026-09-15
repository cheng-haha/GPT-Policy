const entry = document.querySelector("#entry");
let entryDismissed = false;

function dismissEntry() {
  if (entryDismissed) return;
  entryDismissed = true;
  entry.classList.add("hidden");
}

window.addEventListener("wheel", dismissEntry, { once: true, passive: true });
window.addEventListener("pointermove", dismissEntry, { once: true, passive: true });
window.addEventListener("touchstart", dismissEntry, { once: true, passive: true });
window.addEventListener("keydown", dismissEntry, { once: true });
entry.addEventListener("click", dismissEntry, { once: true });
window.setTimeout(dismissEntry, 2200);

const pageSections = [...document.querySelectorAll("main section[id]")];
const navLinks = [...document.querySelectorAll('.contents a')];

const sectionObserver = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entryItem) => entryItem.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  },
  { rootMargin: "-20% 0px -65%", threshold: [0, 0.2, 0.5] }
);

pageSections.forEach((section) => sectionObserver.observe(section));

let currentLanguage = "en";

const zhTranslations = new Map([
  ["In-context robot control with GPT-Policy: five context families", "GPT-Policy 机器人上下文控制：五类上下文"],
  ["A general-purpose VLM combines the task instruction, initial state, and contextual information to guide robot actions. Context includes human videos, robot videos with recorded actions, goal images, human–robot interaction, and self-interaction history.", "通用 VLM 将任务指令、初始状态与上下文信息结合起来，引导机器人动作。上下文包括人类视频、附带动作记录的机器人视频、目标图像、人机交互和自身交互历史。"],
  ["Figure 1.", "图 1。"],
  ["GPT-Policy execution harness", "GPT-Policy 执行框架"],
  ["The VLM generates tool requests from interleaved image and text inputs. The Cartesian adapter resolves targets, samples the pose path, checks inverse-kinematics residuals, times joint references, and executes the motion. Execution or rejection feedback returns to the next decision.", "VLM 根据交错排列的图像与文本输入生成工具请求。笛卡尔适配器解析目标、采样位姿路径、检查逆运动学残差、为关节参考分配时间并执行运动。执行或拒绝反馈返回下一次决策。"],
  ["Figure 3.", "图 3。"],
  ["Human demonstrations and robot executions for towel and notebook pickup", "毛巾与笔记本拾取中的人类示范和机器人执行"],
  ["Example runs for towel and notebook pickup under None and Human Video conditions. Frames progress from left to right.", "毛巾与笔记本拾取在无示范和人类视频条件下的执行示例。画面按从左至右的时间顺序排列。"],
  ["Figure 4.", "图 4。"],
  ["Robot demonstrations and executions for bottle opening and plug reinsertion", "瓶盖旋拧与插头重新插入的机器人示范和执行"],
  ["Example runs under None, Robot Video, and Robot Video + Action conditions. Gold boxes mark reference regions in the demonstrations; red marks local deviations and green marks closer matches.", "无示范、机器人视频及机器人视频加动作三种条件下的执行示例。金色框标注示范中的参考区域，红色标注局部偏差，绿色标注更接近示范的动作。"],
  ["Figure 5.", "图 5。"],
  ["Goal images, self-interaction history, and online human interaction", "目标图像、自身交互历史与在线人机交互"],
  ["Example executions for goal-image following, self-interaction history, and online human interaction, including object arrangement, exploration, tic-tac-toe, and pointed-fruit pickup.", "目标图像跟随、自身交互历史和在线人机交互的执行示例，涵盖物体摆放、探索、井字棋和所指水果拾取。"],
  ["Figure 6.", "图 6。"],
  ["Action references improve alignment with the demonstration", "动作参考改善与示范的对齐"],
  ["Selected bottle-opening runs compare Video and Video + Action. Measured supporting-gripper tilt, target-to-demonstration orientation differences, and corresponding frames show how action references guide trajectory selection. Progress is normalized separately for each run.", "选取的瓶盖旋拧试验对比视频与视频加动作条件。支撑夹爪的实测倾角、目标与示范的姿态差异以及对应画面展示了动作参考如何引导轨迹选择。进度在每次试验内分别归一化。"],
  ["Figure 7.", "图 7。"],
  ["Red-towel pickup across models and context conditions", "不同模型与上下文条件下的红毛巾拾取"],
  ["Human video context helps GPT-6 Astra complete the task with fewer unnecessary intermediate actions in these individual runs. Task progress is completion degree, not success rate; these examples do not establish a reliable model ranking.", "在这些单次试验中，人类视频上下文帮助 GPT-6 Astra 减少不必要的中间动作并完成任务。任务进度表示完成程度，而非成功率；这些示例不足以确立可靠的模型排名。"],
  ["Figure 8.", "图 8。"],

  ["Enabling robots to adapt to unfamiliar environments as readily as humans remains a moonshot goal of embodied AI. No finite collection of demonstrations can cover every task and situation a robot will encounter, making the ability to learn from context at deployment essential for generalization. Such in-context learning (ICL), however, remains largely beyond the reach of existing robotic policies. The broad agentic capabilities of commercial vision-language models (VLMs), such as GPT-6 Astra, raise a compelling question: can these models learn from demonstrations, examples, and interaction feedback, then translate that information into executable and verifiable robot behavior from a new initial state without gradient updates or persistent changes to task-specific parameters?", "让机器人像人类一样适应陌生环境，仍是具身人工智能的长远目标。有限的示范无法覆盖机器人将遇到的所有任务与情境，因此在部署时从上下文中学习是实现泛化的关键。然而，现有机器人策略在很大程度上仍缺乏这种上下文学习（ICL）能力。GPT-6 Astra 等商用视觉语言模型（VLM）的通用智能体能力引出了一个问题：这些模型能否从示范、样例和交互反馈中学习，并在不更新梯度、不持久改变任务特定参数的前提下，从新的初始状态出发，将这些信息转化为可执行、可验证的机器人行为？"],
  [", a general-agent framework for in-context robot learning. GPT-Policy integrates a context compiler that preserves task-relevant visual transitions, a VLM that proposes robot-tool actions, and a constrained controller that verifies and executes each action and reports its outcome. We evaluate its reliability and limitations through task success and efficiency metrics, matched comparisons across models, and controlled context ablations. In real-robot trials, human video demonstrations improve task completion even without robot action labels, while aligned action references yield further gains on contact-sensitive tasks. These findings position GPT-Policy as a step toward robot adaptation through in-context learning, providing an empirical foundation for translating the general-purpose capabilities of VLMs into physical behavior and clarifying the challenges that must be overcome for reliable deployment.", "，一个面向机器人上下文学习的通用智能体框架。GPT-Policy 整合了保留任务相关视觉变化的上下文编译器、提出机器人工具动作的 VLM，以及验证、执行并报告动作结果的受约束控制器。我们通过任务成功率与效率指标、配对模型比较和受控上下文消融评估其可靠性与局限。在真实机器人试验中，即使没有机器人动作标签，人类视频示范也能改善任务完成情况；对齐的动作参考则进一步提升接触敏感任务的表现。这些发现为通过上下文学习实现机器人适应迈出了一步，为将 VLM 的通用能力转化为物理行为提供了实证基础，也明确了可靠部署仍需克服的挑战。"],
  ["How much robotic in-context learning is already accessible through general VLMs that were not trained as dedicated robot policies?", "未经专用机器人策略训练的通用视觉语言模型，已经具备多大程度的机器人上下文学习能力？"],
  ["Can a fixed general-purpose VLM learn from demonstrations, examples, and interaction feedback", "参数固定的通用 VLM 能否从示范、样例和交互反馈中学习"],
  ["Robotic ICL adapts behavior using demonstrations, examples, or interaction experience at test time, without gradient updates or persistent task-specific parameter changes. Each context family supplies information that the current instruction and observation may omit.", "机器人 ICL 利用测试时的示范、样例或交互经验调整行为，无需梯度更新或持久性的任务特定参数变更。每类上下文都能补充当前指令和观测可能遗漏的信息。"],
  ["GPT-Policy connects a general-purpose VLM to robot tools through a shared context-to-action interface. Embodiment-specific adapters translate tool requests into executable commands. Model parameters remain fixed; updated observations and execution or rejection feedback inform each subsequent decision.", "GPT-Policy 通过统一的上下文到动作接口，将通用 VLM 与机器人工具连接起来。针对不同机器人形态的适配器将工具请求转化为可执行指令。模型参数保持固定，更新后的观测以及执行或拒绝反馈用于后续决策。"],
  ["The task instruction, current state, task references, and interaction history are combined with shared instructions and tool schemas to form the VLM input. A fixed VLM selects tool requests, which constrained robot tools execute. Returned observations and execution feedback update the context for the next decision.", "任务指令、当前状态、任务参考和交互历史与统一指令及工具模式共同组成 VLM 输入。参数固定的 VLM 选择工具请求，由受约束的机器人工具执行。返回的观测和执行反馈更新上下文，用于下一次决策。"],
  ["The Cartesian adapter samples poses, checks inverse-kinematics residuals, times joint references, and returns execution or rejection feedback.", "笛卡尔适配器采样位姿、检查逆运动学残差、为关节参考分配时间，并返回执行或拒绝反馈。"],
  ["GPT-6 Astra on real robots; three trials per condition. Success requires the final scene to satisfy the task-specific geometric and semantic criteria. Decisions and time are averaged over all trials, including failures. Tic-Tac-Toe wins and draws both count as success.", "GPT-6 Astra 真实机器人实验，每种条件重复三次。只有最终场景满足任务特定的几何与语义标准才计为成功。决策次数和时间对所有试验取平均，包括失败试验。井字棋获胜与平局均计为成功。"],
  ["A little context goes a long way.", "少量上下文也能发挥重要作用。"],
  ["Human videos raise towel and notebook pickup success from 0/3 to 2/3 without robot action labels. Aligned action references further improve bottle opening and plug reinsertion, but do not consistently reduce execution costs.", "无需机器人动作标签，人类视频即可将毛巾和笔记本拾取成功次数从 0/3 提升至 2/3。对齐动作参考进一步改善瓶盖旋拧和插头重新插入，但并不总能降低执行成本。"],
  ["Good plans, tricky execution.", "规划有效，执行仍有难度。"],
  ["Context can improve task understanding, planning, and adaptation. Precise pose generation, contact execution, outcome verification, and physical safety remain separate challenges.", "上下文可以改善任务理解、规划和适应。精确位姿生成、接触执行、结果验证与物理安全仍是独立的挑战。"],
  ["Good moves take time.", "有效动作仍需要时间。"],
  ["VLM agents can generate useful actions, but decisions can be slow and expensive. Specialized VLA or world-action models may support fast low-level control while VLM agents focus on reasoning, adaptation, and replanning.", "VLM 智能体能够生成有效动作，但决策仍可能耗时且成本较高。专用 VLA 或世界动作模型可能支持快速底层控制，而 VLM 智能体负责推理、适应和重新规划。"],
  ["In individual red-towel runs, human video raises GPT-6 Astra task progress from 55% to 100%, with approximately 35.6% shorter run time and 58.9% lower estimated token usage. Fable 5.1 and Kimi K3 reach 30% and 20% progress. These examples do not establish a reliable model ranking: task progress is distinct from success rate, which is 2/3 for GPT-6 Astra with human video.", "在单次红毛巾任务对比中，人类视频使 GPT-6 Astra 的任务进度从 55% 提升至 100%，运行时间约缩短 35.6%，估算 token 用量约降低 58.9%。Fable 5.1 和 Kimi K3 的任务进度分别为 30% 和 20%。这些案例不足以确立可靠的模型排名：任务进度不同于成功率，GPT-6 Astra 在人类视频条件下的成功次数为 2/3。"],
  ["The evidence covers small task series under selected platforms, models, and context conditions. Incomplete ablations and unobserved pretraining limit causal and novel-skill claims. Context quality, observation–action alignment, execution latency, and human intervention remain constraints. Observed inter-arm collisions show that existing safeguards alone are insufficient for safe autonomous deployment.", "当前证据来自特定平台、模型和上下文条件下的小规模任务序列。不完整的消融和未知的预训练内容限制了因果推断与新技能主张。上下文质量、观测与动作对齐、执行延迟和人工干预仍是制约因素。观测到的双臂碰撞表明，仅靠现有保护措施不足以实现安全自主部署。"],
  ["Future directions", "未来方向"],
  ["Physical safety", "物理安全"],
  ["Check both arms together, monitor separation and contact, and interrupt unsafe commands independently of the VLM.", "联合检查双臂轨迹，监测间距与接触，并独立于 VLM 中止不安全指令。"],
  ["Contact-aware execution", "接触感知执行"],
  ["Expose grasp stability, slip, and object motion; add force-aware limits and local recovery.", "提供抓取稳定性、滑移和物体运动反馈，引入力限制与局部恢复。"],
  ["Fast control and deliberate reasoning", "快速控制与审慎推理"],
  ["Pair a deliberative agent with a fast controller for pose refinement and bimanual coordination.", "将负责审慎推理的智能体与快速控制器结合，用于位姿细化与双臂协作。"],
  ["Mobile manipulation", "移动操作"],
  ["Combine active perception, persistent spatial memory, and coordinated base–arm control.", "结合主动感知、持久空间记忆和底盘与机械臂协同控制。"],
  ["Compositional context", "组合式上下文"],
  ["Compose subskills from multiple demonstrations while retaining completed subgoals and discarding obsolete context.", "组合多个示范中的子技能，同时保留已完成子目标并丢弃失效上下文。"],
  ["Adaptation to physical dynamics", "适应物理动力学"],
  ["Use recent interactions to update predictions of friction, compliance, and object response before failures accumulate.", "利用近期交互更新对摩擦、柔顺性和物体响应的预测，在失败累积前调整动作。"],
  ["None", "无示范"],
  ["Pick Up Notebook", "拿起笔记本"],
  ["Unscrew Bottle Cap", "旋开瓶盖"],
  ["Remove and Reinsert Plug", "拔出并重新插入插头"],
  ["Robot video", "机器人视频"],
  ["Robot video + action", "机器人视频 + 动作"],

  ["MOVE OR SCROLL TO ENTER", "移动鼠标或滚动进入"],
  ["In-Context Robot Learning", "基于视觉语言模型智能体的"],
  ["with VLM Agents", "上下文机器人学习"],
  ["Anonymous authors", "匿名作者"],
  ["and turn it into executable, verifiable behavior from a new initial state?", "并在新的初始状态下将其转化为可执行、可验证的行为？"],
  ["Paper", "论文"],
  ["Code", "代码"],
  ["Contents", "内容导航"],
  ["Overview", "概览"],
  ["Forms of context", "上下文形式"],
  ["Experiments", "实验"],
  ["Demo gallery", "演示画廊"],
  ["Findings", "研究发现"],
  ["Discussion", "讨论"],
  ["Citation", "引用"],
  ["Robotic in-context learning, without parameter updates", "无需更新参数的机器人上下文学习"],
  ["Abstract", "摘要"],
  ["We introduce", "我们提出"],
  ["Central question", "核心问题"],
  ["Context", "上下文"],
  ["Five sources of task information", "五类任务信息来源"],
  ["Procedure", "操作流程"],
  ["Human video", "人类视频"],
  ["Ordered visual keyframes specify how a person carries out a task, without expert robot actions.", "按顺序排列的视觉关键帧说明人类如何完成任务，无需专家机器人动作。"],
  ["Procedural order", "操作顺序"],
  ["No robot actions", "无机器人动作"],
  ["Fine contact", "精细接触"],
  ["Robot demonstration", "机器人示范"],
  ["Aligned visual evidence and optional action references support contact-rich manipulation.", "对齐的视觉证据与可选动作参考可支持接触密集型操作。"],
  ["Embodied motion", "具身运动"],
  ["Optional actions", "可选动作"],
  ["Goal state", "目标状态"],
  ["Goal image", "目标图像"],
  ["Target image", "目标图像"],
  ["A reference image gives the final arrangement and spatial relations without prescribing a sequence.", "参考图像给出最终布局与空间关系，而不规定具体操作顺序。"],
  ["Desired end state", "期望终态"],
  ["Sequence-free", "无需动作序列"],
  ["Memory", "记忆"],
  ["Self-interaction history", "自身交互历史"],
  ["Earlier observations, actions, failures, and outcomes remain available for exploration and recovery.", "先前的观测、动作、失败和结果会被保留，用于探索与恢复。"],
  ["Closed-loop memory", "闭环记忆"],
  ["Failure recovery", "失败恢复"],
  ["Coordination", "协作"],
  ["Human interaction", "人机交互"],
  ["Pointing, turn history, and live feedback can change target selection and action timing online.", "指向手势、轮次历史和实时反馈能够在线改变目标选择与动作时机。"],
  ["Online cues", "在线线索"],
  ["Turn-aware", "轮次感知"],
  ["Method", "方法"],
  ["From context to constrained action", "从上下文到受约束动作"],
  ["Figure 2.", "图 2。"],
  ["Construct context", "构建上下文"],
  ["Combine task references, current views, state, and retained interaction history.", "整合任务参考、当前视角、状态及保留的交互历史。"],
  ["Assemble input", "组装输入"],
  ["Interleave text and images with stable instructions, constraints, and tool schemas.", "将文本与图像交错组织，并附加稳定的指令、约束和工具模式。"],
  ["Select action", "选择动作"],
  ["A fixed general VLM chooses one structured robot-tool request.", "参数固定的通用视觉语言模型选择一个结构化机器人工具请求。"],
  ["Validate and execute", "验证并执行"],
  ["One interface, different kinds of adaptation", "统一接口，多种适应方式"],
  ["context families", "上下文类别"],
  ["trials per condition", "每种条件的试验次数"],
  ["reported metrics", "报告指标"],
  ["Context condition", "上下文条件"],
  ["Task", "任务"],
  ["Success", "成功"],
  ["Failed", "失败"],
  ["Decisions", "决策次数"],
  ["Time (min)", "时间（分钟）"],
  ["Without human video", "无人类视频"],
  ["Pick Red Towel", "拿起红毛巾"],
  ["Remove Glue Cap", "拔下胶帽"],
  ["Arrange T Shape", "摆成 T 形"],
  ["Arrange Fruit", "摆放水果"],
  ["Self history", "自身交互历史"],
  ["Lemon To Pink Plate", "将柠檬放入粉色盘子"],
  ["Movable Exploration", "可移动探索"],
  ["Human-robot interaction", "人机交互"],
  ["Tic-Tac-Toe", "井字棋"],
  ["Pointed Fruit Pickup", "拿取所指水果"],
  ["Interactive demos", "交互式演示"],
  ["See each form of context in action", "查看各类上下文如何发挥作用"],
  ["Human-video and goal-image tasks place the conditioning input beside the corresponding robot rollout. Other tasks retain configuration controls. Long robot runs are accelerated as labeled; source demonstrations remain at real time.", "人类视频与目标图像任务会将条件输入和相应的机器人执行并排展示。其他任务保留配置控件。较长的机器人执行视频会按标注倍速播放；原始示范保持实时速度。"],
  ["Controlled comparison", "受控对比"],
  ["Human-video comparison", "人类视频对比"],
  ["Pick up the red towel with the right hand", "用右手拿起红毛巾"],
  ["Condition", "条件"],
  ["Human video · 8 frames", "人类视频 · 8 帧"],
  ["Elapsed", "用时"],
  ["Outcome", "结果"],
  ["Task completed", "任务完成"],
  ["Task failed", "任务失败"],
  ["Self-history comparison", "自身交互历史对比"],
  ["Find the Sprite bottle and place it in the yellow basket", "找到雪碧瓶并将其放入黄色篮子"],
  ["Interaction history", "交互历史"],
  ["Baseline comparison", "基线对比"],
  ["Text-only comparison", "纯文本对比"],
  ["Remove fruit from the plate", "将水果从盘中移走"],
  ["Text only · no demonstration", "仅文本 · 无示范"],
  ["Interrupted → failed", "中断 → 失败"],
  ["What the current evidence shows", "当前证据表明"],
  ["Context helps at different layers", "上下文在不同层面发挥作用"],
  ["Finding 01", "发现 01"],
  ["Finding 02", "发现 02"],
  ["Finding 03", "发现 03"],
  ["A promising interface, with a clear boundary", "一个前景可期但边界清晰的接口"],
  ["Cite the anonymous submission", "引用匿名投稿"],
  ["Copy citation", "复制引用"],
  ["Copied", "已复制"],
  ["Select and copy", "请选择并复制"],
  ["In-context learning for general VLM agents.", "面向通用视觉语言模型智能体的上下文学习。"],
  ["Back to top", "返回顶部"],
  ["Show previous clips", "查看上一组视频"],
  ["Show more clips", "查看更多视频"],
  ["Human demonstration", "人类示范"],
  ["Conditioning input", "条件输入"],
  ["First-person view", "第一视角"],
  ["First-person view · Real-time playback", "第一视角 · 实时播放"],
  ["Evaluation run", "评估执行"],
  ["Goal-state reference", "目标状态参考"],
  ["Run summary", "运行摘要"],
  ["Online interaction context", "在线交互上下文"],
  ["Video could not be loaded.", "视频无法加载。"],
  ["Original human video", "原始人类视频"],
  ["With human video", "有人类视频"],
  ["No human demonstration", "无人类示范"],
  ["Live interaction", "实时交互"],
  ["Online interaction", "在线交互"],
  ["Head view", "头部视角"],
  ["Real time", "实时"],
  ["Gave up", "放弃"],
  ["Pick up a red towel", "拿起红毛巾"],
  ["Imitate the demonstrated two-hand grasp and lift the red towel with the robot's right hand.", "模仿示范中的双手抓取方式，并用机器人的右手抬起红毛巾。"],
  ["Remove a glue-stick cap", "拔下胶棒帽"],
  ["Use the observed pulling procedure to separate the cap from the glue stick.", "采用观察到的拉拔流程，将胶帽与胶棒分离。"],
  ["Arrange blocks into a T", "将积木摆成 T 形"],
  ["Match the target image's T shape, including block colors, relative positions, and spacing.", "复现目标图像中的 T 形，包括积木颜色、相对位置与间距。"],
  ["Arrange four fruits", "摆放四种水果"],
  ["Reproduce the target layout using the same fruit identities, positions, and spacing.", "使用相同的水果种类、位置与间距复现目标布局。"],
  ["Find the plate and place the lemon", "找到盘子并放入柠檬"],
  ["Explore the scene, locate the pink plate, and place the lemon onto it.", "探索场景，找到粉色盘子，并将柠檬放到盘中。"],
  ["Movable exploration", "可移动探索"],
  ["Search for the Sprite bottle by changing viewpoint or moving safe obstacles, then place it in the yellow basket.", "通过改变视角或移动安全障碍物寻找雪碧瓶，再将其放入黄色篮子。"],
  ["Play tic-tac-toe", "玩井字棋"],
  ["Track the live board and human moves, obey turn-taking, and choose a legal winning or blocking move.", "跟踪实时棋盘与人类落子，遵守轮次，并选择合法的制胜或封堵落点。"],
  ["Pick the pointed fruit", "拿取所指水果"],
  ["Wait for a human gesture, then pick the indicated fruit and place it on the plate.", "等待人类手势，然后拿起所指水果并放到盘中。"],
  ["Learning from interaction history", "从交互历史中学习"],
  ["Earlier observations, actions, failures, and discoveries remain available for exploration and recovery.", "先前的观测、动作、失败和发现会被保留，用于探索与恢复。"],
  ["Coordinating through live human cues", "通过实时人类线索进行协作"],
  ["Gestures, turn history, and live feedback guide target selection and action timing.", "手势、轮次历史和实时反馈会引导目标选择与动作时机。"],
  ["Explore and localize", "探索与定位"],
  ["The top and wrist views reveal the lemon, but not the pink plate. Small 4–8 cm arm motions create parallax; <code>locate_point</code> matches stable features on the lemon—its dark spot and tip—to triangulate its position instead of treating pixels as coordinates.", "顶部与腕部视角可见柠檬，但看不到粉色盘子。机械臂小幅移动 4–8 cm 建立视差，并用 <code>locate_point</code> 匹配柠檬的黑斑与尖端等稳定特征进行三角定位，而不是直接将像素当作坐标。"],
  ["Identify the occluder", "识别遮挡物"],
  ["The top view reveals the near corner of the cloth. The same parallax procedure localizes that corner in metric coordinates.", "顶部视角显示出布料靠近机器人的一角；采用相同的视差方法，将该布角定位到米制坐标。"],
  ["Uncover the plate", "揭开盘子"],
  ["The fingers approach the cloth corner from above, descend for inspection, and then close. A 3 cm lift confirms the cloth moves with the gripper before it is pulled outward in stages until the plate appears, then released.", "夹指从上方接近布角，下降观察后闭合。抬升 3 cm 验证布料随夹爪移动，随后分段将其拉向工作区外侧，直至盘子露出，再松开布料。"],
  ["Relocalize the lemon", "重新定位柠檬"],
  ["Because moving the cloth may also move the lemon, its position is triangulated again and the previous coordinates are discarded.", "移动布料可能带动柠檬，因此放下布料后重新三角定位柠檬，并废弃先前坐标。"],
  ["Grasp and verify", "抓取并验证"],
  ["The gripper descends to the lemon’s side, closes, and lifts 3 cm. A stable lemon-to-finger relationship in the wrist view, while the background moves, confirms the grasp.", "夹爪下降至柠檬侧面高度后闭合并抬升 3 cm。腕部视角中柠檬与夹指相对位置不变、背景发生移动，由此确认抓取牢固。"],
  ["Place on the plate", "放入盘中"],
  ["The plate rim and center are triangulated. The arm routes around the right side, descends in two stages above the center, and opens the gripper to release the lemon.", "三角定位盘子外缘与中心。机械臂沿右侧绕行至盘心上方，分两段下降后打开夹爪释放柠檬。"],
  ["Retreat and validate", "退出并验收"],
  ["With the gripper open, the arm retreats 2–3 cm horizontally and lifts. Visible clearance from both fingers and support from the plate confirm completion before <code>done</code> is called.", "夹爪保持张开，水平退让 2–3 cm 并抬高。确认柠檬与两指均有间隙且由盘面承托后，再调用 <code>done</code>。"],
  ["Survey and retain the basket", "巡视并记住篮子位置"],
  ["With the yellow basket visible ahead but no confirmed bottle, the robot lowers and leans forward, then scans unexplored sectors by rotating in place. It retains the basket direction and nearby fan, cables, and furniture as landmarks instead of restarting the search.", "黄色篮子虽在前方可见，但尚未确认目标瓶。机器人降低腰部并前俯，再原地旋转扫描未观察区域；同时记住篮子方向及附近的风扇、线缆和家具，把它们作为地标，而不是重新开始搜索。"],
  ["Verify and approach the Sprite can", "核实并接近雪碧罐"],
  ["A green can appears on a white table. The robot confirms the Sprite branding and pull-tab rather than relying on color, then raises to table height, aligns the left arm, and drives to a comfortable grasping distance.", "白桌上出现一个绿色罐体。机器人通过雪碧品牌标识和拉环确认目标，而非仅凭颜色判断；随后抬升至桌面作业高度，对齐左臂并前进到舒适的抓取距离。"],
  ["Test and reject the left-hand grasp", "尝试并否定左手抓取"],
  ["The left arm approaches through checked waypoints. When a motion chunk and a vertical descent are rejected, the robot changes the command structure and waist height. It closes around the can, but a 6 cm lift leaves the can on the table, correctly rejecting the grasp.", "左臂沿分段检查的路径接近。运动片段和垂直下降先后被拒绝后，机器人调整命令结构与腰部高度。夹爪随后闭合，但抬升 6 cm 时罐体仍留在桌面，因此正确判定此次抓取失败。"],
  ["Diagnose the gripper and switch hands", "诊断夹爪并切换手臂"],
  ["The empty left gripper fails to reopen even in a standalone test. The robot withdraws it, verifies that the right gripper opens normally, shifts sideways, and repositions the can in front of the right arm.", "左夹爪在空载单独测试中仍无法重新张开。机器人将其撤回，确认右夹爪能够正常开合，再横向移动底盘，把雪碧罐重新置于右臂前方。"],
  ["Grasp and verify with the right hand", "用右手抓取并验证"],
  ["The right fingers are lowered from the can rim to its mid-upper body, then close until contact. A 6 cm lift shows the base clearing the table and the can remaining fixed in the wrist view, confirming a secure grasp before the arm retracts.", "右侧夹指从罐口高度下降至罐身中上部，随后闭合至接触。抬升 6 cm 后，罐底明显离开桌面，且罐体在腕部视角中与夹爪保持相对固定，由此确认抓牢，再将手臂收回。"],
  ["Return using remembered landmarks", "依据已记住的地标返回"],
  ["With the can held close, the robot backs away from the table, turns toward the stored basket direction, reacquires the yellow basket, and sidesteps away from a person and chair before approaching with a lowered, forward-pitched waist.", "机器人将罐体收拢后退出桌边，转向先前记住的篮子方向并重新找到黄色篮子；随后横移以远离人员和椅子，再降低腰部并前俯接近篮子。"],
  ["Use the open space in the basket", "利用篮内空位"],
  ["The basket center and left side are occupied, so the robot selects a clear region along the right inner wall and moves the can over the near rim. When a direct downward path fails kinematic checks, it lowers and pitches the waist further instead of replaying the rejected motion.", "篮子中央和左侧已有物品，因此机器人选择右侧内壁旁的空位，并让罐体越过近侧篮沿。直接下放未通过运动学检查后，机器人进一步降低腰部并前俯，没有重复被拒绝的轨迹。"],
  ["Release only after support", "确认支撑后再释放"],
  ["After the can enters the basket, its upward shift relative to the fingers indicates bottom support. The robot stops lowering, opens the right gripper, and confirms that the can remains tilted but stable among the basket contents instead of following the hand.", "罐体进入篮内后，相对夹指向上移动，表明罐底已获得支撑。机器人停止下压并打开右夹爪，确认罐体倾斜但稳定地留在篮内物品之间，没有随手移动。"],
  ["Open by taking the center", "首步占据中心"],
  ["With Green moving first, the top and wrist views confirm an empty board and a clear workspace. A 3 cm wrist lift creates parallax, and <code>locate_point</code> triangulates the first green piece. After a joint-limit detour and a failed grasp that shifts the piece about 16 mm, the robot relocalizes, regrips, verifies a 3 cm lift, routes around the center post, and releases in the center cell.", "绿方先手。顶部与腕部视角确认棋盘为空且人手已离开。腕部抬升 3 cm 建立视差，并用 <code>locate_point</code> 三角定位首枚绿子。经历一次关节限位绕行和一次使棋子偏移约 16 mm 的失败抓取后，机器人重新定位并抓取，以抬升 3 cm 验证抓牢，绕过中央支杆后将棋子释放到中心格。"],
  ["Wait for Blue’s first move", "等待蓝方首步"],
  ["Both arms remain still while the board is observed repeatedly. A blue piece appears in the lower-left cell, but the robot waits until the human hand has fully withdrawn before starting Green’s turn.", "双臂保持静止并持续观察棋盘。蓝子出现在左下格后，机器人仍等待人手完全撤离，才开始绿方回合。"],
  ["Take the upper-right corner", "占据右上角"],
  ["The upper-right cell creates two diagonal threats. An initial empty grasp prompts fresh parallax localization, revealing a 29 mm offset. During transport, the shifted board invalidates the old grid coordinates, so the right grid line is remapped. A low approach avoids another joint limit; the piece slips during release but is verified inside the intended cell.", "右上格可同时形成两条对角线威胁。首次空抓后重新进行视差定位，发现棋子偏移 29 mm。运输途中棋盘整体移动，旧网格坐标失效，因此重新标定右侧格线。低位接近避开另一处关节限位；棋子释放时虽发生滑脱，但最终确认落在目标格内。"],
  ["Wait for Blue’s second move", "等待蓝方第二步"],
  ["Five observation cycles show no new piece, and waiting is not mistaken for a terminal state. When a human begins placing Blue in the middle-left cell, the robot stays still until the hand leaves and then confirms the move.", "连续五轮观测未见新增棋子，机器人没有将等待误判为终局。人类开始把蓝子放入左中格时，机器人保持静止，待手离开后再确认落子。"],
  ["Block the left column", "封堵左列"],
  ["Blue now occupies the lower-left and middle-left cells, making the upper-left cell an immediate defensive priority. The third green piece is localized again at the center of its upward-facing diamond, grasped, carried to the board, and released in the upper-left cell. A retreat view confirms that the threat is blocked.", "蓝方已占据左下格和左中格，左上格因此成为必须立即防守的位置。机器人重新定位第三枚绿子朝上菱形面的中心，将其抓取并运至棋盘，释放到左上格。退爪后的视角确认威胁已被封堵。"],
  ["Wait for Blue’s third move", "等待蓝方第三步"],
  ["The robot holds position until Blue is detected and confirmed in the upper-middle cell.", "机器人保持静止，直至检测并确认蓝子落在上中格。"],
  ["Win in the lower-right corner", "右下角制胜"],
  ["The upper-left–center–lower-right diagonal is one move from completion. Insufficient parallax triggers a wider baseline; finger occlusion prompts a switch from the piece center to a visible top corner, followed by a close-range refinement. The piece is grasped, routed along the front, placed in the inferred lower-right center, and verified as completing the diagonal before <code>done</code> is called.", "左上—中心—右下这条对角线只差一子即可完成。视差不足时扩大观测基线；夹指遮挡棋子中心后，改用可见的顶部角点定位，再在近处精修。抓取后沿前侧运输，将棋子放到推算出的右下格中心；确认对角线连成后调用 <code>done</code>。"],
  ["Wait for the first gesture", "等待首次手势"],
  ["No pointing hand or OK gesture is visible in any of the three camera views, so both arms remain still. Although the avocado is moved outside the plate and the scene layout changes, object motion alone is not treated as an instruction.", "三路相机均未发现指向手势或 OK 手势，因此双臂保持静止。即使牛油果被移到盘外、场景布局发生变化，模型也不会仅凭物体移动将其视为指令。"],
  ["Identify the red apple", "识别红苹果"],
  ["The left-wrist and top views agree that an index finger points to the red apple. The target is locked, but the robot waits through three observation cycles because the hand remains near the fruit and gripper. Only the camera ray through a stable top-surface feature is retained; a single RGB observation is not used as a metric coordinate.", "左腕与顶部视角一致检测到食指指向红苹果。目标虽已锁定，但人手仍靠近水果与夹爪，机器人连续三轮保持等待。系统只保留穿过苹果顶部稳定特征的相机射线，不会将单张 RGB 图像直接当作米制坐标。"],
  ["Triangulate the apple", "视差定位红苹果"],
  ["After the hand withdraws, the left wrist rises about 4 cm to create parallax. The first estimate is rejected because the 4.71° parallax angle is degenerate. A further 5 cm lift widens the baseline, allowing the same surface feature to yield a valid metric position.", "人手撤离后，左腕抬升约 4 cm 建立视差。首次估计因 4.71° 的视差角退化而被拒绝；再抬升 5 cm 扩大基线后，利用同一表面特征获得有效的米制位置。"],
  ["Approach and grasp", "接近并抓取"],
  ["A vertical top-down grasp is rejected at a joint limit, so the wrist switches to a reachable 45° forward-downward pose. The gripper descends in three checked stages while keeping neighboring grapes outside the grasp region. After closing to 0.55, a 3 cm lift confirms that the apple remains fixed relative to the fingers while the background peach shifts.", "垂直俯抓姿态因关节限位被拒绝，腕部改用可达的向前下方 45° 姿态。夹爪分三段下降，每段都确认邻近葡萄未进入抓取区域。闭合至 0.55 后抬升 3 cm；苹果与夹指相对位置不变而背景桃子发生位移，由此确认抓取成功。"],
  ["Place the apple", "放置红苹果"],
  ["With the apple held at a safe height, a second viewing angle triangulates the partially visible blue plate; the plate-center ray and rim-height plane define the placement region. The robot descends from 0.065 m to 0.025 m, releases, then retreats about 3 cm and lifts to verify that the apple remains on the plate and is clear of both fingers.", "将苹果保持在安全高度后，机器人从第二个视角三角定位部分可见的蓝盘，并用盘心射线与盘缘高度平面确定放置区域。机械臂从 0.065 m 分段下降至 0.025 m 后释放，再退让约 3 cm 并抬高，确认苹果留在盘内且与两指完全分离。"],
  ["Recognize the peach", "识别桃子"],
  ["Back at the observation pose, the robot detects a finger pointing to the peach. It records the next target but remains still until the hand withdraws, retaining a camera ray through the peach’s green leaf tip.", "返回观察位后，机器人检测到食指指向桃子。系统记录下一目标，但在手撤离前保持静止，同时保留穿过桃子绿色叶尖的相机射线。"],
  ["Triangulate the peach", "视差定位桃子"],
  ["An 8 cm forward translation creates sufficient parallax to localize the leaf tip. Because the tip is not a suitable grasp center, a second match on the fruit’s central vertical texture refines the body position and finger alignment.", "向前平移 8 cm 建立充分视差并定位叶尖。由于叶尖并非合适的抓取中心，系统进一步匹配果实中央的纵向纹理，以精修主体位置和夹指对准点。"],
  ["Approach and grasp again", "接近并抓取桃子"],
  ["The robot approaches along the left in checked stages—from the pre-approach pose to roughly 6 cm, 1.4 cm, and then grasp depth above the target—while ensuring the grapes remain outside the fingers. After closing, a 3 cm lift and the same foreground-background parallax test confirm a secure grasp before retreating to a safe height.", "机器人沿左侧分段接近，从预接近位依次下降至目标上方约 6 cm、1.4 cm 和夹持深度，并持续确认葡萄未进入夹指之间。闭合后抬升 3 cm，通过相同的前景—背景视差检验确认抓牢，再退回安全高度。"],
  ["Place the peach and finish", "放置桃子并完成任务"],
  ["The existing plate-rim estimate is reused. With the apple occupying the near side, the robot selects an open region farther into the plate, descends in stages, and releases the peach. A retreat view confirms that both fruits remain on the plate and clear of the gripper; the robot returns to observation and calls <code>done</code>.", "复用已有的盘缘定位。由于苹果占据盘子近侧，机器人选择盘内较远处的空位，分段下降后释放桃子。退爪视角确认两种水果均留在盘中并与夹爪分离；机器人随后返回观察位并调用 <code>done</code>。"],
  ["Page contents", "页面内容"],
  ["Code on GitHub", "在 GitHub 查看代码"],
  ["Open method figure at full size", "全尺寸打开方法示意图"],
  ["Overview of the GPT-Policy context-to-action framework", "GPT-Policy 从上下文到动作的框架概览"],
  ["GPT-Policy pipeline", "GPT-Policy 流程"],
  ["Whiteboard human-interaction demonstration asking Who are you?", "写有“WHO ARE YOU?”的人机交互白板演示"],
  ["Target arrangement for arrange blocks into a t", "积木 T 形的目标排列"],
  ["Target arrangement for arrange four fruits", "四种水果的目标排列"]
]);

[
  [
    "Localize the bottle",
    "定位瓶子"
  ],
  [
    "The agent observes the cap from the top and left wrist views, then uses parallax to estimate its position before approaching.",
    "智能体从顶部与左腕视角观察瓶盖，通过视差估计位置后再接近。"
  ],
  [
    "Grasp and present the cap",
    "抓瓶并呈现瓶盖"
  ],
  [
    "The left hand grasps the bottle from the side, makes a small lift to check the grasp, and turns the bottle neck toward the right hand.",
    "左手从侧面抓住瓶身，小幅提起检查抓持，然后将瓶颈转向右手。"
  ],
  [
    "Align the cap grasp",
    "对准瓶盖"
  ],
  [
    "The right wrist obtains a second viewpoint, localizes the cap, and approaches along the demonstrated contact direction before closing.",
    "右腕获取另一视角并定位瓶盖，沿示范的接触方向接近后闭合夹爪。"
  ],
  [
    "Loosen, recover, and regrasp",
    "旋松、恢复与换抓"
  ],
  [
    "An early grasp slips. The agent deepens the grasp and continues in small rotations, periodically releasing and resetting the wrist while the left hand supports the bottle.",
    "首次抓持出现滑脱。智能体加深抓持并逐步小角度旋拧，定期松开、复位右腕，左手持续支撑瓶身。"
  ],
  [
    "Separate and place the cap",
    "分离并放下瓶盖"
  ],
  [
    "After the open bottle mouth and independent cap become visible, the right hand withdraws and lowers the cap onto the clear table surface.",
    "观察到开放瓶口与独立瓶盖后，右手退开，将瓶盖放到无遮挡的桌面上。"
  ],
  [
    "Return the bottle upright",
    "将瓶子直立放回"
  ],
  [
    "With the right hand clear, the left hand restores the upright orientation, lowers the bottle to the table, and opens its fingers.",
    "右手避让后，左手恢复瓶身直立姿态，降低到桌面并张开夹爪。"
  ],
  [
    "Withdraw and check",
    "撤离并检查"
  ],
  [
    "The final notes report visible finger clearance, an upright bottle, and the separated cap on the table; the left hand retreats to check unsupported standing.",
    "最终记录显示夹指与瓶身有间隙、瓶子直立、瓶盖已独立放在桌面；左手撤离，检查瓶子能否独立站稳。"
  ],
  [
    "Establish left-hand support",
    "建立左手支撑"
  ],
  [
    "The agent localizes the strip from wrist-view parallax and places the closed left fingers on a solid support area beside the socket.",
    "智能体通过腕部视角的视差定位插排，将闭合的左指放在插孔旁的实心支撑区域。"
  ],
  [
    "Approach and refine the grasp",
    "接近并修正抓持"
  ],
  [
    "The right hand approaches in the demonstrated orientation. After a shallow grasp, it reopens, shifts, and lowers to grasp the plug body from the sides.",
    "右手按示范姿态接近。首次抓持较浅，随后张开、横移并下降，从两侧抓住插头本体。"
  ],
  [
    "Adjust support and retry",
    "调整支撑并重试"
  ],
  [
    "Early lifting also moves the strip. The agent lowers the right hand to unload, strengthens left-hand support, and retries with a small upward motion.",
    "初次上提带动了插排。智能体降低右手卸载，加强左手支撑，再以小幅上提重试。"
  ],
  [
    "Withdraw the plug",
    "拔出插头"
  ],
  [
    "Once the metal prongs and a gap become visible, the right hand lifts further until the prongs are fully clear of the socket.",
    "观察到金属插脚与间隙后，右手继续上提，使全部插脚离开插孔。"
  ],
  [
    "Align and reinsert",
    "对准并重新插入"
  ],
  [
    "The right hand follows the return path, makes millimeter-scale alignment corrections, and alternates unloading with short insertion motions when contact blocks progress.",
    "右手沿返回路径接近，进行毫米级对准修正；接触受阻时先卸载，再短程推进插入。"
  ],
  [
    "Release and inspect",
    "释放并检查"
  ],
  [
    "The fingers open and retreat after the prongs are no longer visibly exposed. The notes identify a remaining gap at the plug body.",
    "插脚不再明显外露后，夹爪张开并撤离；记录中仍观察到插头本体处的剩余间隙。"
  ],
  [
    "Complete seating with short presses",
    "短程补压完成就位"
  ],
  [
    "Following the demonstration, the empty right gripper closes and makes limited presses on the insulated outer end, then unloads to inspect the fit.",
    "参照示范，空右夹爪闭合，对绝缘外端进行有限的短程补压，随后卸载检查就位情况。"
  ],
  [
    "Remove support and verify",
    "撤去支撑并验证"
  ],
  [
    "Both arms retreat. The final notes report that the plug remains in the original socket with no exposed prongs after left-hand support is removed.",
    "双臂撤离。最终记录显示撤去左手支撑后，插头仍稳定留在原插孔内，未见插脚外露。"
  ],
  [
    "Pick up a notebook",
    "拿起笔记本"
  ],
  [
    "Lift the notebook from the table using the human demonstration as context.",
    "参考人类示范，从桌面拿起笔记本。"
  ],
  [
    "Unscrew a bottle cap",
    "旋开瓶盖"
  ],
  [
    "Use the robot demonstration and aligned action references to remove the cap and return the bottle to the table.",
    "参考机器人示范及对齐动作记录，取下瓶盖并将瓶子放回桌面。"
  ],
  [
    "Bottle-opening sequence",
    "瓶盖旋拧过程"
  ],
  [
    "Remove and reinsert a plug",
    "拔出并重新插回插头"
  ],
  [
    "Support the power strip, remove the plug, and fully reinsert it into the original socket.",
    "支撑插排，拔出插头，再将其完全插回原插孔。"
  ],
  [
    "Plug-removal and reinsertion sequence",
    "插头拔出与回插过程"
  ],
  [
    "One representative robot demo per task. Videos are accelerated as labeled. Success rates and mean decisions summarize all trials; the displayed outcome refers to the selected demo.",
    "每个任务展示一个代表性机器人演示，按标注倍速播放。成功率与平均决策次数汇总所有试验；显示的执行结果对应当前演示。"
  ],
  [
    "Robot video + action",
    "机器人视频 + 动作"
  ]
].forEach(([english, chinese]) => zhTranslations.set(english, chinese));

const originalTextNodes = new WeakMap();
const originalAttributes = new WeakMap();

function normalizedText(value) {
  return value.replace(/\s+/g, " ").trim();
}

function translateText(value) {
  if (currentLanguage !== "zh-CN") return value;
  const normalized = normalizedText(value);
  if (!normalized) return value;
  if (zhTranslations.has(normalized)) return zhTranslations.get(normalized);

  let match = normalized.match(/^(\d+)\s*\/\s*(\d+) success$/);
  if (match) return `${match[1]} / ${match[2]} 成功`;
  match = normalized.match(/^(\d+(?:\.\d+)?) mean decisions · (\d+(?:\.\d+)?) min · (.+)$/);
  if (match) return `${match[1]} 次平均决策 · ${match[2]} 分钟 · ${translateText(match[3])}`;
  match = normalized.match(/^(\d+(?:\.\d+)?) s$/);
  if (match) return `${match[1]} 秒`;
  match = normalized.match(/^(\d+(?:\.\d+)?) min$/);
  if (match) return `${match[1]} 分钟`;
  match = normalized.match(/^(\d+) clips$/);
  if (match) return `${match[1]} 段视频`;
  if (normalized === "1 clip") return "1 段视频";
  match = normalized.match(/^(.+): demonstration and conditioning comparison$/);
  if (match) return `${translateText(match[1])}：示范与条件对比`;
  match = normalized.match(/^(.+): target image and robot rollout$/);
  if (match) return `${translateText(match[1])}：目标图像与机器人执行`;
  match = normalized.match(/^(.+) robot rollouts$/);
  if (match) return `${translateText(match[1])}机器人执行视频`;
  match = normalized.match(/^(.+) configurations$/);
  if (match) return `${translateText(match[1])}配置`;
  match = normalized.match(/^(GPT-6 Astra|Claude Fable 5\.1|Kimi K3) · (.+)$/);
  if (match) return `${match[1]} · ${translateText(match[2])}`;
  return normalized;
}

function translateDocumentText() {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);

  nodes.forEach((node) => {
    const parent = node.parentElement;
    if (!parent || parent.closest("#language-toggle, #bibtex, .history-summary li p, script, style")) return;
    if (!originalTextNodes.has(node)) originalTextNodes.set(node, node.textContent);
    const original = originalTextNodes.get(node);
    if (currentLanguage === "en") {
      node.textContent = original;
      return;
    }
    if (!normalizedText(original)) return;
    const leading = original.match(/^\s*/)?.[0] || "";
    const trailing = original.match(/\s*$/)?.[0] || "";
    node.textContent = `${leading}${translateText(original)}${trailing}`;
  });
}

function translateDocumentAttributes() {
  document.querySelectorAll("[aria-label], [alt], [title]").forEach((element) => {
    if (element.id === "language-toggle") return;
    if (!originalAttributes.has(element)) originalAttributes.set(element, new Map());
    const originals = originalAttributes.get(element);
    ["aria-label", "alt", "title"].forEach((attribute) => {
      if (!element.hasAttribute(attribute)) return;
      if (!originals.has(attribute)) originals.set(attribute, element.getAttribute(attribute));
      const original = originals.get(attribute);
      element.setAttribute(attribute, currentLanguage === "en" ? original : translateText(original));
    });
  });
}

function translateHistorySummaries() {
  document.querySelectorAll(".history-summary[data-summary-task]").forEach((summary) => {
    const task = demoTasks[Number(summary.dataset.summaryTask)];
    if (!task?.historySteps) return;
    summary.querySelectorAll("[data-summary-step]").forEach((item) => {
      const step = task.historySteps[Number(item.dataset.summaryStep)];
      if (!step) return;
      item.querySelector("p").innerHTML = `<strong>${currentLanguage === "en" ? step.title : translateText(step.title)}</strong>${currentLanguage === "en" ? step.text : translateText(step.text)}`;
    });
  });
}

function applyLanguage(language) {
  currentLanguage = language;
  document.documentElement.lang = language;
  document.title = language === "zh-CN" ? "基于视觉语言模型智能体的上下文机器人学习" : "In-Context Robot Learning with VLM Agents";
  const description = document.querySelector('meta[name="description"]');
  description.content = language === "zh-CN"
    ? "《基于视觉语言模型智能体的上下文机器人学习》及 GPT-Policy 项目主页。"
    : "Project page for In-Context Robot Learning with VLM Agents and GPT-Policy.";
  translateDocumentText();
  translateDocumentAttributes();
  translateHistorySummaries();
  document.querySelectorAll("video").forEach((video) => video.refreshDurationBadge?.());
  const citationButton = document.querySelector("#copy-citation");
  if (citationButton) citationButton.textContent = language === "zh-CN" ? "复制引用" : "Copy citation";

  const toggle = document.querySelector("#language-toggle");
  const label = language === "zh-CN" ? "切换到英文" : "Switch to Chinese";
  toggle.setAttribute("aria-label", label);
  toggle.title = label;
}

const demoTasks = [
  {
    family: "Human video",
    title: "Pick up a red towel",
    prompt: "Imitate the demonstrated two-hand grasp and lift the red towel with the robot's right hand.",
    reference: {
      src: "assets/videos/towel-reference.mp4?v=20260914-human-source",
      poster: "assets/images/towel-human-poster.jpg",
      label: "Original human video",
      playback: "Real time"
    },
    configs: [
      { label: "With human video", model: "GPT-6 Astra", context: "Human demonstration", success: "2 / 3", decisions: "76.7", time: "18.9 min", src: "assets/videos/towel-with-demo.mp4", poster: "assets/images/towel-with-human-poster.jpg", trial: "Success", view: "Head view", usesReference: true },
      { label: "Without human video", model: "GPT-6 Astra", context: "No human demonstration", success: "0 / 3", decisions: "96.3", time: "24.6 min", src: "assets/videos/towel-no-demo.mp4", poster: "assets/images/towel-without-human-poster.jpg", trial: "Gave up", view: "Head view", usesReference: false }
    ]
  },
  {
    family: "Human video",
    title: "Remove a glue-stick cap",
    prompt: "Use the observed pulling procedure to separate the cap from the glue stick.",
    reference: {
      src: "assets/videos/glue-reference.mp4?v=20260914-human-source",
      poster: "assets/images/glue-human-poster.jpg",
      label: "Original human video",
      playback: "Real time"
    },
    configs: [
      { label: "With human video", model: "GPT-6 Astra", context: "Human demonstration", success: "3 / 3", decisions: "65.7", time: "16.9 min", src: "assets/videos/glue-with-demo.mp4", poster: "assets/images/glue-with-human-poster.jpg", trial: "Success", view: "Head view", usesReference: true },
      { label: "Without human video", model: "GPT-6 Astra", context: "No human demonstration", success: "3 / 3", decisions: "49.3", time: "12.2 min", src: "assets/videos/glue-no-demo.mp4", poster: "assets/images/glue-without-human-poster.jpg", trial: "Success", view: "Head view", usesReference: false }
    ]
  },
  {
    family: "Goal image",
    title: "Arrange blocks into a T",
    prompt: "Match the target image's T shape, including block colors, relative positions, and spacing.",
    targetImage: "assets/images/5cubes-in-T-shape.jpg?v=20260915-latest-target",
    configs: [
      { label: "Goal image", model: "GPT-6 Astra", context: "Target image", success: "3 / 3", decisions: "66.7", time: "15.8 min", src: "assets/videos/blocks-t.mp4", poster: "assets/images/blocks-t-run-poster.jpg", trial: "Success", view: "Head view" }
    ]
  },
  {
    family: "Goal image",
    title: "Arrange four fruits",
    prompt: "Reproduce the target layout using the same fruit identities, positions, and spacing.",
    targetImage: "assets/images/go-image-4fruits.jpg",
    configs: [
      { label: "Goal image", model: "GPT-6 Astra", context: "Target image", success: "3 / 3", decisions: "49.0", time: "12.4 min", src: "assets/videos/fruit-layout.mp4", poster: "assets/images/fruit-layout-run-poster.jpg", trial: "Success", view: "Head view" }
    ]
  },
  {
    family: "Self history",
    title: "Find the plate and place the lemon",
    prompt: "Explore the scene, locate the pink plate, and place the lemon onto it.",
    historyTitle: "Self-interaction history",
    historyTimeline: [0, 3.2, 5.2, 12.4, 13.6, 16.9, 19.8],
    historySteps: [
      {
        title: "Explore and localize",
        text: "The top and wrist views reveal the lemon, but not the pink plate. Small 4–8 cm arm motions create parallax; <code>locate_point</code> matches stable features on the lemon—its dark spot and tip—to triangulate its position instead of treating pixels as coordinates."
      },
      {
        title: "Identify the occluder",
        text: "The top view reveals the near corner of the cloth. The same parallax procedure localizes that corner in metric coordinates."
      },
      {
        title: "Uncover the plate",
        text: "The fingers approach the cloth corner from above, descend for inspection, and then close. A 3 cm lift confirms the cloth moves with the gripper before it is pulled outward in stages until the plate appears, then released."
      },
      {
        title: "Relocalize the lemon",
        text: "Because moving the cloth may also move the lemon, its position is triangulated again and the previous coordinates are discarded."
      },
      {
        title: "Grasp and verify",
        text: "The gripper descends to the lemon’s side, closes, and lifts 3 cm. A stable lemon-to-finger relationship in the wrist view, while the background moves, confirms the grasp."
      },
      {
        title: "Place on the plate",
        text: "The plate rim and center are triangulated. The arm routes around the right side, descends in two stages above the center, and opens the gripper to release the lemon."
      },
      {
        title: "Retreat and validate",
        text: "With the gripper open, the arm retreats 2–3 cm horizontally and lifts. Visible clearance from both fingers and support from the plate confirm completion before <code>done</code> is called."
      }
    ],
    configs: [
      { label: "Self history", model: "GPT-6 Astra", context: "Interaction history", success: "3 / 3", decisions: "35.3", time: "8.1 min", src: "assets/videos/lemon-search.mp4", trial: "Success", view: "Head view" }
    ]
  },
  {
    family: "Self history",
    title: "Movable exploration",
    prompt: "Search for the Sprite bottle by changing viewpoint or moving safe obstacles, then place it in the yellow basket.",
    historyTitle: "Self-interaction history",
    historyTimeline: [0, 3.8, 7.4, 16.8, 20.5, 26.0, 34.5, 42.2],
    historySteps: [
      {
        title: "Survey and retain the basket",
        text: "With the yellow basket visible ahead but no confirmed bottle, the robot lowers and leans forward, then scans unexplored sectors by rotating in place. It retains the basket direction and nearby fan, cables, and furniture as landmarks instead of restarting the search."
      },
      {
        title: "Verify and approach the Sprite can",
        text: "A green can appears on a white table. The robot confirms the Sprite branding and pull-tab rather than relying on color, then raises to table height, aligns the left arm, and drives to a comfortable grasping distance."
      },
      {
        title: "Test and reject the left-hand grasp",
        text: "The left arm approaches through checked waypoints. When a motion chunk and a vertical descent are rejected, the robot changes the command structure and waist height. It closes around the can, but a 6 cm lift leaves the can on the table, correctly rejecting the grasp."
      },
      {
        title: "Diagnose the gripper and switch hands",
        text: "The empty left gripper fails to reopen even in a standalone test. The robot withdraws it, verifies that the right gripper opens normally, shifts sideways, and repositions the can in front of the right arm."
      },
      {
        title: "Grasp and verify with the right hand",
        text: "The right fingers are lowered from the can rim to its mid-upper body, then close until contact. A 6 cm lift shows the base clearing the table and the can remaining fixed in the wrist view, confirming a secure grasp before the arm retracts."
      },
      {
        title: "Return using remembered landmarks",
        text: "With the can held close, the robot backs away from the table, turns toward the stored basket direction, reacquires the yellow basket, and sidesteps away from a person and chair before approaching with a lowered, forward-pitched waist."
      },
      {
        title: "Use the open space in the basket",
        text: "The basket center and left side are occupied, so the robot selects a clear region along the right inner wall and moves the can over the near rim. When a direct downward path fails kinematic checks, it lowers and pitches the waist further instead of replaying the rejected motion."
      },
      {
        title: "Release only after support",
        text: "After the can enters the basket, its upward shift relative to the fingers indicates bottom support. The robot stops lowering, opens the right gripper, and confirms that the can remains tilted but stable among the basket contents instead of following the hand."
      }
    ],
    configs: [
      { label: "GPT-6 Astra", model: "GPT-6 Astra", context: "Interaction history", success: "3 / 3", decisions: "40.33", time: "25.53 min", src: "assets/videos/mobile-gpt6.mp4", trial: "Success", view: "Head view", speed: "30× robot run" }
    ]
  },
  {
    family: "Human interaction",
    title: "Play tic-tac-toe",
    prompt: "Track the live board and human moves, obey turn-taking, and choose a legal winning or blocking move.",
    historyTitle: "Online interaction context",
    historyTimeline: [0, 13.0, 15.0, 26.2, 29.5, 38.0, 40.8],
    historySteps: [
      {
        title: "Open by taking the center",
        text: "With Green moving first, the top and wrist views confirm an empty board and a clear workspace. A 3 cm wrist lift creates parallax, and <code>locate_point</code> triangulates the first green piece. After a joint-limit detour and a failed grasp that shifts the piece about 16 mm, the robot relocalizes, regrips, verifies a 3 cm lift, routes around the center post, and releases in the center cell."
      },
      {
        title: "Wait for Blue’s first move",
        text: "Both arms remain still while the board is observed repeatedly. A blue piece appears in the lower-left cell, but the robot waits until the human hand has fully withdrawn before starting Green’s turn."
      },
      {
        title: "Take the upper-right corner",
        text: "The upper-right cell creates two diagonal threats. An initial empty grasp prompts fresh parallax localization, revealing a 29 mm offset. During transport, the shifted board invalidates the old grid coordinates, so the right grid line is remapped. A low approach avoids another joint limit; the piece slips during release but is verified inside the intended cell."
      },
      {
        title: "Wait for Blue’s second move",
        text: "Five observation cycles show no new piece, and waiting is not mistaken for a terminal state. When a human begins placing Blue in the middle-left cell, the robot stays still until the hand leaves and then confirms the move."
      },
      {
        title: "Block the left column",
        text: "Blue now occupies the lower-left and middle-left cells, making the upper-left cell an immediate defensive priority. The third green piece is localized again at the center of its upward-facing diamond, grasped, carried to the board, and released in the upper-left cell. A retreat view confirms that the threat is blocked."
      },
      {
        title: "Wait for Blue’s third move",
        text: "The robot holds position until Blue is detected and confirmed in the upper-middle cell."
      },
      {
        title: "Win in the lower-right corner",
        text: "The upper-left–center–lower-right diagonal is one move from completion. Insufficient parallax triggers a wider baseline; finger occlusion prompts a switch from the piece center to a visible top corner, followed by a close-range refinement. The piece is grasped, routed along the front, placed in the inferred lower-right center, and verified as completing the diagonal before <code>done</code> is called."
      }
    ],
    configs: [
      { label: "Live interaction", model: "GPT-6 Astra", context: "Online interaction", success: "3 / 3", decisions: "69.7", time: "13.6 min", src: "assets/videos/tic-tac-toe.mp4", trial: "Success", view: "Head view" }
    ]
  },
  {
    family: "Human interaction",
    title: "Pick the pointed fruit",
    prompt: "Wait for a human gesture, then pick the indicated fruit and place it on the plate.",
    historyTitle: "Online interaction context",
    historyTimeline: [0, 1.9, 4.2, 7.4, 11.5, 18.3, 20.8, 22.7, 26.7],
    historySteps: [
      {
        title: "Wait for the first gesture",
        text: "No pointing hand or OK gesture is visible in any of the three camera views, so both arms remain still. Although the avocado is moved outside the plate and the scene layout changes, object motion alone is not treated as an instruction."
      },
      {
        title: "Identify the red apple",
        text: "The left-wrist and top views agree that an index finger points to the red apple. The target is locked, but the robot waits through three observation cycles because the hand remains near the fruit and gripper. Only the camera ray through a stable top-surface feature is retained; a single RGB observation is not used as a metric coordinate."
      },
      {
        title: "Triangulate the apple",
        text: "After the hand withdraws, the left wrist rises about 4 cm to create parallax. The first estimate is rejected because the 4.71° parallax angle is degenerate. A further 5 cm lift widens the baseline, allowing the same surface feature to yield a valid metric position."
      },
      {
        title: "Approach and grasp",
        text: "A vertical top-down grasp is rejected at a joint limit, so the wrist switches to a reachable 45° forward-downward pose. The gripper descends in three checked stages while keeping neighboring grapes outside the grasp region. After closing to 0.55, a 3 cm lift confirms that the apple remains fixed relative to the fingers while the background peach shifts."
      },
      {
        title: "Place the apple",
        text: "With the apple held at a safe height, a second viewing angle triangulates the partially visible blue plate; the plate-center ray and rim-height plane define the placement region. The robot descends from 0.065 m to 0.025 m, releases, then retreats about 3 cm and lifts to verify that the apple remains on the plate and is clear of both fingers."
      },
      {
        title: "Recognize the peach",
        text: "Back at the observation pose, the robot detects a finger pointing to the peach. It records the next target but remains still until the hand withdraws, retaining a camera ray through the peach’s green leaf tip."
      },
      {
        title: "Triangulate the peach",
        text: "An 8 cm forward translation creates sufficient parallax to localize the leaf tip. Because the tip is not a suitable grasp center, a second match on the fruit’s central vertical texture refines the body position and finger alignment."
      },
      {
        title: "Approach and grasp again",
        text: "The robot approaches along the left in checked stages—from the pre-approach pose to roughly 6 cm, 1.4 cm, and then grasp depth above the target—while ensuring the grapes remain outside the fingers. After closing, a 3 cm lift and the same foreground-background parallax test confirm a secure grasp before retreating to a safe height."
      },
      {
        title: "Place the peach and finish",
        text: "The existing plate-rim estimate is reused. With the apple occupying the near side, the robot selects an open region farther into the plate, descends in stages, and releases the peach. A retreat view confirms that both fruits remain on the plate and clear of the gripper; the robot returns to observation and calls <code>done</code>."
      }
    ],
    configs: [
      { label: "Live interaction", model: "GPT-6 Astra", context: "Online interaction", success: "3 / 3", decisions: "67.3", time: "15.0 min", src: "assets/videos/pointed-fruit.mp4", trial: "Success", view: "Head view" }
    ]
  }
];


// Preserve existing clips; show one representative robot run per task.
demoTasks.forEach(task => { task.configs = task.configs.slice(0, 1); });
demoTasks.splice(1, 0, {
  "family": "Human video",
  "title": "Pick up a notebook",
  "prompt": "Lift the notebook from the table using the human demonstration as context.",
  "configs": [
    {
      "label": "Human video",
      "model": "GPT-6 Astra",
      "context": "Human demonstration",
      "success": "2 / 3",
      "decisions": "66.7",
      "time": "16.1 min",
      "src": "assets/experiments/notebook-astra-human-r1.mp4",
      "poster": "assets/experiments/notebook-astra-human-r1.jpg",
      "trial": "Success",
      "view": "Head view"
    }
  ]
});
demoTasks.splice(3, 0, {
  "family": "Robot video + action",
  "title": "Unscrew a bottle cap",
  "prompt": "Use the robot demonstration and aligned action references to remove the cap and return the bottle to the table.",
  "historyTitle": "Bottle-opening sequence",
  "historyTimeline": [
    0,
    3.1,
    6.13,
    10.58,
    37.275,
    40.14,
    43.87
  ],
  "historySteps": [
    {
      "title": "Localize the bottle",
      "text": "The agent observes the cap from the top and left wrist views, then uses parallax to estimate its position before approaching."
    },
    {
      "title": "Grasp and present the cap",
      "text": "The left hand grasps the bottle from the side, makes a small lift to check the grasp, and turns the bottle neck toward the right hand."
    },
    {
      "title": "Align the cap grasp",
      "text": "The right wrist obtains a second viewpoint, localizes the cap, and approaches along the demonstrated contact direction before closing."
    },
    {
      "title": "Loosen, recover, and regrasp",
      "text": "An early grasp slips. The agent deepens the grasp and continues in small rotations, periodically releasing and resetting the wrist while the left hand supports the bottle."
    },
    {
      "title": "Separate and place the cap",
      "text": "After the open bottle mouth and independent cap become visible, the right hand withdraws and lowers the cap onto the clear table surface."
    },
    {
      "title": "Return the bottle upright",
      "text": "With the right hand clear, the left hand restores the upright orientation, lowers the bottle to the table, and opens its fingers."
    },
    {
      "title": "Withdraw and check",
      "text": "The final notes report visible finger clearance, an upright bottle, and the separated cap on the table; the left hand retreats to check unsupported standing."
    }
  ],
  "configs": [
    {
      "label": "Robot video + action",
      "model": "GPT-6 Astra",
      "context": "Robot video + action",
      "success": "3 / 3",
      "decisions": "54.7",
      "time": "17.9 min",
      "src": "assets/experiments/bottle-astra-action-r1.mp4",
      "poster": "assets/experiments/bottle-astra-action-r1.jpg",
      "trial": "Success",
      "view": "Head view"
    }
  ]
}, {
  "family": "Robot video + action",
  "title": "Remove and reinsert a plug",
  "prompt": "Support the power strip, remove the plug, and fully reinsert it into the original socket.",
  "historyTitle": "Plug-removal and reinsertion sequence",
  "historyTimeline": [
    0,
    7.43,
    13.075,
    16.865,
    17.625,
    23.915,
    25.03,
    28.135
  ],
  "historySteps": [
    {
      "title": "Establish left-hand support",
      "text": "The agent localizes the strip from wrist-view parallax and places the closed left fingers on a solid support area beside the socket."
    },
    {
      "title": "Approach and refine the grasp",
      "text": "The right hand approaches in the demonstrated orientation. After a shallow grasp, it reopens, shifts, and lowers to grasp the plug body from the sides."
    },
    {
      "title": "Adjust support and retry",
      "text": "Early lifting also moves the strip. The agent lowers the right hand to unload, strengthens left-hand support, and retries with a small upward motion."
    },
    {
      "title": "Withdraw the plug",
      "text": "Once the metal prongs and a gap become visible, the right hand lifts further until the prongs are fully clear of the socket."
    },
    {
      "title": "Align and reinsert",
      "text": "The right hand follows the return path, makes millimeter-scale alignment corrections, and alternates unloading with short insertion motions when contact blocks progress."
    },
    {
      "title": "Release and inspect",
      "text": "The fingers open and retreat after the prongs are no longer visibly exposed. The notes identify a remaining gap at the plug body."
    },
    {
      "title": "Complete seating with short presses",
      "text": "Following the demonstration, the empty right gripper closes and makes limited presses on the insulated outer end, then unloads to inspect the fit."
    },
    {
      "title": "Remove support and verify",
      "text": "Both arms retreat. The final notes report that the plug remains in the original socket with no exposed prongs after left-hand support is removed."
    }
  ],
  "configs": [
    {
      "label": "Robot video + action",
      "model": "GPT-6 Astra",
      "context": "Robot video + action",
      "success": "2 / 3",
      "decisions": "48.3",
      "time": "10.8 min",
      "src": "assets/experiments/plug-astra-action-r1.mp4",
      "poster": "assets/experiments/plug-astra-action-r1.jpg",
      "trial": "Success",
      "view": "Head view"
    }
  ]
});

const demoGrid = document.querySelector("#demo-grid");

function statusClass(success) {
  if (/^(0|1)\s*\/\s*3/.test(success)) return "low";
  if (/2\s*\/\s*3/.test(success)) return "mid";
  return "high";
}

function activateRolloutRail(group) {
  const rail = group.querySelector(".rollout-rail");
  const previous = group.querySelector(".rollout-nav.previous");
  const next = group.querySelector(".rollout-nav.next");

  function updateControls() {
    const maxScroll = Math.max(0, rail.scrollWidth - rail.clientWidth);
    previous.disabled = rail.scrollLeft <= 3;
    next.disabled = rail.scrollLeft >= maxScroll - 3;
  }

  previous.addEventListener("click", () => rail.scrollBy({ left: -rail.clientWidth * .82, behavior: "smooth" }));
  next.addEventListener("click", () => rail.scrollBy({ left: rail.clientWidth * .82, behavior: "smooth" }));
  rail.addEventListener("scroll", updateControls, { passive: true });
  if ("ResizeObserver" in window) new ResizeObserver(updateControls).observe(rail);
  window.requestAnimationFrame(updateControls);
}

function renderHumanVideoTask(task, taskIndex) {
  const group = document.createElement("article");
  group.className = "demo-rollout-group";
  group.dataset.family = task.family;
  group.dataset.clips = String(task.configs.length + 1);

  const resultClips = task.configs.map((config) => `
    <figure class="rollout-card">
      <div class="rollout-media">
        <video controls muted playsinline preload="metadata" data-playback="${config.speed || "20× robot run"}" poster="${config.poster}" src="${config.src}?v=20260914-head"></video>
        <div class="video-overlay"><span>${config.speed || "20× robot run"}</span><span>${config.view}</span></div>
      </div>
      <figcaption>
        <div class="rollout-caption-head"><h4>${config.label}</h4><span class="rollout-result ${statusClass(config.success)}">${config.success} success</span></div>
        <p>${config.model} · ${config.context}</p>
        <p class="rollout-detail">${config.decisions} mean decisions · ${config.time} · ${config.trial}</p>
      </figcaption>
    </figure>`).join("");

  group.innerHTML = `
    <header class="rollout-group-head">
      <div class="rollout-group-title">
        <p class="rollout-kicker"><span class="demo-index">${String(taskIndex + 1).padStart(2, "0")}</span>${task.family}</p>
        <h3>${task.title}</h3>
        <p>${task.prompt}</p>
      </div>
      <span class="clip-count">${task.configs.length + 1} clips</span>
    </header>
    <div class="rollout-rail-wrap">
      <button class="rollout-nav previous" type="button" aria-label="Show previous clips"><span aria-hidden="true">&#8249;</span></button>
      <div class="rollout-rail" aria-label="${task.title}: demonstration and conditioning comparison">
        <figure class="rollout-card">
          <div class="rollout-media">
            <video controls muted playsinline preload="metadata" data-playback="${task.reference.playback}" poster="${task.reference.poster}" src="${task.reference.src}"></video>
            <div class="video-overlay"><span>${task.reference.playback}</span><span>First-person view</span></div>
          </div>
          <figcaption>
            <div class="rollout-caption-head"><h4>Human demonstration</h4><span class="rollout-input">Conditioning input</span></div>
            <p>First-person view · Real-time playback</p>
          </figcaption>
        </figure>
        ${resultClips}
      </div>
      <button class="rollout-nav next" type="button" aria-label="Show more clips"><span aria-hidden="true">&#8250;</span></button>
    </div>`;

  group.querySelectorAll("video").forEach((video) => {
    video.addEventListener("error", () => video.closest(".rollout-media").classList.add("video-load-failed"));
  });
  activateRolloutRail(group);
  return group;
}

function renderGoalImageTask(task, taskIndex) {
  const config = task.configs[0];
  const group = document.createElement("article");
  group.className = "demo-rollout-group goal-image-group";
  group.dataset.family = task.family;
  group.dataset.clips = "2";

  group.innerHTML = `
    <header class="rollout-group-head">
      <div class="rollout-group-title">
        <p class="rollout-kicker"><span class="demo-index">${String(taskIndex + 1).padStart(2, "0")}</span>${task.family}</p>
        <h3>${task.title}</h3>
        <p>${task.prompt}</p>
      </div>
      <span class="clip-count">2 clips</span>
    </header>
    <div class="rollout-rail-wrap">
      <button class="rollout-nav previous" type="button" aria-label="Show previous clips"><span aria-hidden="true">&#8249;</span></button>
      <div class="rollout-rail" aria-label="${task.title}: target image and robot rollout">
        <figure class="rollout-card">
          <div class="rollout-media goal-image-media">
            <img src="${task.targetImage}" alt="Target arrangement for ${task.title.toLowerCase()}" />
          </div>
          <figcaption>
            <div class="rollout-caption-head"><h4>Target image</h4><span class="rollout-input">Conditioning input</span></div>
            <p>Goal-state reference</p>
          </figcaption>
        </figure>
        <figure class="rollout-card">
          <div class="rollout-media">
            <video controls muted playsinline preload="metadata" data-playback="${config.speed || "20× robot run"}" poster="${config.poster}" src="${config.src}?v=20260914-goal-image"></video>
            <div class="video-overlay"><span>${config.speed || "20× robot run"}</span><span>${config.view}</span></div>
          </div>
          <figcaption>
            <div class="rollout-caption-head"><h4>Evaluation run</h4><span class="rollout-result ${statusClass(config.success)}">${config.success} success</span></div>
            <p>${config.model} · ${config.context}</p>
            <p class="rollout-detail">${config.decisions} mean decisions · ${config.time} · ${config.trial}</p>
          </figcaption>
        </figure>
      </div>
      <button class="rollout-nav next" type="button" aria-label="Show more clips"><span aria-hidden="true">&#8250;</span></button>
    </div>`;

  group.querySelector("video").addEventListener("error", (event) => {
    event.currentTarget.closest(".rollout-media").classList.add("video-load-failed");
  });
  activateRolloutRail(group);
  return group;
}

const contextFamilyDetails = {
  "Self history": {
    title: "Learning from interaction history",
    description: "Earlier observations, actions, failures, and discoveries remain available for exploration and recovery."
  },
  "Human interaction": {
    title: "Coordinating through live human cues",
    description: "Gestures, turn history, and live feedback guide target selection and action timing."
  }
};

function renderContextFamilyGroup(tasks, startIndex) {
  const family = tasks[0].family;
  const details = contextFamilyDetails[family];
  const isSingleTask = tasks.length === 1;
  const historySteps = isSingleTask ? tasks[0].historySteps : null;
  const historyTitle = isSingleTask ? tasks[0].historyTitle : null;
  const historyId = `context-summary-${startIndex + 1}`;
  const group = document.createElement("article");
  group.className = `demo-rollout-group context-family-group${historySteps ? " history-enriched-group" : ""}`;
  group.dataset.family = family;
  group.dataset.clips = String(tasks.length);

  const cards = tasks.map((task, taskOffset) => {
    const tabs = task.configs.length > 1
      ? `<div class="rollout-config-tabs" role="tablist" aria-label="${task.title} configurations">
          ${task.configs.map((config, configIndex) => `
            <button type="button" role="tab" aria-selected="${configIndex === 0}" tabindex="${configIndex === 0 ? 0 : -1}" data-config="${configIndex}">${config.label}</button>`).join("")}
        </div>`
      : "";
    return `
      <figure class="rollout-card" data-context-task="${taskOffset}">
        <div class="rollout-media">
          <video class="result-video" controls muted playsinline preload="metadata"></video>
          <div class="video-overlay"><span class="speed-badge"></span><span class="view-badge"></span></div>
          <p class="video-error" hidden>Video could not be loaded.</p>
        </div>
        <figcaption aria-live="polite">
          <div class="rollout-caption-head">
            <h4>${isSingleTask ? "Evaluation run" : task.title}</h4>
            <span class="rollout-result" data-field="success"></span>
          </div>
          <p class="rollout-model" data-field="model"></p>
          ${isSingleTask ? "" : `<p class="rollout-task-copy">${task.prompt}</p>`}
          <p class="rollout-detail" data-field="detail"></p>
          ${tabs}
        </figcaption>
      </figure>`;
  }).join("");

  const historySummary = historySteps
    ? `<aside class="history-summary" data-summary-task="${startIndex}" aria-labelledby="${historyId}">
        <div class="history-summary-heading">
          <p>Run summary</p>
          <h4 id="${historyId}">${historyTitle || "Interaction history"}</h4>
        </div>
        <ol>
          ${historySteps.map((step, index) => `
            <li data-summary-step="${index}"${tasks[0].historyTimeline ? ` data-start="${tasks[0].historyTimeline[index]}"` : ""}>
              <span>${String(index + 1).padStart(2, "0")}</span>
              <p><strong>${step.title}</strong>${step.text}</p>
            </li>`).join("")}
        </ol>
      </aside>`
    : "";

  group.innerHTML = `
    <header class="rollout-group-head">
      <div class="rollout-group-title">
        <p class="rollout-kicker"><span class="demo-index">${isSingleTask ? String(startIndex + 1).padStart(2, "0") : `${String(startIndex + 1).padStart(2, "0")}&ndash;${String(startIndex + tasks.length).padStart(2, "0")}`}</span>${family}</p>
        <h3>${isSingleTask ? tasks[0].title : details.title}</h3>
        <p>${isSingleTask ? tasks[0].prompt : details.description}</p>
      </div>
      <span class="clip-count">${tasks.length} ${tasks.length === 1 ? "clip" : "clips"}</span>
    </header>
    <div class="${historySteps ? "history-enriched-layout" : ""}">
      <div class="rollout-rail-wrap">
        <button class="rollout-nav previous" type="button" aria-label="Show previous clips"><span aria-hidden="true">&#8249;</span></button>
        <div class="rollout-rail" aria-label="${family} robot rollouts">${cards}</div>
        <button class="rollout-nav next" type="button" aria-label="Show more clips"><span aria-hidden="true">&#8250;</span></button>
      </div>
      ${historySummary}
    </div>`;

  group.querySelectorAll("[data-context-task]").forEach((card, taskOffset) => {
    const task = tasks[taskOffset];
    const player = card.querySelector(".result-video");
    const buttons = [...card.querySelectorAll("[data-config]")];

    function selectConfig(index) {
      const config = task.configs[index];
      const wasPlaying = !player.paused;
      player.pause();
      player.dataset.playback = config.speed || "20× robot run";
      player.src = `${config.src}?v=20260915-one-demo`;
      if (config.poster) player.poster = config.poster;
      player.load();
      if (wasPlaying) player.play().catch(() => {});
      const result = card.querySelector('[data-field="success"]');
      result.textContent = `${config.success} success`;
      result.className = `rollout-result ${statusClass(config.success)}`;
      card.querySelector('[data-field="model"]').textContent = `${config.model} · ${config.context}`;
      card.querySelector('[data-field="detail"]').textContent = `${config.decisions} mean decisions · ${config.time} · ${config.trial}`;
      card.querySelector(".speed-badge").textContent = config.speed || "20× robot run";
      card.querySelector(".view-badge").textContent = config.view;
      buttons.forEach((button, buttonIndex) => {
        const selected = buttonIndex === index;
        button.setAttribute("aria-selected", String(selected));
        button.tabIndex = selected ? 0 : -1;
      });
    }

    buttons.forEach((button, index) => {
      button.addEventListener("click", () => selectConfig(index));
      button.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        const next = event.key === "ArrowRight"
          ? (index + 1) % buttons.length
          : (index - 1 + buttons.length) % buttons.length;
        buttons[next].focus();
        selectConfig(next);
      });
    });
    player.addEventListener("error", () => { card.querySelector(".video-error").hidden = false; });
    player.addEventListener("loadeddata", () => { card.querySelector(".video-error").hidden = true; });
    selectConfig(0);
  });

  setupHistorySummary(group);
  activateRolloutRail(group);
  return group;
}

function setupHistorySummary(group) {
  const summary = group.querySelector(".history-summary");
  const media = group.querySelector(".rollout-media");
  if (!summary || !media) return;

  const matchMediaHeight = () => {
    const height = media.getBoundingClientRect().height;
    if (height > 0) summary.style.height = `${Math.round(height)}px`;
  };
  window.requestAnimationFrame(matchMediaHeight);
  if ("ResizeObserver" in window) new ResizeObserver(matchMediaHeight).observe(media);
  else window.addEventListener("resize", matchMediaHeight, { passive: true });

  const video = media.querySelector("video");
  const list = summary.querySelector("ol");
  const items = [...summary.querySelectorAll("[data-start]")];
  if (!video || !list || !items.length) return;

  summary.classList.add("is-synced");
  let activeIndex = -1;

  function showTimelineStep(behavior = "smooth") {
    const currentTime = video.currentTime || 0;
    let nextIndex = 0;
    items.forEach((item, index) => {
      if (currentTime >= Number(item.dataset.start)) nextIndex = index;
    });
    if (nextIndex === activeIndex) return;
    activeIndex = nextIndex;

    items.forEach((item, index) => {
      const active = index === activeIndex;
      item.classList.toggle("is-active", active);
      item.classList.toggle("is-complete", index < activeIndex);
      if (active) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    });

    const activeItem = items[activeIndex];
    const top = activeItem.offsetTop - list.offsetTop - (list.clientHeight - activeItem.offsetHeight) / 2;
    list.scrollTo({ top: Math.max(0, top), behavior });
  }

  video.addEventListener("timeupdate", () => showTimelineStep("smooth"));
  video.addEventListener("seeking", () => showTimelineStep("auto"));
  video.addEventListener("loadedmetadata", () => showTimelineStep("auto"));
  video.addEventListener("ended", () => showTimelineStep("auto"));
  showTimelineStep("auto");
}

demoTasks.forEach((task, index) => {
  demoGrid.appendChild(task.family === "Goal image"
    ? renderGoalImageTask(task, index)
    : renderContextFamilyGroup([task], index));
});

function formatClipDuration(seconds) {
  return `${seconds.toFixed(1)} s`;
}

function setupDurationBadge(video) {
  const stage = video.closest(".rollout-media, .comparison-stage");
  const badge = stage?.querySelector(".video-overlay span:first-child");
  if (!badge) return;

  function updateBadge() {
    const playback = (video.dataset.playback || "Real time").replace(/\s+robot run$/i, "");
    const showDuration = !stage.classList.contains("comparison-stage");
    const duration = showDuration && Number.isFinite(video.duration) && video.duration > 0
      ? ` · ${currentLanguage === "zh-CN" ? `${video.duration.toFixed(1)} 秒` : formatClipDuration(video.duration)}`
      : "";
    badge.textContent = `${translateText(playback)}${duration}`;
  }

  video.refreshDurationBadge = updateBadge;
  video.addEventListener("loadedmetadata", updateBadge);
  video.addEventListener("durationchange", updateBadge);
  updateBadge();
}

const autoplayObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entryItem) => {
      const video = entryItem.target;
      if (entryItem.isIntersecting && entryItem.intersectionRatio >= .35) {
        video.muted = true;
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  },
  { rootMargin: "0px 0px -5%", threshold: [0, .35, .75] }
);

document.querySelectorAll("video").forEach((video) => {
  setupDurationBadge(video);
  video.muted = true;
  video.loop = true;
  autoplayObserver.observe(video);
});

const copyButton = document.querySelector("#copy-citation");
copyButton.addEventListener("click", async () => {
  const bibtex = document.querySelector("#bibtex").textContent;
  try {
    await navigator.clipboard.writeText(bibtex);
    copyButton.textContent = currentLanguage === "zh-CN" ? "已复制" : "Copied";
    window.setTimeout(() => {
      copyButton.textContent = currentLanguage === "zh-CN" ? "复制引用" : "Copy citation";
    }, 1600);
  } catch {
    copyButton.textContent = currentLanguage === "zh-CN" ? "请选择并复制" : "Select and copy";
  }
});

const languageToggle = document.querySelector("#language-toggle");
languageToggle.addEventListener("click", () => {
  applyLanguage(currentLanguage === "en" ? "zh-CN" : "en");
});

applyLanguage("en");
