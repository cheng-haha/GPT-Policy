# RoboDojo 十任务评测审计

目标：GPT-as-Policy 的 `robodojo_panel50_scope_v2`，十任务各五个案例。审计日期：2026-09-23。

## 案例与源码

- 50 个布局文件、10 个支持轨迹均按发布清单校验 SHA-256。
- 发布清单列出 169 个 RoboDojo 源码文件；164 个与当前文件逐字节一致。另 5 个是冻结当前哈希的接入改动：相机标定、观测矩阵、单个 layout 选择、EE/IK 执行反馈、同步渲染。其他源码漂移会在运行前报错。
- `organize_table` 和 `stack_blocks_by_language` 的任务说明恢复到发布源提交 `ee67a146`。任务配置、场景配置和布局目录与该提交没有差异。
- 每个案例启动新的原生 `benchmark` 进程。仿真启动 seed 为 0；`ROBODOJO_LAYOUT_ID` 将目标 ID 交给原生 `reset`，无需先跑前面的案例。
- 同一 GPU 同时只运行一个案例进程。每张卡依次完成一个任务的五个案例，空闲后领取下一任务。每个案例有独立 run ID、日志与 `_result.json`。
- `_result.json` 必须恰好包含目标 layout ID、一次完整评测和自洽的原生成功率、分数。有效任务失败计入分母；超时、进程错误或缺失结果记为未完成并停止派发新任务。续跑复核已完成结果后跳过。

## 动作与反馈

- ARX X5 使用 RoboDojo 原生 25 Hz 插值和物理步进；活动配置不启用额外的等待稳定循环。可选等待路径即使启用也有界。
- GPT-Policy 每次 EE 请求限制为 0.05 m、0.35 rad，与 GPT-as-Policy 发布的目标幅度相同。GPT-Policy 仍调用 RoboDojo/cuRobo IK；GPT-as-Policy 使用局部 DLS IK，并把目标分成更小的关节控制步。两个策略控制器仍有差异。
- 原生控制执行后尚未到位的动作标记为 `incomplete`，保留 IK 成功及实测姿态。只有连续两次对目标无明显进展才判为执行受阻。工作空间边界仍生效。
- `done` 必须经 RoboDojo 奖励检查确认；连续三次未确认且没有仿真动作时，结束为策略失败，避免无限策略循环。

## 已知边界

- 发布清单的 `policy_rng_seed=0` 指 OpenPI JAX 初始 key；GPT-Policy 的 Codex 策略不使用该 key。
- 本机 ICL 数据缺少 `classify_objects_by_language` 示例。统一十任务比较先用 `--icl-mode none`；`--require-icl` 会拒绝不完整的 ICL 条件。
- 单案例默认 3600 秒墙钟上限。模型请求耗时可能成为瓶颈；多 GPU 能并行仿真，实际吞吐要以并发 smoke 的计时为准。
