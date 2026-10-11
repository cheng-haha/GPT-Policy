# Architecture comparison

The HTML comparison in `index.html#policy-comparison` is adapted from the `rr-methods` / `rr-method` / `rr-flow` / `rr-branches` method diagram in [GPT 6 Astra as an Embodied Policy](https://anonymous-report-421.github.io/public-website/?view=1), retrieved 2026-10-11.

`policy-comparison.css` copies the source diagram's 11 relevant declarations and adds narrowly scoped integration rules for equal row heights, local font aliases, and responsive stacking. The source's plain arrows, thin borders, white nodes, and blue/green policy highlights are retained. Its React wrapper and KaTeX dependency are replaced with static semantic HTML and inline mathematical notation.

- **Left:** the source's Direct flow: live images / proprioception / instruction, execution history with tools and persistent notes, GPT 6 Astra, bimanual end-effector targets (1–5 steps), inverse kinematics, execution, and renewed observations. IK is described in the source report's method text.
- **Right:** GPT-Policy from paper Sections 3.1–3.4 and Figures 2 and 6: task references, context compilation, frozen VLM policy with live observations/instruction/history, geometric grounding, action chunking, validated execution, and returned observations/feedback. Optional robot actions are not assumed for every reference. The diagram is a conceptual overview; context compilation happens before execution, not on every feedback iteration.

The comparison is an **additional**, unnumbered figure before the existing Figure 2. Figure 2 and the context-construction Figure 3 stay unchanged. Benchmark data and other figures are untouched.
