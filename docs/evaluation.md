# Resumable evaluation

`scripts/evaluate.py` runs existing policy commands with a scene-reset confirmation before each trial and a human outcome after it. It preserves interrupted/unreviewed trials for review on resume. Hardware is used only with `--run`; the default mode lists the queue and checks input files.

```bash
python scripts/evaluate.py --config configs/examples/yam-local.json --agent gpt --dry-run
# After configuring and checking your real machine:
python scripts/evaluate.py --machine lab-yam --agent gpt \
  --plan configs/evaluation.example.json --campaign block-test --repetitions 3 --run
python scripts/evaluate.py --machine lab-yam --agent gpt --campaign block-test --status
```

`--config` and `--machine` are alternatives. The latter loads `configs/machines/<name>.json`. Run from an activated environment with the package installed. `scripts/eval-gpt.sh` and `scripts/eval-claude.sh` select a provider and forward the remaining options; `PYTHON` can select another interpreter.

The example plan contains one text-only block task. Supply your own tasks, reset instructions, success criteria and conditions. `request` paths are relative to the repository root (absolute paths also work); media inside a request are relative to that JSON. Raw videos must be frozen to reviewed keyframes first. The script hashes both the request and referenced image bytes, records the Git commit/configuration, and rejects incompatible campaign resumption. Commit source changes before physical evaluation.

`--task` and `--condition` select subsets. `--history none` is the default and starts fresh trials. Optional `documented`/`reviewed` modes can credit explicitly supplied historical entries; historical credits are kept separate from new human success rates. No private history or paper demonstration set is bundled.

Outputs are under `var/evaluations/<campaign>/`: JSON state, CSV trials, and summary JSON. Missing usage metrics remain unknown. The foreground terminal must stay available for reset/review; Ctrl+C waits for policy cleanup. A hardware emergency stop remains independent of this software flow.

## Provider protocol check

`python scripts/check_agent.py --agent claude` (or `kimi`) checks image/context handling using synthetic images. It does call the configured provider, but never opens hardware or executes returned robot tools. Use `--output var/check-agent.json` to keep the report local.
