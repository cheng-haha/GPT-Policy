"""Portable native-panel evidence and accounting; standard-library verification."""
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens",
              "reasoning_output_tokens", "total_tokens")
CONFIG_KEYS = ("model", "effort", "max_decisions", "live_image_window",
               "control_step_limit", "episode_wall_time_s", "early_arrival",
               "stop_on_arrival", "ik_backend", "max_joint_increment_rad",
               "chunk_feedback", "pass_through", "distribute_closed_gripper_moves")
PRIVATE_PATH = re.compile(r"/(?:mnt|root|Users|home)/[^\s\"\\,}]+")
REFERENCE_URL = "https://anonymous-report-421.github.io/public-website/?view=1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_file(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, separators=(",", ":"), allow_nan=False)
                                + "\n" for r in rows))


def native_report(original):
    """Adapt explicit final selections, never directory globbing or best-of selection."""
    require(len(original["episodes"]) == 50, "Expected 50 explicit selected results")
    tasks = {}
    for row in original["episodes"]:
        path = Path(row["result"])
        require(digest(path) == row["result_sha256"], "Selected source hash changed")
        result = read(path)
        require(result["complete"] is True and result["valid_for_success_rate"] is True,
                "Selected episode is not native terminal")
        require(result["native_control_steps"] <= result["native_step_limit"],
                "Selected episode exceeds native horizon")
        require(result["native_process_score_100"] == row["native_score_100"],
                "Final manifest score disagrees with source")
        name = row["base_task"]
        task = tasks.setdefault(name, {"task": name, "name": row["name"], "cases": []})
        task["cases"].append({
            "source": str(path), "source_sha256": row["result_sha256"],
            "layout": row["case"]["layout_id"], "variant": row["case"]["variant"],
            "score_100": row["native_score_100"], "basis": "stored_native_process_score",
        })
    return {"tasks": list(tasks.values()),
            "overall": {"confirmed_native_successes": original["successes"],
                        "task_macro_mean_score_100": original["mean_score_100"]}}


def public_usage(call):
    """Exclude provider/account/thread IDs and raw transport metadata."""
    keys = ("call", "phase", "model", "status", "elapsed_s", "usage_final")
    return {**{k: call[k] for k in keys},
            "usage": None if call.get("usage") is None else
            {k: call["usage"].get(k) for k in TOKEN_KEYS}}


def usage_totals(calls):
    totals = {k: sum((c.get("usage") or {}).get(k) or 0 for c in calls)
              for k in TOKEN_KEYS}
    for c in calls:
        u = c.get("usage")
        if u and u.get("total_tokens") is not None:
            require(u["input_tokens"] + u["output_tokens"] == u["total_tokens"],
                    "Token total mismatch; reasoning must not be added twice")
            require(0 <= u["cached_input_tokens"] <= u["input_tokens"],
                    "Cache accounting inconsistent")
    missing = sum(not c.get("usage") or c["usage"].get("total_tokens") is None for c in calls)
    incomplete = sum(not c.get("usage_final") or not c.get("usage")
                     or c["usage"].get("total_tokens") is None for c in calls)
    return {
        **totals,
        "uncached_input_tokens": totals["input_tokens"] - totals["cached_input_tokens"],
        "model_calls": len(calls),
        "completed_model_calls": sum(c["status"] == "completed" for c in calls),
        "failed_model_calls": sum(c["status"] != "completed" for c in calls),
        "usage_missing_calls": missing,
        "usage_incomplete_calls": incomplete,
        "token_totals_complete": incomplete == 0,
    }


def public_executions(events):
    """One record per tool result, including zero-control and partial rejections."""
    rows = []
    for event in events:
        if event["event"] != "execution_result":
            continue
        result = event.get("result", {})
        controls = result.get("acknowledged_controls", 0)
        require(type(controls) is int and controls >= 0, "Invalid control ACK count")
        rows.append({"decision_index": event["step"], "tool": event["name"],
                     "acknowledged_controls": controls,
                     "start_step": result.get("start_step"),
                     "end_step": result.get("end_step")})
    return rows


def execution_totals(rows):
    motion = [r for r in rows if r["tool"] == "move_eef_chunk"]
    controls = sum(r["acknowledged_controls"] for r in rows)
    return {"decisions": len(rows), "control_steps": controls,
            "executed_action_segments": sum(r["acknowledged_controls"] > 0 for r in motion),
            "zero_control_motion_calls": sum(r["acknowledged_controls"] == 0 for r in motion),
            "non_motion_tool_calls": len(rows) - len(motion),
            "physical_duration_s": round(controls * 0.04, 6)}


def portable_context(payload):
    """Change only storage paths; preserve annotation/numeric content and image order."""
    payload = json.loads(json.dumps(payload))
    for part in payload["content"]:
        if isinstance(part, dict):
            require(set(part) <= {"image", "label", "detail"} and "image" in part,
                    "Unexpected prepared media schema")
            part["image"] = "keyframes/" + Path(part["image"]).name
    for i, part in enumerate(payload["content"]):
        if isinstance(part, str):
            # Original dataset/package locations are provenance, not runtime inputs.
            payload["content"][i] = PRIVATE_PATH.sub(
                lambda m: Path(m[0]).name, part)
    return payload


def efficiency_summary(rows):
    additive = [*TOKEN_KEYS, "uncached_input_tokens", "model_calls", "completed_model_calls",
                "failed_model_calls", "usage_missing_calls", "usage_incomplete_calls",
                "decisions", "control_steps", "executed_action_segments",
                "zero_control_motion_calls", "non_motion_tool_calls", "executed_waypoints"]
    out = {k: sum(r[k] for r in rows) for k in additive}
    out.update(episodes=len(rows), physical_duration_s=round(out["control_steps"] * .04, 6),
               mean_physical_duration_s=round(out["control_steps"] * .04 / len(rows), 6),
               token_totals_complete=all(r["token_totals_complete"] for r in rows),
               cases_with_missing_usage=sum(not r["token_totals_complete"] for r in rows),
               successes=sum(r["success"] for r in rows),
               success_rate=sum(r["success"] for r in rows) / len(rows),
               mean_score_100=sum(r["score_100"] for r in rows) / len(rows))
    return out


def enrich(root, original, episodes, source_map, contexts, source_hashes, repo):
    """Add portable ICL, prompts, audited usage and reproducer inputs to a native panel."""
    by_id = {row["episode_id"]: Path(row["source"]) for row in source_map}
    seen, efficiencies, assets, configs, selection, source_versions = {}, [], {}, {}, [], {}
    def track(path):
        source_hashes[str(path)] = digest(path)
        return source_hashes[str(path)]
    for e in episodes:
        p = by_id[e["episode_id"]]
        result, manifest = read(p / "result.json"), read(p / "manifest.json")
        folder = root / Path(e["metrics"]).parent
        e["native_step_limit"] = result["native_step_limit"]
        calls = [public_usage(c) for c in jsonl(p / "policy/usage.jsonl")]
        executions = public_executions(jsonl(p / "policy/events.jsonl"))
        waypoints = [{k: r[k] for k in ("waypoint_index", "mode", "start_step", "end_step",
                                       "requested_controls", "executed_controls", "saved_controls",
                                       "stop_reason")}
                     for r in jsonl(p / "sim/waypoint-execution.jsonl")]
        usage, execution = usage_totals(calls), execution_totals(executions)
        for k in TOKEN_KEYS:
            require(usage[k] == result["usage"]["tokens"][k], "Usage ledger/source mismatch")
        require(usage["usage_missing_calls"] == result["usage"]["usage_missing_calls"],
                "Usage coverage/source mismatch")
        require(execution["control_steps"] == e["control_steps"]
                and execution["decisions"] == e["decisions"], "Execution evidence mismatch")
        require(sum(r["executed_controls"] for r in waypoints) == e["control_steps"],
                "Waypoint ACK mismatch")
        e["usage_calls"] = (Path(e["metrics"]).parent / "usage_calls.jsonl").as_posix()
        e["execution_segments"] = (Path(e["metrics"]).parent / "execution_segments.jsonl").as_posix()
        e["waypoints"] = (Path(e["metrics"]).parent / "waypoints.jsonl").as_posix()
        write_jsonl(root / e["usage_calls"], calls)
        write_jsonl(root / e["execution_segments"], executions)
        write_jsonl(root / e["waypoints"], waypoints)
        for rel in ("policy/usage.jsonl", "policy/events.jsonl", "sim/waypoint-execution.jsonl",
                    "agent_context.json", "sim/loaded_layers.json"):
            track(p / rel)
        record = {"episode_id": e["episode_id"], "task": e["task"], "success": e["success"],
                  "score_100": e["score_100"], **execution, **usage,
                  "executed_waypoints": sum(w["executed_controls"] > 0 for w in waypoints)}
        efficiencies.append(record)
        provenance = read(folder / "provenance.json")
        provenance.update(original_usage_ledger_sha256=digest(p / "policy/usage.jsonl"),
                          original_execution_events_sha256=digest(p / "policy/events.jsonl"))
        layers = read(p / "sim/loaded_layers.json")
        asset_layers = [{"path": "Assets/" + x["path"].split("/Assets/", 1)[1],
                         "sha256": x["sha256"]}
                        for x in layers if x.get("path") and "/Assets/" in x["path"]]
        require(any(x["path"] == "Assets/Robots/x5/ARX.usd" for x in asset_layers),
                "Official X5 layer missing")
        require(not any("high_friction" in x["path"] for x in asset_layers),
                "Unexpected modified robot layer loaded")
        write(folder / "loaded_assets.json", asset_layers)
        e["loaded_assets"] = (Path(e["metrics"]).parent / "loaded_assets.json").as_posix()
        c = manifest["context"]
        ctx_path = Path(c["path"])
        require(track(ctx_path) == c["sha256"], "Prepared input changed")
        ctx_id = e["context_id"]
        agent = read(p / "agent_context.json")
        agent_sha = digest(p / "agent_context.json")
        if ctx_id in seen:
            require(seen[ctx_id]["agent_sha256"] == agent_sha, "Task prompt changed across layouts")
        else:
            rel = Path("contexts") / e["task"] / "prepared"
            out = root / rel
            (out / "keyframes").mkdir(parents=True)
            original_input = read(ctx_path)
            portable = portable_context(original_input)
            image_blocks = [x for x in portable["content"] if isinstance(x, dict)]
            require(len(image_blocks) == len(c["images"]) == 48, "Context image count mismatch")
            image_manifest = []
            for block, old in zip(image_blocks, c["images"]):
                image_source = Path(old["path"])
                require(track(image_source) == old["sha256"], "Context image bytes changed")
                target = out / block["image"]
                require(not target.exists(), "Duplicate image basename")
                shutil.copyfile(image_source, target)
                require(digest(target) == old["sha256"], "Context image copy mismatch")
                image_manifest.append({"path": (rel / block["image"]).as_posix(),
                                       "sha256": old["sha256"]})
            write(out / "input.json", portable)
            # Prove portability only changed the explicit storage path mapping.
            require(portable == portable_context(original_input), "Context conversion drift")
            instructions = Path("contexts") / e["task"] / "agent_context.json"
            write(root / instructions, agent)
            info = {"task": e["task"], "input": (rel / "input.json").as_posix(),
                    "original_input_sha256": c["sha256"],
                    "portable_input_sha256": digest(out / "input.json"),
                    "agent_context": instructions.as_posix(), "agent_sha256": agent_sha,
                    "images": image_manifest,
                    "path_only_conversion": True,
                    "conversion": "Relative image paths; dataset and original-package locations reduced to basenames. All other text, numerical references, order and image bytes unchanged."}
            seen[ctx_id] = info
            contexts[ctx_id].update(raw_demonstration_distributed=True, **info)
        provenance["portable_context_id"] = ctx_id
        write(folder / "provenance.json", provenance)
        write(folder / "result.json", e)
        config = read(p / "run_config.json")
        public_config = {k: config[k] for k in CONFIG_KEYS}
        require(public_config["control_step_limit"] == "native"
                and public_config["live_image_window"] is None, "Unexpected protocol")
        configs[e["native_task"]] = public_config
        audit = manifest["assets"]["audit"]
        if e["native_task"] not in assets:
            raw = read(audit["path"])
            track(Path(audit["path"]))
            # Official content only: local-only unused extras are documented separately.
            official = [x for x in raw["files"] if x["official_digest"] is not None]
            require(all(x["matches"] for x in official), "Official asset mismatch")
            public_audit = {k: raw[k] for k in
                            ("official_repository", "revision", "task", "seed", "layouts",
                             "scopes", "scope_note")}
            public_audit["scopes"] = ["Assets/" + s.split("/Assets/", 1)[1] for s in raw["scopes"]]
            public_audit.update(files=official, missing=[], matched=len(official),
                                total=len(official), all_match=True,
                                excluded_local_extra_paths=[x["path"] for x in raw["files"]
                                                            if x["official_digest"] is None])
            rel = Path("protocol/assets") / (e["native_task"] + ".json")
            write(root / rel, public_audit)
            assets[e["native_task"]] = rel.as_posix()
        selection.append({"episode_id": e["episode_id"],
                          "source_cohort": p.parent.name,
                          "source_result_sha256": digest(p / "result.json"),
                          "source_commit": manifest["source"]["commit"]})
        source_versions[manifest["source"]["commit"]] = {
            "python_files": manifest["source"]["python_sha256"],
            "dependencies": manifest["dependencies"],
            "provider": manifest["provider"]["version"],
            "host_python_version": manifest["host"]["python_version"],
            "host_packages": manifest["host"]["packages"],
            "simulation_python_version": manifest["host"]["simulation_python_version"],
        }
    require(len(seen) == 10, "Expected exactly one context per task")
    require(all(v == next(iter(configs.values())) for v in configs.values()),
            "Mixed execution configuration")
    write(root / "protocol/contexts.json", contexts)
    write(root / "protocol/context_bundle.json", seen)
    write(root / "protocol/runtime_fingerprints.json", source_versions)
    write(root / "results/efficiency_episodes.json", efficiencies)
    csv_file(root / "results/efficiency_episodes.csv", efficiencies)
    totals = efficiency_summary(efficiencies)
    write(root / "results/efficiency.json", totals)
    task_stats = [{"task": task, **efficiency_summary([x for x in efficiencies if x["task"] == task])}
                  for task in dict.fromkeys(x["task"] for x in efficiencies)]
    csv_file(root / "results/efficiency_tasks.csv", task_stats)
    write(root / "protocol/selection_history.json", {
        "rule": original["selection_rule"], "selected": selection,
        "original_attempts": original["original_attempts"],
        "total_attempts_in_recorded_cohorts": original["total_attempts"],
        "superseded_valid_attempts": original["superseded_valid_attempts"],
        "archived_invalid_attempts": original["archived_invalid_attempts"],
        "scope": original.get("attempt_count_scope"),
        "source_records_preserved": True,
        "note": "The full five-case packing repeat and five-case Make Kong v4 cohort replace the previous ten outcomes irrespective of score. Imitation layout 0 uses its valid infrastructure recovery. Earlier development is outside this cohort inventory.",
    })
    prepared = {e["native_task"]: seen[e["context_id"]]["input"] for e in episodes}
    reproduction = {**next(iter(configs.values())),
                    "robodojo_commit": next(iter(source_versions.values()))["dependencies"]["."],
                    "prepared_inputs": prepared, "asset_audits": assets,
                    "robodojo_root": "SET_ROBODOJO_ROOT",
                    "simulation_python": "SET_ISAAC_PYTHON",
                    "codex_bin": "SET_CODEX_BINARY",
                    "output_root": "SET_OUTPUT_ROOT"}
    write(root / "protocol/reproduction_config.json", reproduction)
    write(root / "protocol/suite.json", {"episodes": [
        {"task": e["native_task"], "seed": e["seed"], "layout": e["layout"]} for e in episodes]})
    write(root / "protocol/section6_3_reference.json", {
        "url": REFERENCE_URL, "section": "6.3 Fewer Tokens, Higher Success",
        "access_date": "2026-10-11",
        "reported_only_not_rerun_here": True,
        "hybrid": {"success_rate": .48, "control_steps": 42750, "executed_action_segments": 3776,
                   "mean_physical_duration_s": 34.20, "total_tokens": 624762828,
                   "cached_input_tokens": 607555840, "uncached_input_tokens": 15901963,
                   "output_tokens": 1305025},
        "direct": {"success_rate": .26, "control_steps": 38221, "executed_action_segments": 7729,
                   "mean_physical_duration_s": 30.58, "total_tokens": 1132343772,
                   "cached_input_tokens": 1107349760, "uncached_input_tokens": 22907448,
                   "output_tokens": 2086564},
    })
    shutil.copyfile(repo / "scripts/robodojo_release_evidence.py",
                    root / "robodojo_release_evidence.py")
    shutil.copyfile(repo / "scripts/configure_robodojo_release.py",
                    root / "configure.py")
    shutil.copyfile(repo / "docs/robodojo.md", root / "REPRODUCE.md")
    return totals


def verify_evidence(root, episodes):
    root = Path(root)
    contexts = read(root / "protocol/context_bundle.json")
    require("unlimited decisions and controls" not in (root / "index.html").read_text(),
            "Native gallery incorrectly describes unlimited controls")
    efficiency = read(root / "results/efficiency_episodes.json")
    require(len(contexts) == 10 and len(efficiency) == len(episodes), "Evidence count mismatch")
    recalculated = []
    for e in episodes:
        u = usage_totals(jsonl(root / e["usage_calls"]))
        x = execution_totals(jsonl(root / e["execution_segments"]))
        waypoints = jsonl(root / e["waypoints"])
        require(x["control_steps"] == e["control_steps"] <= e["native_step_limit"],
                "Native horizon/ACK mismatch")
        require(x["decisions"] == e["decisions"], "Decision count mismatch")
        require(sum(w["executed_controls"] for w in waypoints) == e["control_steps"],
                "Waypoint count mismatch")
        trace = jsonl(root / e["native_score_trace"])
        require(len(trace) == e["control_steps"] + 1
                and [r["native_control_step"] for r in trace] == list(range(len(trace))),
                "Native score trace not contiguous")
        require(trace[-1]["score_100"] == e["score_100"]
                and trace[-1]["native_complete"] is True
                and trace[-1]["native_layout_valid"] is True, "Terminal score mismatch")
        recalculated.append({"episode_id": e["episode_id"], "task": e["task"],
                             "success": e["success"], "score_100": e["score_100"],
                             **x, **u, "executed_waypoints": sum(w["executed_controls"] > 0 for w in waypoints)})
    require(recalculated == efficiency, "Efficiency rows cannot be reproduced")
    require(efficiency_summary(efficiency) == read(root / "results/efficiency.json"),
            "Efficiency aggregate mismatch")
    for ctx in contexts.values():
        require(digest(root / ctx["input"]) == ctx["portable_input_sha256"], "Portable input changed")
        data = read(root / ctx["input"])
        blocks = [p for p in data["content"] if isinstance(p, dict)]
        require(len(blocks) == len(ctx["images"]) == 48, "Missing demonstration view")
        for part, ref in zip(blocks, ctx["images"]):
            path = Path(ctx["input"]).parent / part["image"]
            require(not path.is_absolute() and ".." not in path.parts, "Unsafe context path")
            require(path.as_posix() == ref["path"] and digest(root / path) == ref["sha256"],
                    "Demonstration image mismatch")
        require((root / ctx["agent_context"]).is_file(), "Missing public prompts/tools")
    with (root / "results/efficiency_episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(efficiency), "Efficiency CSV count mismatch")
    for row, expected in zip(rows, efficiency):
        for k, v in expected.items():
            require(row[k] == str(v), "Efficiency CSV value mismatch")
    return {"contexts": len(contexts), "demonstration_images": 480,
            "efficiency_recalculated": True, "native_score_traces": len(episodes)}


def write_native_docs(root, summary, totals, table):
    native = read(root / "protocol/reproduction_config.json")
    (root / "README.md").write_text(f"""# GPT-Policy · RoboDojo native-horizon results

**{summary['successes']}/50 successes ({summary['success_rate']:.0%}); Mean score {summary['task_macro_mean_score_100']:.2f}/100.**
All 50 selected episodes reached valid native termination. All failures remain
in the denominator and in every aggregate.

## Explore and reproduce

- [Three-camera video gallery](index.html)
- [Results](results/results.md) · [episode CSV](results/episodes.csv)
- [Execution and token accounting](results/efficiency.md)
- [Demonstrations and public prompts](contexts/README.md)
- [Reproduce the evaluation](REPRODUCE.md) — Hugging Face: [SII-CDZ/gpt-policy-robodojo-contexts](https://huggingface.co/datasets/SII-CDZ/gpt-policy-robodojo-contexts)
- [Protocol and selection](protocol/EVALUATION.md)

Run **python verify.py** to check hashes, native scores, controls, token accounting,
contexts and artifact references. Videos use H.264/yuv420p and fast-start MP4.

## Results

{table(summary)}

## Contents and selection

tasks/ holds all selected successes and failures: three camera videos, initial
and final previews, numerical controls, native per-control scores, tool execution
counts and sanitized per-call usage. contexts/ contains the actual ten expert
demonstrations and recorded public instructions/tool schemas. protocol/ binds
each case to versions, scene identities, settings, contexts and selection history.

See [evaluation protocol and run provenance](protocol/EVALUATION.md#run-provenance-and-demonstration-scope)
for the recorded versions, cohort selection and demonstration scope.

Only allowlisted evidence is exported. Private conversations, model reasoning,
credentials, account identifiers and private storage paths are excluded.
""")
    (root / "README.zh-CN.md").write_text(f"""# GPT-Policy 十任务原生步数评测

最终入选 **{summary["successes"]}/50 成功，成功率 {summary["success_rate"]:.0%}，Mean score {summary["task_macro_mean_score_100"]:.2f}**。全部 50 条均为有效原生终局；\n{summary["native_failures"]} 条失败完整保留，未按结果挑选重试。

- [三视角视频](index.html) · [完整结果](results/results.md)
- [控制步、动作段、token 统计](results/efficiency.md)
- [十任务示范与公共提示](contexts/README.md)
- [仿真复现](REPRODUCE.md)：下载对应上下文，配置环境后运行固定 50 条场景。
- Hugging Face：[SII-CDZ/gpt-policy-robodojo-contexts](https://huggingface.co/datasets/SII-CDZ/gpt-policy-robodojo-contexts)。

评测设置、版本及结果来源见[评测协议](protocol/EVALUATION.md#run-provenance-and-demonstration-scope)。
复现采用原生控制步上限与原生成功判定。
token 包含缓存输入；{totals["usage_missing_calls"]} 次失败重试没有返回用量，总量标为已记录下界，
不能当作完整账单或完整耗用。运行 **python verify.py** 可独立复算并校验。
""")
    scene_lines = "\n".join(f"- {t['task']}: " + ", ".join(
        f"{s['variant']} {s['layout']}" for s in t["scenes"]) + "."
        for t in summary["tasks"])
    (root / "protocol/EVALUATION.md").write_text(f"""# Evaluation protocol

## Fixed settings and scenes

- gpt-6-astra / medium; Codex 0.160.0; one JSON tool decision per model completion.
- Native control horizons; wall limit 9,000 seconds; unlimited model decisions.
- No live image window or host conversation replay. Recoverable API failures
  retry the pending decision in the same model conversation.
- Native IK; 0.15-rad joint increment; move/hold, early arrival, pass-through,
  and distributed closed-gripper transport enabled.
- RoboDojo pin: {native['robodojo_commit']}. Dependencies, actual runtime hashes,
  asset manifests and case-specific prompts are supplied.
- Hidden simulator object states and stage scores are not supplied to the policy.
  Native success conditions and action/path validation are unchanged.

{scene_lines}

Imitation uses layout 5 in place of the previously native-invalid layout 4;
it is therefore not the same complete 50-scene set as a 0–4 panel.
Three generalization tasks use standard 0–1 plus random 0–2.

## Outcomes and score

Success requires complete=true, valid_for_success_rate=true and native_success=true.
All 50 selected cases are valid native terminal outcomes. Mean score is the
arithmetic mean of all five native 0–100 final process scores per task.
The overall value is the mean of ten task means, equal here to the 50-case mean.
Every case includes its original on-run score trace; no score was replay-reconstructed.
Make Kong has binary native scoring in this pin. Intermediate visible progress
is not assigned invented points.

## Run provenance and demonstration scope

The original cohort and explicitly authorized replacement cohorts contain 69
recorded attempts: 50 selected, 10 superseded valid outcomes and 9 infrastructure
interruptions. These are cohort counts, not the total development compute.
The five packing repeats and five Make Kong v4 outcomes were selected together
regardless of score; the valid imitation layout-0 recovery is retained.
See selection_history.json and per-case provenance. No original attempt was deleted.

The selected runtime commits are 65d16dac and 3d21435. Their runtime difference
handles exhausted recoverable provider retries while preserving the current
conversation and simulation. Make Kong uses its v4 demonstration. Recorded
prompts, action schema and numerical controls are available for inspection.

These task/layout families were used during prompt, context and execution
development.
One historical expert demonstration per task is fixed across its five scenes.
Language classification explicitly relabels Classify episode 77; random variants
reuse their standard task's context. Historical expert success is not current
policy success. The expert dataset does not provide original seed/layout metadata;
overlap with evaluation layouts has not been verified.

Official asset manifests cover task-scoped resources, not every file in a
global asset tree. Local-only unused extras are named in each audit and excluded
from the reproduction dependency list. Loaded layers confirm the official X5
ARX.usd; the high-friction alternate robot was not loaded.

## Comparison and efficiency

The [reference report, section 6.3]({REFERENCE_URL}) counts selected runs,
cached input tokens and physical time separately. We use those categories and
provide raw per-case and per-call records. An executed action segment here is one
move_eef_chunk result with at least one acknowledged control; partial execution
counts once, validation-only rejection counts zero. Waypoints and all model
decisions are separately reported because a chunk can contain several targets.

Our ICL demonstration, medium effort, tool protocol, selected replacement history
and imitation scene differ from the reference's xhigh setup. Published baseline
values are attributed references, not baselines rerun under this harness.
Missing failed-call token usage prevents a complete total-cost comparison.
Shorter physical time alone is not evidence of faster end-to-end completion.
""")
    (root / "protocol/SCHEMA.md").write_text("""# Evidence fields

- results/episodes.json repeats each task result exactly. Native flags determine success.
- context_id, configuration_id and version_id resolve in protocol/*.json.
- actions.jsonl contains contiguous acknowledged joint controls, not intended-only actions.
- native_score_trace.jsonl contains initial plus every acknowledged control's native score.
- execution_segments.jsonl records each tool result and its acknowledged controls.
- waypoints.jsonl preserves executed/requested controls at the target level.
- usage_calls.jsonl exports only accounting fields. usage=null means unknown, not zero.
- input_tokens includes cached input; total=input+output. Reasoning is part of output.
- Context originals and portable inputs have separate hashes because paths were rewritten.
- loaded_assets.json lists actual asset-layer paths relative to the upstream checkout.
""")
    (root / "THIRD_PARTY.md").write_text("""# Attribution and scope

- [GPT-Policy](https://github.com/cheng-haha/GPT-Policy): harness and release tools.
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo):
  task definitions, simulator and source expert trajectories.
- [RoboDojo assets](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo).
- [GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy):
  scene-composition and efficiency-reporting reference.
- [RoboDawn](https://github.com/Hugo-AGI/RoboDawn): imitation layout substitution reference.

This bundle contains selected demonstration images/annotations/numeric references,
generated evaluation videos and execution evidence. It does not contain the full
expert dataset, simulator asset tree or model weights. Historical expert data was
provided as successful trajectories; see context metadata for source task/episode
and the explicit language-instruction relabelling. The original dataset release
identifier/license metadata is not supplied by that local source.

Third-party material retains its original rights and terms. LICENSE.txt preserves
the existing review/evaluation notice; this export does not grant a new license
for third-party content. Demonstration contexts: https://huggingface.co/datasets/SII-CDZ/gpt-policy-robodojo-contexts.
""")
    lines = ["# Demonstrations and public prompts", "",
             "Hugging Face: [SII-CDZ/gpt-policy-robodojo-contexts](https://huggingface.co/datasets/SII-CDZ/gpt-policy-robodojo-contexts).", "",
             "One fixed expert episode per task; 16 timepoints × three camera views.",
             "All annotations and numeric action/state references used by the panel are included.",
             "Only storage paths were made portable; images retain their exact source bytes.", "",
             "| Task | Source episode | Prepared input | Public instructions/tools |",
             "|---|---:|---|---|"]
    contexts = read(root / "protocol/contexts.json")
    for c in contexts.values():
        task = c["task"]
        lines.append(f"| {task} | {c['expert_episode_index']} | "
                     f"[input.json]({task}/prepared/input.json) | "
                     f"[agent_context.json]({task}/agent_context.json) |")
    lines += ["", "Numeric next-state references are",
              "not original issued commands and are not replayed as a policy.",
              "Poses are link6 in shared environment_origin, metres and xyzw.",
              "", "Download and validate the context bundle using the reproduction guide."]
    (root / "contexts/README.md").write_text("\n".join(lines) + "\n")
    refs = read(root / "protocol/section6_3_reference.json")
    rows = [
        ("Success rate", f"{totals['success_rate']:.0%}", "48%", "26%"),
        ("Actual control steps", f"{totals['control_steps']:,}", "42,750", "38,221"),
        ("Executed action segments", f"{totals['executed_action_segments']:,}", "3,776", "7,729"),
        ("Mean physical duration (s)", f"{totals['mean_physical_duration_s']:.2f}", "34.20", "30.58"),
        ("Total recorded tokens (cached input included)", f"{totals['total_tokens']:,}*", "624,762,828", "1,132,343,772"),
        ("Cached input tokens", f"{totals['cached_input_tokens']:,}*", "607,555,840", "1,107,349,760"),
        ("Uncached input tokens", f"{totals['uncached_input_tokens']:,}*", "15,901,963", "22,907,448"),
        ("Output tokens", f"{totals['output_tokens']:,}*", "1,305,025", "2,086,564"),
    ]
    table_text = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    (root / "results/efficiency.md").write_text(f"""# Execution and token accounting

Categories follow [GPT-as-Policy §6.3]({refs['url']}), accessed 2026-10-11.
Reference columns reproduce their published figures; they were not rerun here.

| Metric | GPT-Policy, selected 50 | Reference hybrid | Reference direct |
|---|---:|---:|---:|
{table_text}

**\\* Recorded lower bound, not a complete token bill:** {totals['usage_missing_calls']}
failed calls across {totals['cases_with_missing_usage']} selected episodes did not return
usage. All {totals['completed_model_calls']:,} completed calls did. Unknown usage is
retained as null in per-call evidence, not fabricated or estimated. Reasoning tokens
are already part of output and are not counted twice. These figures exclude
earlier unselected attempts and demonstration preparation/development compute.

An executed segment is one motion chunk that acknowledged at least one control,
including partially executed chunks. There were {totals['zero_control_motion_calls']}
zero-control motion calls and {totals['non_motion_tool_calls']} other tool decisions.
All {totals['decisions']:,} tool decisions and {totals['executed_waypoints']:,} executed
waypoints are available separately. A chunk is not a single control or waypoint.

Physical duration is control_steps × 0.04 seconds; it excludes model waits.
The first video frame is the initial observation, so video frame counts are
controls + 1. Neither token counts nor physical duration represent account charges
or wall-clock completion time.

Our medium-effort ICL/chunk harness differs from the reference's xhigh direct/hybrid
setups and their selected scene/retry histories. This descriptive table does not
isolate a causal token or success improvement.

- [Per-episode CSV](efficiency_episodes.csv)
- [Per-task CSV](efficiency_tasks.csv)
- [Aggregate JSON](efficiency.json)
""")
