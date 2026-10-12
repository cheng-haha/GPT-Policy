"""Export a portable, allowlisted RoboDojo results release without changing sources."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

from verify_robodojo_release import aggregate, digest, read, require, verify

CAMERAS = {"cam_high": "overhead", "cam_left_wrist": "left_wrist",
           "cam_right_wrist": "right_wrist"}
CONFIG_KEYS = ("model", "effort", "max_decisions", "live_image_window",
               "control_step_limit", "episode_wall_time_s")
SAFE_STATUSES = {"native_completed", "budget_censored", "invalid_native_layout",
                 "runtime_error", "incomplete", "controller_interrupted", "quota_error"}
SAFE_REASONS = {"native_terminal", "wall_time_budget", "decision_budget",
                "model_give_up", "invalid_native_layout", "runtime_error",
                "engineering_support_demo_checked", "offline_replay_end"}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite numeric data")
    return value


def vector(value, length):
    require(type(value) is list and len(value) == length, "Wrong numeric vector width")
    return [number(x) for x in value]


def public_action(row):
    # Do not copy notes, provider IDs, free-form feedback, or unknown future keys.
    out = {"control_step": row["step"], "joint_command": vector(row["command"], 14)}
    require(type(out["control_step"]) is int and out["control_step"] > 0, "Invalid step")
    requested = row.get("requested", {})
    targets = {}
    for arm in ("left", "right"):
        if arm in requested:
            targets[arm] = {"pose_xyzquat": vector(requested[arm]["pose_xyzquat"], 7),
                            "gripper": number(requested[arm]["gripper"])}
    if targets:
        out["requested_targets"] = targets
        out["requested_chunk_steps"] = requested["steps"]
    return out


def write_actions(source, target):
    count = 0
    with source.open() as inp, target.open("x") as out:
        for count, line in enumerate(inp, 1):
            row = public_action(json.loads(line))
            require(row["control_step"] == count, "Source controls are not contiguous")
            out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    return count


def public_context(manifest, stored_input):
    source = {}
    title = None
    for part in stored_input.get("content", []):
        if not isinstance(part, str):
            continue
        try:
            data = json.loads(part)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("source"), dict):
            source = data["source"]
            title = data.get("title")
            break
    context = manifest["context"]
    data = {
        "input_sha256": context["sha256"],
        "expert_dataset_task": Path(source.get("dataset", "unknown")).name,
        "expert_episode_index": source.get("episode_index"),
        "historical_image_count": context["image_count"],
        "timepoints": context["image_count"] // 3,
        "views": list(CAMERAS.values()),
        "instruction": title,
        "historical_expert_success_is_not_evaluation_success": True,
        "raw_demonstration_distributed": False,
    }
    relabel = source.get("instruction_relabel")
    if relabel:
        data["instruction_relabel"] = {
            k: relabel[k] for k in
            ("original_instruction", "replacement_instruction", "reason",
             "native_target_success_verified") if k in relabel
        }
    return data


def safe_status(result):
    require(result["status"] in SAFE_STATUSES, "Unknown result status")
    return result["status"]


def metadata(result):
    reason = result.get("reason")
    return {
        "status": safe_status(result),
        "reason": reason if reason in SAFE_REASONS else "recorded_in_private_source",
        "native_complete": result.get("complete") is True,
        "native_valid_for_success_rate": result.get("valid_for_success_rate") is True,
        "native_success": result.get("native_success"),
        "decisions": result.get("decisions"),
        "control_steps": result.get("native_control_steps", result.get("controls")),
    }


def public_identity(task, row, identity):
    variant = row.get("variant", identity.get("variant", "standard"))
    require(variant in {"standard", "random"}, "Unknown scene variant")
    native_task = task + ("_random" if variant == "random" else "")
    require(identity["task"] == native_task
            and identity["layout_id"] == row["layout"], "Case identity mismatch")
    require(identity.get("variant", variant) == variant, "Variant identity mismatch")
    episode_id = f"{native_task}__g{identity['eval_seed']}__l{row['layout']}"
    leaf = ("random_" if variant == "random" else "") + f"layout_{row['layout']}"
    relative = Path("tasks") / task / f"seed_{identity['eval_seed']}" / leaf
    return variant, native_task, episode_id, relative


def probe(path):
    value = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_frames,duration,avg_frame_rate",
        "-of", "json", str(path)], text=True))["streams"][0]
    require(value["codec_name"] == "h264" and value["pix_fmt"] == "yuv420p",
            "Source video must be H264/yuv420p")
    return value


def export_video(source, target):
    info = probe(source)
    # Stream copy preserves every encoded frame and removes container metadata.
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(source),
                    "-map", "0:v:0", "-c", "copy", "-an", "-map_metadata", "-1",
                    "-movflags", "+faststart", str(target)], check=True)
    after = probe(target)
    require(info == after, "Video stream changed during lossless remux")
    return info


def frames(source, initial, final):
    from PIL import Image
    for target, args in [
        (initial, ["-i", str(source)]),
        (final, ["-sseof", "-1", "-i", str(source), "-vf", "reverse"]),
    ]:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *args,
                        "-frames:v", "1", "-q:v", "2", str(target)], check=True)
        # Re-save pixels to ensure previews carry no source metadata.
        image = Image.open(target).convert("RGB")
        image.save(target, quality=90)


def table(summary):
    lines = [
        "| Task | Mean score / 100 | Successes | Success rate | Timeouts | Native failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in summary["tasks"]:
        lines.append(f"| {task['task']} | {task['mean_score_100']:.2f} | "
                     f"{task['successes']}/{task['episodes']} | {task['success_rate']:.0%} | "
                     f"{task['timeouts']} | {task['native_failures']} |")
    lines.append(f"| **Total / macro mean** | **{summary['task_macro_mean_score_100']:.2f}** | "
                 f"**{summary['successes']}/{summary['episodes']}** | "
                 f"**{summary['success_rate']:.0%}** | **{summary['timeouts']}** | "
                 f"**{summary['native_failures']}** |")
    return "\n".join(lines)


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gallery(episodes, summary):
    data = json.dumps(episodes, ensure_ascii=False).replace("<", "\\u003c")
    head = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT-Policy · RoboDojo results</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#172438;background:#f4f6f9}
body{max-width:1240px;margin:auto;padding:36px 24px}h1{font-size:36px;margin:8px 0}
.subtitle{color:#556477;max-width:940px;line-height:1.65}
.metrics{display:flex;gap:16px;margin:26px 0}.metric{background:white;border:1px solid #dbe2ea;border-radius:12px;padding:18px 28px;flex:1}
.metric strong{display:block;font-size:32px}.metric span{color:#627085}
nav{display:flex;gap:18px;margin:22px 0}a{color:#205eb7;text-decoration:none}a:hover{text-decoration:underline}
.filters{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}select{padding:10px;border:1px solid #cad3df;border-radius:8px;background:white}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:22px}
article{border:1px solid #dbe2ea;background:white;border-radius:12px;overflow:hidden}
article .body{padding:16px}h2{font-size:17px;margin:0 0 8px}video{display:block;width:100%;background:#111;aspect-ratio:4/3}
.badge{display:inline-block;border-radius:5px;padding:3px 7px;font-size:12px;background:#e5f5eb;color:#176239}
.timeout{background:#fff1d3;color:#785100}.native_failure{background:#fbe5e4;color:#9d2727}
.meta{font-size:13px;color:#627085;line-height:1.7;margin:9px 0}
.views{display:flex;gap:8px}.views button{font:inherit;font-size:12px;border:1px solid #c8d2df;border-radius:6px;background:#f8fafc;cursor:pointer;padding:6px}
.views button.active{background:#245eba;color:white;border-color:#245eba}
footer{margin:34px 0;color:#627085;font-size:13px;line-height:1.7}
</style>
<body><div>GPT-POLICY / ROBO DOJO / 10 TASKS</div><h1>Evaluation results</h1>
<p class="subtitle">A portable archive of 50 selected episodes. All successes, native failures and timeouts are included.
This consolidated panel spans implementation versions. __RECONSTRUCTED__ historical stage scores were reconstructed by fixed-action replay.
The invalid imitation layout 4 is excluded and replaced by layout 5.</p>
<div class="metrics"><div class="metric"><strong>__RATE__</strong><span>Success · __SUCCESS__ / __COUNT__</span></div>
<div class="metric"><strong>__MEAN__</strong><span>Mean native stage score / 100</span></div>
<div class="metric"><strong>__VIDEOS__</strong><span>Videos · three views per episode</span></div></div>
<nav><a href="README.md">Readme</a><a href="results/summary.csv">Summary CSV</a>
<a href="results/episodes.csv">Episode CSV</a><a href="protocol/EVALUATION.md">Protocol & limitations</a></nav>
<div class="filters"><select id="task" aria-label="Task"><option value="">All 10 tasks</option></select>
<select id="outcome" aria-label="Outcome"><option value="">All outcomes</option><option value="success">Success</option>
<option value="timeout">Timeout</option><option value="native_failure">Native failure</option></select>
<span id="count"></span></div><main id="grid" class="grid"></main>
"""
    tail = """
<script>
const episodes=JSON.parse(document.getElementById('episode-data').textContent);
const tasks=[...new Set(episodes.map(e=>e.task))];
tasks.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t.replaceAll('_',' ');document.getElementById('task').append(o)});
function render(){
 const task=document.getElementById('task').value,outcome=document.getElementById('outcome').value;
 const rows=episodes.filter(e=>(!task||e.task===task)&&(!outcome||e.outcome===outcome));
 document.getElementById('count').textContent=rows.length+' episodes';
 const grid=document.getElementById('grid');grid.replaceChildren();
 rows.forEach(e=>{
  const card=document.createElement('article'),v=document.createElement('video');v.controls=true;v.preload='none';v.poster=e.frames.final;v.src=e.videos.overhead;card.append(v);
  const body=document.createElement('div');body.className='body';
  const h=document.createElement('h2');h.textContent=e.task.replaceAll('_',' ')+' · '+(e.variant||'standard')+' layout '+e.layout;body.append(h);
  const badge=document.createElement('span');badge.className='badge '+e.outcome;badge.textContent=e.outcome.replaceAll('_',' ')+' · score '+e.score_100;body.append(badge);
  const meta=document.createElement('div');meta.className='meta';meta.textContent=e.decisions+' decisions · '+e.control_steps+' controls · seed '+e.seed;body.append(meta);
  const views=document.createElement('div');views.className='views';
  Object.entries(e.videos).forEach(([name,path])=>{const b=document.createElement('button');b.textContent=name.replaceAll('_',' ');if(name==='overhead')b.className='active';
   b.onclick=()=>{const t=v.currentTime,playing=!v.paused;v.src=path;v.addEventListener('loadedmetadata',()=>{v.currentTime=t;if(playing)v.play()},{once:true});views.querySelectorAll('button').forEach(x=>x.className='');b.className='active'};views.append(b)});
  body.append(views);const link=document.createElement('a');link.href=e.metrics;link.textContent='Metrics JSON';const p=document.createElement('p');p.append(link);body.append(p);
  if(e.score_basis==='offline_replay_reconstruction'){const note=document.createElement('div');note.className='meta';note.textContent='Stage score reconstructed offline; episode outcome unchanged.';body.append(note)}
  card.append(body);grid.append(card);
 });
}
document.getElementById('task').onchange=render;document.getElementById('outcome').onchange=render;render();
</script><footer>9000-second wall-time budget; unlimited decisions and controls; native terminal predicates unchanged.
Each video encodes one frame per control step at 25 fps, not model wall-clock latency.
Run <code>python verify.py</code> to verify all checksums and recompute the table.
See LICENSE.txt and THIRD_PARTY.md for the existing review/evaluation terms and attribution.</footer></body></html>
"""
    for key, value in {
        "__RECONSTRUCTED__": summary["reconstructed_score_cases"],
        "__RATE__": f"{summary['success_rate']:.0%}",
        "__SUCCESS__": summary["successes"], "__COUNT__": summary["episodes"],
        "__MEAN__": f"{summary['task_macro_mean_score_100']:.2f}",
        "__VIDEOS__": len(episodes) * 3,
    }.items():
        head = head.replace(key, str(value))
    return head + '<script id="episode-data" type="application/json">' + data + "</script>" + tail


def export(args):
    original_report = read(args.report)
    native_panel = "episodes" in original_report and "valid" in original_report
    if native_panel:
        from robodojo_release_evidence import native_report
        report = native_report(original_report)
    else:
        report = original_report
    root = args.staging.resolve()
    require(not root.exists() and not args.output.exists(), "Output already exists")
    root.mkdir(parents=True)
    (root / "results").mkdir()
    (root / "protocol").mkdir()
    source_hashes = {str(args.report.resolve()): digest(args.report)}
    source_map, episodes, contexts, configs, versions, video_audit = [], [], {}, {}, {}, []
    replay_records = []
    for task in report["tasks"]:
        for row in task["cases"]:
            source = Path(row["source"])
            require(digest(source) == row["source_sha256"], "Selected result changed")
            source_hashes[str(source)] = row["source_sha256"]
            parent = source.parent
            result = read(source)
            require(safe_status(result) in {"native_completed", "budget_censored"},
                    "Final release still contains unresolved episodes")
            manifest = read(parent / "manifest.json")
            config = read(parent / "run_config.json")
            for name in ("manifest.json", "run_config.json", "model-input-manifest.json"):
                source_hashes[str(parent / name)] = digest(parent / name)
            identity = manifest["case"]
            variant, native_task, episode_id, relative = public_identity(task["task"], row, identity)
            folder = root / relative
            folder.mkdir(parents=True)
            (folder / "videos").mkdir()
            (folder / "frames").mkdir()
            keys = CONFIG_KEYS
            if native_panel:
                from robodojo_release_evidence import CONFIG_KEYS as native_config_keys
                keys = native_config_keys
            config_public = {k: config.get(k) for k in keys}
            config_id = "config_" + fingerprint(config_public)[:12]
            configs[config_id] = config_public
            ctx = public_context(manifest, read(parent / "model-input-manifest.json"))
            context_id = "context_" + ctx["input_sha256"][:16]
            if context_id in contexts:
                require(contexts[context_id] == ctx, "Context hash metadata mismatch")
            contexts[context_id] = ctx
            version = {
                "source_commit": manifest["source"]["commit"],
                "runtime_python_fingerprint": fingerprint(manifest["source"]["python_sha256"]),
                "dependency_commits": manifest["dependencies"],
                "provider_version": manifest["provider"]["version"],
            }
            version_id = "version_" + fingerprint(version)[:12]
            versions[version_id] = version
            meta = metadata(result)
            success = (meta["native_complete"] and meta["native_valid_for_success_rate"]
                       and meta["native_success"] is True)
            outcome = ("success" if success else "timeout"
                       if result["status"] == "budget_censored" else "native_failure")
            episode = {
                "episode_id": episode_id, "task": task["task"], "task_name": task["name"],
                "seed": identity["eval_seed"], "layout": row["layout"],
                "variant": variant, "native_task": native_task,
                **meta, "success": success, "outcome": outcome,
                "score_100": number(row["score_100"]), "score_basis": row["basis"],
                "configuration_id": config_id, "context_id": context_id,
                "version_id": version_id,
                "metrics": (relative / "result.json").as_posix(),
                "actions": (relative / "actions.jsonl").as_posix(),
                "provenance": (relative / "provenance.json").as_posix(),
                "frames": {k: (relative / "frames" / (k + ".jpg")).as_posix()
                           for k in ("initial", "final")}, "videos": {},
            }
            count = write_actions(parent / "sim/actions.jsonl", folder / "actions.jsonl")
            require(count == episode["control_steps"], f"Action count mismatch: {episode_id}")
            source_hashes[str(parent / "sim/actions.jsonl")] = digest(parent / "sim/actions.jsonl")
            provenance = {
                "original_result_sha256": row["source_sha256"],
                "original_manifest_sha256": digest(parent / "manifest.json"),
                "original_model_input_manifest_sha256": digest(parent / "model-input-manifest.json"),
                "layout_sha256": identity["layout_sha256"],
                "source_commit": version["source_commit"],
                "context_input_sha256": ctx["input_sha256"],
                "historical_expert_outcome_is_not_evaluation_outcome": True,
                "original_video_sha256": {},
            }
            for old_name, public_name in CAMERAS.items():
                original = parent / "sim" / (old_name + ".mp4")
                target = folder / "videos" / (public_name + ".mp4")
                sha = digest(original)
                source_hashes[str(original)] = sha
                provenance["original_video_sha256"][public_name] = sha
                info = export_video(original, target)
                episode["videos"][public_name] = (relative / "videos" / target.name).as_posix()
                video_audit.append({"path": episode["videos"][public_name],
                                    "metadata_removed": True, "lossless_remux": True, **info})
            frames(folder / "videos/overhead.mp4",
                   folder / "frames/initial.jpg", folder / "frames/final.jpg")
            trace = parent / "sim/native-process-scores.jsonl"
            if trace.exists():
                source_hashes[str(trace)] = digest(trace)
                fields = ("native_control_step", "native_complete", "native_layout_valid", "score_100")
                with trace.open() as inp, (folder / "native_score_trace.jsonl").open("x") as target:
                    for line in inp:
                        d = json.loads(line)
                        target.write(json.dumps({k: d[k] for k in fields}) + "\n")
                episode["native_score_trace"] = (relative / "native_score_trace.jsonl").as_posix()
            if row["basis"] == "offline_replay_reconstruction":
                original = Path(row["replay_result"])
                replay = read(original)
                require(replay["reconstructed_score_100"] == row["score_100"]
                        and replay["replayed_all_controls"] is True
                        and replay["model_calls"] == 0, "Replay score evidence mismatch")
                fields = ("score_kind", "reconstructed_score_100", "controls_applied",
                          "controls_expected", "replayed_all_controls", "model_calls",
                          "max_link6_position_error_m", "max_joint_error_rad",
                          "max_gripper_error", "all_rgb_exact", "image_mae_255",
                          "exact_original_score_claim")
                public_replay = {"episode_id": episode_id, "source_sha256": digest(original),
                                 **{k: replay[k] for k in fields}}
                replay_records.append(public_replay)
                write(folder / "score_reconstruction.json", public_replay)
                episode["score_reconstruction"] = (relative / "score_reconstruction.json").as_posix()
                source_hashes[str(original)] = digest(original)
            write(folder / "provenance.json", provenance)
            write(folder / "result.json", episode)
            episodes.append(episode)
            source_map.append({"episode_id": episode_id, "source": str(parent),
                               "public": relative.as_posix()})
            print(json.dumps({"exported": len(episodes), "episode": episode_id}), flush=True)

    summary = aggregate(episodes)
    expected = report.get("overall", report)
    expected_successes = expected.get("confirmed_native_successes", expected.get("confirmed_successes"))
    require(summary["successes"] == expected_successes
            and summary["task_macro_mean_score_100"] == expected["task_macro_mean_score_100"],
            "Release metrics differ from final report")
    write(root / "results/summary.json", summary)
    write(root / "results/episodes.json", episodes)
    (root / "results/results.md").write_text("# Final selected panel\n\n" + table(summary) + "\n")
    write_csv(root / "results/summary.csv", summary["tasks"],
              ["task", "name", "episodes", "successes", "success_rate", "mean_score_100",
               "timeouts", "native_failures", "layouts", "scenes"])
    write_csv(root / "results/episodes.csv", episodes,
              ["episode_id", "task", "task_name", "seed", "variant", "layout", "native_task", "outcome", "success",
               "score_100", "score_basis", "decisions", "control_steps", "context_id",
               "configuration_id", "version_id", "metrics"])
    write(root / "protocol/configurations.json", configs)
    write(root / "protocol/contexts.json", contexts)
    write(root / "protocol/versions.json", versions)
    write(root / "protocol/score_reconstruction.json", replay_records)
    write(root / "protocol/video_validation.json", video_audit)

    history = []
    for path in args.history_manifest:
        data = read(path)
        for case in data["cases"]:
            attempts = []
            for old in case.get("prior_attempts", []):
                p = Path(old["output"]) / "result.json"
                require(digest(p) == old["result_sha256"], "Historical attempt changed")
                result = read(p)
                attempts.append({"attempt": old["attempt"],
                                 "original_result_sha256": digest(p), **metadata(result)})
                source_hashes[str(p)] = digest(p)
            history.append({"task": case["spec"]["task"], "seed": case["spec"]["seed"],
                            "layout": case["spec"]["layout"],
                            "selected_attempt": case["attempt_number"],
                            "earlier_attempts": attempts})
    write(root / "protocol/selection_history.json", {
        "rule": "Latest authorized attempt, never best-of; one native-invalid scene advanced to next layout.",
        "coverage": "The 15-case recovery and subsequent two-case exception batch; earlier Classify development runs are outside this selected-panel release.",
        "changes": history,
    })
    skipped = []
    exclusion_report = report
    if "previous_report" in report and "skipped_invalid_layouts" not in report:
        previous_path = Path(report["previous_report"])
        require(digest(previous_path) == report["previous_report_sha256"], "Previous report changed")
        source_hashes[str(previous_path)] = digest(previous_path)
        exclusion_report = read(previous_path)
    for row in exclusion_report.get("skipped_invalid_layouts", []):
        original = Path(row["source"])
        require(digest(original) == row["source_sha256"], "Invalid-scene evidence changed")
        result = read(original)
        value = {"task": row["task"], "seed": row["seed"], "layout": row["layout"],
                 "replacement_layout": row["replacement_layout"], **metadata(result),
                 "model_calls": result["model_calls"], "support_checked": result["support_checked"],
                 "native_layout_valid": result["native_layout_valid"],
                 "original_result_sha256": row["source_sha256"], "counted_in_metrics": False}
        skipped.append(value)
        write(root / "excluded" / row["task"] / f"layout_{row['layout']}.json", value)
        source_hashes[str(original)] = row["source_sha256"]
    write(root / "protocol/excluded_layouts.json", skipped)
    for task in summary["tasks"]:
        group = [e for e in episodes if e["task"] == task["task"]]
        lines = [f"# {task['task']} / {task['name']}", "",
                 f"Success: {task['successes']}/5. Mean score: {task['mean_score_100']:.2f}.", "",
                 "| Variant / layout | Outcome | Score | Metrics | Overhead video |",
                 "|---|---|---:|---|---|"]
        for e in group:
            rel = Path(e["metrics"]).parent.relative_to(Path("tasks") / task["task"]).as_posix()
            lines.append(f"| {e['variant']} / {e['layout']} | {e['outcome']} | {e['score_100']:.2f} | "
                         f"[JSON]({rel}/result.json) | [Video]({rel}/videos/overhead.mp4) |")
        (root / "tasks" / task["task"] / "README.md").write_text("\n".join(lines) + "\n")

    repo = Path(__file__).resolve().parents[1]
    shutil.copyfile(repo / "LICENSE", root / "LICENSE.txt")
    shutil.copyfile(Path(__file__).with_name("verify_robodojo_release.py"), root / "verify.py")
    if args.previous_release:
        previous = args.previous_release
        previous_episodes = read(previous / "results/episodes.json")
        old_by_id = {e["episode_id"]: e for e in previous_episodes}
        new_by_id = {e["episode_id"]: e for e in episodes}
        fields = ("episode_id", "task", "seed", "layout", "outcome", "success",
                  "score_100", "score_basis", "context_id", "version_id")
        def selected_record(e):
            return {**{k: e[k] for k in fields},
                    "variant": e.get("variant", "standard")}
        write(root / "protocol/panel_update.json", {
            "reason": "Scene composition alignment: standard layouts 0,1 plus random layouts 0,1,2 for largest-number, folding and packing. Selection is independent of observed score.",
            "previous_release_checksum_manifest_sha256": digest(previous / "SHA256SUMS"),
            "previous_summary": read(previous / "results/summary.json"),
            "removed_from_current_panel": [selected_record(e) for k, e in old_by_id.items()
                                          if k not in new_by_id],
            "added_to_current_panel": [selected_record(e) for k, e in new_by_id.items()
                                      if k not in old_by_id],
            "previous_release_archived_unchanged": True,
        })
        selection = read(root / "protocol/selection_history.json")
        selection["panel_update"] = "protocol/panel_update.json"
        for change in selection["changes"]:
            change["in_current_panel"] = any(
                e["task"] == change["task"] and e["seed"] == change["seed"]
                and e["layout"] == change["layout"] and e["variant"] == "standard"
                for e in episodes)
        write(root / "protocol/selection_history.json", selection)
    write_docs(root, summary, contexts, versions, episodes)
    if native_panel:
        from robodojo_release_evidence import enrich, write_native_docs
        totals = enrich(root, original_report, episodes, source_map, contexts,
                        source_hashes, repo)
        write_native_docs(root, summary, totals, table)
        write(root / "results/episodes.json", episodes)
        write_csv(root / "results/episodes.csv", episodes,
                  ["episode_id", "task", "task_name", "seed", "variant", "layout",
                   "native_task", "outcome", "success", "score_100", "score_basis",
                   "decisions", "control_steps", "native_step_limit", "context_id",
                   "configuration_id", "version_id", "metrics"])
        if args.previous_release:
            update = read(root / "protocol/panel_update.json")
            update["reason"] = "New native-horizon selected panel replaces the historical extended-control release. Complete cohorts retained irrespective of outcomes."
            update["changed_selected_results"] = [
                {"episode_id": e["episode_id"],
                 "previous": selected_record(old_by_id[e["episode_id"]]),
                 "current": selected_record(e)}
                for e in episodes if e["episode_id"] in old_by_id]
            write(root / "protocol/panel_update.json", update)
    html = gallery(episodes, summary)
    if native_panel:
        import re
        html = re.sub(r'<p class="subtitle">.*?</p>',
            '<p class="subtitle">50 selected native-horizon episodes, including all failures. '
            'The panel consolidates recorded versions and explicitly selected replacement cohorts. '
            'Ten demonstrations, 480 images and independently verifiable execution/token records are included.</p>',
            html, count=1, flags=re.DOTALL)
        html = html.replace('<nav>', '<nav><a href="results/efficiency.md">Efficiency</a>'
                            '<a href="contexts/README.md">Demonstrations</a>'
                            '<a href="REPRODUCE.md">Reproduce</a>')
        html = html.replace("unlimited decisions and controls",
                            "unlimited decisions; native task control limits")
    (root / "index.html").write_text(html)
    write(root / "protocol/release_manifest.json", {
        "release": args.output.name, "schema_version": 2 if native_panel else 1,
        "native_horizon": native_panel,
        "demonstrations_included": native_panel,
        "original_merged_report_sha256": digest(args.report),
        "exporter_source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "exporter_script_sha256": digest(__file__),
        "source_results_preserved": True, "self_contained_paths": True,
        "raw_private_sessions_distributed": False,
        "selection_history_included": True, "mixed_implementation_versions": True,
    })
    sums = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            sums.append(f"{digest(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    verification = verify(root)
    for path, sha in source_hashes.items():
        require(digest(path) == sha, f"Source changed during export: {path}")
    audit = {
        "verification": verification, "sources_unchanged": True,
        "source_files_checked": len(source_hashes), "source_hashes": source_hashes,
        "source_map": source_map, "target": str(args.output.resolve()),
        "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
    }
    write(args.audit, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root.rename(args.output)
    print(json.dumps({"release": str(args.output), **verification, "bytes": audit["bytes"]}), flush=True)


def write_docs(root, summary, contexts, versions, episodes):
    n = summary["episodes"]
    successes = summary["successes"]
    rate = summary["success_rate"]
    score = summary["task_macro_mean_score_100"]
    reconstructed = summary["reconstructed_score_cases"]
    traces = sum("native_score_trace" in e for e in episodes)
    scene_lines = []
    for task in summary["tasks"]:
        group = [e for e in episodes if e["task"] == task["task"]]
        scenes = ", ".join(f"{e['variant']} {e['layout']}" for e in group)
        scene_lines.append(f"- {task['task']}: {scenes}.")
    scene_text = "\n".join(scene_lines)
    intro = f"""# GPT-Policy · RoboDojo ten-task results

Final selected panel: **{successes}/{n} successes ({rate:.2%})**,
**Mean score {score:.2f} / 100**. Outcomes: {summary['timeouts']} wall-time cutoffs
and {summary['native_failures']} native terminal failures, alongside successes.
This consolidated panel spans implementation and context versions and uses an
extended control budget; it is not one homogeneous native-horizon benchmark run.

## Start here

- [Offline three-camera video gallery](index.html)
- [Results table](results/results.md), [summary CSV](results/summary.csv),
  [per-episode CSV](results/episodes.csv), [per-episode JSON](results/episodes.json)
- [Evaluation protocol and limitations](protocol/EVALUATION.md)
- [Panel selection update](protocol/panel_update.json)
- Run **python verify.py** to verify checksums, artifacts and metric recalculation.

## Results

"""
    if not (root / "protocol/panel_update.json").exists():
        intro = intro.replace("- [Panel selection update](protocol/panel_update.json)\n", "")
    outro = f"""
## Contents and scene identities

Task folders contain result.json, numerical actions, provenance, initial/final
previews and three H.264 videos. All paths are relative and no symlinks are used.
Standard scenes use tasks/<task>/seed_0/layout_<id>/; random scenes use
tasks/<task>/seed_0/random_layout_<id>/. The variant is explicit in JSON, CSV and
the gallery, so standard 0 and random 0 cannot be confused.

{scene_text}

Video playback is 25 fps with one frame per control plus the initial frame;
model thinking latency is excluded. Videos are losslessly remuxed, metadata is
removed, and fast-start H.264/yuv420p playback is retained.

{reconstructed} historical stage scores are explicitly labelled offline replay
reconstructions, not exact original online observations. {traces} selected episodes
include original native per-control score traces; missing traces are not fabricated.
Expert demonstrations are identified by episode and hashes, without redistributing
their raw images or datasets. Private conversations, provider reasoning, credentials,
account identifiers and server paths are excluded.

The previous release and all original experiments are preserved. This operation
updates the local publication candidate only. Existing LICENSE.txt review/evaluation
terms remain unchanged; THIRD_PARTY.md records attribution and distribution scope.
"""
    (root / "README.md").write_text(intro + table(summary) + "\n" + outro)
    (root / "README.zh-CN.md").write_text(
        f"# GPT-Policy 十任务评测结果\n\n"
        f"最终入选 {n} 条：{successes} 条原生成功、{summary['timeouts']} 条超时未成功、"
        f"{summary['native_failures']} 条原生失败。成功率 **{rate:.2%}**，Mean score **{score:.2f}**。\n\n"
        "打开 [index.html](index.html) 查看三视角视频；results/ 保存表格和逐条 CSV/JSON。"
        "最大数、折衣服、装箱采用 standard 0、1 加 random 0、1、2；"
        "standard 与 random 在目录、逐条记录和视频标题中明确区分。\n\n"
        + table(summary) + "\n\n"
        f"使用扩展控制预算：9000 秒，不限决策与控制步数；原生成功和动作校验不变。"
        f"Mean score 包含 {reconstructed} 条历史回放重建分数，逐条标注；"
        "超时保留在成功率分母中，不冒充原生终局失败。"
        "本表跨历史实现及上下文版本，不是统一版本、原生控制预算下的受控比较。\n\n"
        "每条实际场景、上下文、版本与计分依据见 result.json 和 protocol/。"
        "运行 python verify.py 可完整校验。旧发布版已归档，原始实验保留。"
        "本目录是本地公开发布候选包，未上传平台；许可声明保持原样。\n")
    (root / "protocol/EVALUATION.md").write_text(f"""# Evaluation protocol and limitations

- Model: gpt-6-astra / medium; wall time 9000 seconds, max_decisions=null,
  control_step_limit=null, live_image_window=null.
- Existing extended-controls execution without added smoothing; native success,
  action and path checks unchanged. The extended horizon differs from the original
  task control limits used by GPT-as-Policy, which also uses xhigh.
- Ten tasks, seed 0, five selected scenes per task. Actual scene identities:

{scene_text}

- Three generalization tasks use two standard and three random scenes. This aligns
  their selected scene composition with GPT-as-Policy. Scene alignment does not
  establish identical simulator versions, model effort, control budgets or tools.
  The recorded imitation substitution and excluded invalid-scene evidence remain
  in excluded/ and excluded_layouts.json.
- Success requires complete=true, valid_for_success_rate=true, native_success=true.
  Success rate divides these confirmed successes by all five selected cases,
  including wall-time cutoffs. Timeouts retain their original incomplete native
  flags and are not relabelled native terminal failures.
- Mean score is the arithmetic mean of the five 0..100 scores. All ten tasks have
  equal size, so task macro mean equals the 50-case mean. This is not the official
  full-benchmark aggregation over dimensions and seeds.
- {reconstructed} missing historical scores are reconstructed by fixed-control offline replay
  with no new model calls. Robot joint/FK/gripper checks matched; RGB was not
  bit-identical. Original object dynamics and original scores are not certified
  identical. Replay never changes the recorded success.
- Actual native per-control scores are retained where recorded, including scores
  at the 9000-second wall cutoff. Make Kong has no staged get_score in this pin;
  its historical timeout uses the documented official zero-score fallback.
- Source versions and contexts differ across historical attempts. Selection uses
  the latest authorized attempt rather than best-of; the earlier recovery history
  is retained in selection_history.json. Any subsequent scene-composition change
  is documented in panel_update.json, with removed and added cases and prior metrics.
- One expert demonstration per task, 16 timepoints / 48 three-view images.
  Context IDs identify actual expert episodes and any instruction relabelling.
  Random tasks reuse the explicitly mapped standard-task demonstration; they do
  not get a newly selected demonstration for each evaluation layout.
- Expert/evaluation initial arrangements differed in visual audits, but original
  dataset seed/layout provenance is unavailable; no formal disjoint-split
  certificate is claimed. Expert success never counts as evaluation success.
- Only public observations and robot state enter policy control. Native scores and
  predicate diagnostics are not supplied to the model.
""")
    (root / "protocol/DATA_DICTIONARY.md").write_text("""# Data dictionary

Per-case result.json is authoritative; results/episodes.json repeats it exactly.

- episode_id: native task, evaluation seed and layout identity.
- task: base benchmark task used for grouping; native_task includes the random suffix.
- variant: standard or random; scene identity is (task, variant, seed, layout).
- outcome: success, native_failure, or timeout. success requires all three native
  eligibility flags; timeouts retain native_complete=false and never become successes.
- score_100: native 0..100 stage score, documented fallback, or labelled reconstruction.
- score_basis: stored_normalized_native_score, stored_native_process_score,
  offline_replay_reconstruction, or official_no_get_score_fallback_at_user_wall_cutoff.
- decisions: model tool decisions; control_steps: executed simulator control ACKs.
- context_id / configuration_id / version_id: keys into protocol/*.json.
- actions.jsonl: contiguous numerical controls; joint_command contains left six
  joints + gripper, then right six joints + gripper. Joints are radians, gripper
  0 closed / 1 open. requested_targets are environment-frame link6 XYZ metres and
  xyzw quaternion requests, not claims of measured achievement. Model notes omitted.
- provenance.json: original artifact hashes, distinct from exported-file SHA256SUMS.
- native_score_trace.jsonl: original on-run native scores, only where recorded.
- score_reconstruction.json: fixed-action replay audit, only for labelled reconstructions.
- summary scenes: explicit variant/layout pairs; layouts alone may repeat across variants.

Videos are 25 fps, one frame per control plus initial frame; not model wall time.
""")
    (root / "THIRD_PARTY.md").write_text("""# Attribution and scope

- GPT-Policy: https://github.com/cheng-haha/GPT-Policy
- RoboDojo: https://github.com/RoboDojo-Benchmark/RoboDojo
- Assets: https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo
- Scene-composition reference: https://github.com/anonymous-report-421/GPT-as-Policy
- Invalid imitation layout substitution: https://github.com/Hugo-AGI/RoboDawn

Exact implementation/dependency commits are in protocol/versions.json.
This bundle distributes generated recordings and numerical execution evidence,
not simulator assets, expert datasets, model weights or simulator source.
Third-party content retains its rights and license terms. LICENSE.txt preserves
the project's existing review/evaluation notice; no new redistribution or
commercial license is assigned by this packaging operation.
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--staging", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--history-manifest", type=Path, action="append", default=[])
    p.add_argument("--previous-release", type=Path)
    export(p.parse_args())
