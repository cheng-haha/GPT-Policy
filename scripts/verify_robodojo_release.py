"""Verify a portable results bundle using only Python's standard library."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local_path(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe artifact path")
    target = root / p
    require(target.resolve().is_relative_to(root.resolve()), "Artifact escapes bundle")
    return target


def aggregate(episodes):
    tasks = []
    for task in dict.fromkeys(e["task"] for e in episodes):
        group = [e for e in episodes if e["task"] == task]
        n = len(group)
        for e in group:
            score = e["score_100"]
            require(type(score) in (int, float) and math.isfinite(score)
                    and 0 <= score <= 100, "Invalid score")
            success = (e["native_complete"] is True
                       and e["native_valid_for_success_rate"] is True
                       and e["native_success"] is True)
            require(e["success"] is success, "Success eligibility mismatch")
            expected = ("success" if success else "native_failure"
                        if e["native_complete"] is True
                        and e["native_valid_for_success_rate"] is True
                        and e["native_success"] is False else "timeout")
            require(e["outcome"] == expected, "Outcome mismatch")
        successes = sum(e["success"] for e in group)
        tasks.append({
            "task": task, "name": group[0]["task_name"], "episodes": n,
            "successes": successes, "success_rate": successes / n,
            "mean_score_100": sum(e["score_100"] for e in group) / n,
            "timeouts": sum(e["outcome"] == "timeout" for e in group),
            "native_failures": sum(e["outcome"] == "native_failure" for e in group),
            "layouts": [e["layout"] for e in group],
            "scenes": [{"variant": e.get("variant", "standard"), "layout": e["layout"]}
                       for e in group],
        })
    count = len(episodes)
    return {
        "schema_version": 1, "episodes": count, "tasks": tasks,
        "successes": sum(e["success"] for e in episodes),
        "success_rate": sum(e["success"] for e in episodes) / count,
        "task_macro_success_rate": sum(t["success_rate"] for t in tasks) / len(tasks),
        "task_macro_mean_score_100": sum(t["mean_score_100"] for t in tasks) / len(tasks),
        "episode_mean_score_100": sum(e["score_100"] for e in episodes) / count,
        "timeouts": sum(e["outcome"] == "timeout" for e in episodes),
        "native_failures": sum(e["outcome"] == "native_failure" for e in episodes),
        "reconstructed_score_cases": sum(
            e["score_basis"] == "offline_replay_reconstruction" for e in episodes),
    }


def verify(root):
    root = Path(root).resolve()
    require(root.is_dir(), "Bundle directory missing")
    sums = root / "SHA256SUMS"
    expected = {}
    for line in sums.read_text().splitlines():
        sha, relative = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", sha), "Invalid checksum")
        require(relative not in expected, "Duplicate checksum entry")
        path = local_path(root, relative)
        require(path.is_file() and not path.is_symlink(), "Artifact missing or linked")
        require(digest(path) == sha, f"Checksum mismatch: {relative}")
        expected[relative] = sha
    actual = set()
    forbidden_names = re.compile(
        r"(?:auth|credential|session|sqlite|\.env|private_runtime)", re.IGNORECASE)
    private_text = re.compile(
        r"/(?:mnt|root|Users|home)/|[A-Za-z]:\\Users\\|"
        r"\b(?:sk|hf|ghp)[_-][A-Za-z0-9]{16,}|"
        r'"(?:api_key|access_token|refresh_token|account_id|thread_id)"\s*:'
    )
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Symlink in release")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel != "SHA256SUMS":
            actual.add(rel)
        require(not forbidden_names.search(path.name), f"Private filename: {rel}")
        if path.suffix in {".json", ".jsonl", ".csv", ".md", ".html", ".txt", ".py"}:
            require(not private_text.search(path.read_text()), f"Private text: {rel}")
    require(actual == set(expected), "Unlisted or missing artifacts")
    episodes = read(root / "results/episodes.json")
    require(len(episodes) == 50, "Expected 50 selected episodes")
    require(len({e["episode_id"] for e in episodes}) == 50, "Duplicate episode")
    identities = [(e["task"], e.get("variant", "standard"), e["seed"], e["layout"])
                  for e in episodes]
    require(len(set(identities)) == 50, "Duplicate scene identity")
    require(len({e["metrics"] for e in episodes}) == 50, "Duplicate scene artifact path")
    for e in episodes:
        variant = e.get("variant", "standard")
        require(variant in {"standard", "random"}, "Unknown scene variant")
        native_task = e["task"] + ("_random" if variant == "random" else "")
        require(e.get("native_task", native_task) == native_task, "Native task mismatch")
        require(e["episode_id"] == f"{native_task}__g{e['seed']}__l{e['layout']}",
                "Variant missing from episode identity")
    summary = aggregate(episodes)
    require(summary == read(root / "results/summary.json"), "Summary mismatch")
    require(len(summary["tasks"]) == 10 and all(t["episodes"] == 5 for t in summary["tasks"]),
            "Task panel mismatch")
    configs = read(root / "protocol/configurations.json")
    contexts = read(root / "protocol/contexts.json")
    versions = read(root / "protocol/versions.json")
    for episode in episodes:
        require(episode["configuration_id"] in configs, "Unknown config")
        require(episode["context_id"] in contexts, "Unknown context")
        require(episode["version_id"] in versions, "Unknown source version")
        require(read(local_path(root, episode["metrics"])) == episode, "Case metrics mismatch")
        require(set(episode["videos"]) == {"overhead", "left_wrist", "right_wrist"},
                "Missing camera view")
        for relative in [*episode["videos"].values(), *episode["frames"].values(),
                         episode["actions"], episode["provenance"]]:
            require(local_path(root, relative).is_file(), "Referenced artifact missing")
        commands = local_path(root, episode["actions"])
        count = 0
        with commands.open() as f:
            for count, line in enumerate(f, 1):
                row = json.loads(line)
                require(row["control_step"] == count, "Noncontiguous control record")
                require(len(row["joint_command"]) == 14, "Wrong command width")
        require(count == episode["control_steps"], "Control count mismatch")
    with (root / "results/episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 50, "CSV episode count mismatch")
    for row, episode in zip(rows, episodes):
        require(row["episode_id"] == episode["episode_id"], "CSV identity mismatch")
        require(float(row["score_100"]) == episode["score_100"], "CSV score mismatch")
    reconstructions = read(root / "protocol/score_reconstruction.json")
    require({r["episode_id"] for r in reconstructions}
            == {e["episode_id"] for e in episodes
                if e["score_basis"] == "offline_replay_reconstruction"},
            "Reconstruction inventory mismatch")
    videos = read(root / "protocol/video_validation.json")
    require(len(videos) == 150
            and {v["path"] for v in videos} == {p for e in episodes for p in e["videos"].values()},
            "Video inventory mismatch")
    by_video = {v["path"]: v for v in videos}
    for e in episodes:
        for video in e["videos"].values():
            info = by_video[video]
            require(info["codec_name"] == "h264" and info["pix_fmt"] == "yuv420p"
                    and int(info["nb_frames"]) == e["control_steps"] + 1,
                    "Video encoding or frame count mismatch")
    with (root / "results/summary.csv").open(newline="") as stream:
        task_rows = list(csv.DictReader(stream))
    require(len(task_rows) == len(summary["tasks"]), "Summary CSV task count")
    for row, task in zip(task_rows, summary["tasks"]):
        require(row["task"] == task["task"]
                and int(row["successes"]) == task["successes"]
                and float(row["mean_score_100"]) == task["mean_score_100"],
                "Summary CSV metrics mismatch")
    # Embedded gallery data must agree with the machine-readable release.
    html = (root / "index.html").read_text()
    match = re.search(r'<script id="episode-data" type="application/json">(.*?)</script>',
                      html, re.DOTALL)
    require(match and json.loads(match[1]) == episodes, "Gallery data mismatch")
    extra = {}
    if (root / "protocol/context_bundle.json").exists():
        # Verification must leave the immutable download unchanged on repeat runs.
        import sys
        sys.dont_write_bytecode = True
        # Available both beside this source script and in the standalone bundle.
        from robodojo_release_evidence import verify_evidence
        extra = verify_evidence(root, episodes)
    return {**extra, "verified": True, "files": len(expected), "episodes": 50, "videos": 150,
            "success_rate": summary["success_rate"],
            "mean_score_100": summary["task_macro_mean_score_100"],
            "checks": ["checksums", "no_symlinks", "private_metadata_scan",
                       "native_success_eligibility", "score_recalculation",
                       "all_references_local", "control_counts", "csv", "gallery"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
