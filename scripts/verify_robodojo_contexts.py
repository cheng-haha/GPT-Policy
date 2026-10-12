"""Verify a portable RoboDojo context bundle without simulator or model dependencies."""
import argparse
import hashlib
import json
import math
import re
from itertools import pairwise
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe bundle path")
    target = root / path
    require(target.resolve().is_relative_to(root.resolve()), "Path escapes bundle")
    require(target.is_file() and not target.is_symlink(), "Missing or linked artifact")
    return target


def read(path):
    return json.loads(path.read_text())


def verify(root):
    root = Path(root).resolve()
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        sha, relative = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", sha), "Invalid checksum")
        require(relative not in expected, "Duplicate checksum path")
        require(digest(local_path(root, relative)) == sha, "Checksum mismatch: " + relative)
        expected[relative] = sha
    # These are transport metadata created by Hugging Face, not dataset payload.
    ignored = {"SHA256SUMS", ".gitattributes"}
    actual = set()
    private = re.compile(
        r"/(?:mnt|root|Users|home)/|[A-Za-z]:\\Users\\|"
        r"\b(?:sk|hf|ghp)[_-][A-Za-z0-9]{16,}|"
        r'\"(?:api_key|access_token|refresh_token|account_id|thread_id)\"\s*:'
    )
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative.startswith((".cache/huggingface/", ".git/")):
            continue
        require(not path.is_symlink(), "Symlink in bundle")
        if not path.is_file() or relative in ignored:
            continue
        actual.add(relative)
        if path.suffix in {".json", ".jsonl", ".md", ".txt", ".py"}:
            require(not private.search(path.read_text()), "Private metadata: " + relative)
    require(actual == set(expected), "Unlisted or missing payload files")
    contexts = read(root / "protocol/context_bundle.json")
    require(len(contexts) == 10, "Expected ten task demonstrations")
    tasks, rows_total = {}, 0
    for context in contexts.values():
        task = context["task"]
        require(task not in tasks, "Duplicate task")
        p = local_path(root, context["input"])
        require(digest(p) == context["portable_input_sha256"], "Input checksum mismatch")
        agent = local_path(root, context["agent_context"])
        require(digest(agent) == context["agent_sha256"], "Public prompt checksum mismatch")
        data = read(p)
        require(set(data) == {"instruction", "model", "content"}, "Unexpected input schema")
        require(bool(data["instruction"].strip()), "Empty instruction")
        images = [x for x in data["content"] if isinstance(x, dict)]
        require(len(images) == len(context["images"]) == 48, "Expected 48 images")
        for part, reference in zip(images, context["images"]):
            rel = (p.parent.relative_to(root) / part["image"]).as_posix()
            image = local_path(root, rel)
            require(rel == reference["path"], "Image order/path mismatch")
            require(digest(image) == reference["sha256"], "Image checksum mismatch")
            require(image.read_bytes().startswith(b"\xff\xd8"), "Expected JPEG image")
        objects = []
        for part in data["content"]:
            if isinstance(part, str) and part.lstrip().startswith("{"):
                objects.append(json.loads(part))
        header = [x for x in objects if "source" in x and "title" in x]
        stages = [x for x in objects if "t_s" in x and "image_ids" in x]
        require(len(header) == 1 and len(stages) == 16, "Expected metadata and 16 stages")
        verified = header[0]["coordinate_frame"]["verified_source"]
        require(verified["site"] == "link6" and verified["frame"] == "environment_origin"
                and verified["position_unit"] == "m", "Unexpected source pose frame")
        columns = header[0]["action_sample_encoding"]["columns"]
        times = [s["t_s"] for s in stages]
        require(all(math.isfinite(t) for t in times)
                and all(a < b for a, b in pairwise(times)), "Invalid stage times")
        rows = 0
        for index, stage in enumerate(stages):
            require(stage["image_ids"] == {
                view: f"image_{3 * index + offset}"
                for offset, view in enumerate(("top", "left_wrist", "right_wrist"))
            }, "Three-view stage alignment mismatch")
            for arm in ("left", "right"):
                pose = stage["state"][arm]["endpose_xyz_xyzw_gripper"]
                require(len(pose) == 8 and all(math.isfinite(v) for v in pose),
                        "Invalid measured pose")
                require(abs(sum(v * v for v in pose[3:7]) - 1) < .002,
                        "Non-unit measured xyzw quaternion")
            action = stage.get("action")
            if action is None:
                require(index == 15, "Missing nonterminal action interval")
                continue
            require(action["next_keyframe"] == index + 1, "Incorrect next keyframe")
            samples = action["sample_rows"]
            require(bool(samples), "Empty action references")
            previous = None
            for row in samples:
                require(len(row) == len(columns), "Action row width mismatch")
                decoded = []
                for col, value in enumerate(row):
                    if value == "=":
                        require(previous is not None and col >= 2, "Invalid repeated field")
                        value = previous[col]
                    decoded.append(value)
                for value in decoded:
                    values = value if isinstance(value, list) else [value]
                    require(all(v is None or isinstance(v, (int, float))
                                and math.isfinite(v) for v in values),
                            "Non-numeric action reference")
                previous = decoded
            rows += len(samples)
        tasks[task] = {"source_episode": header[0]["source"]["episode_index"],
                       "timepoints": 16, "images": 48, "action_references": rows}
        rows_total += rows
    config = read(root / "protocol/reproduction_config.json")
    suite = read(root / "protocol/suite.json")["episodes"]
    require(len(suite) == 50, "Expected 50 configured cases")
    identities = {(s["task"], s["seed"], s["layout"]) for s in suite}
    require(len(identities) == 50, "Duplicate evaluation case")
    require(config["control_step_limit"] == "native"
            and config["max_decisions"] is None and config["live_image_window"] is None,
            "Unexpected released execution settings")
    for case in suite:
        task = case["task"]
        local_path(root, config["prepared_inputs"][task])
        audit = read(local_path(root, config["asset_audits"][task]))
        require(audit["task"] == task and audit["seed"] == case["seed"]
                and case["layout"] in [int(x) for x in audit["layouts"].split(",")],
                "Asset manifest does not cover case")
    return {"verified": True, "files": len(expected), "tasks": tasks,
            "demonstration_images": 480, "action_references": rows_total,
            "evaluation_cases": 50}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().root), indent=2))
