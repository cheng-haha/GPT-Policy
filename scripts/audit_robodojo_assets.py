"""Compare task-scoped local simulation assets with a fixed official HF revision."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from gpt_policy.simulation.robodojo.io import sha256, write_json


def object_scope(assets, kind, category, item):
    """Follow native layout_manager's group category and clutter asset family."""
    family = "Clutter" if item.get("type") == "cluttered" else kind
    return assets / f"Object/RoboDojo/{family}/{category}/{item['category_idx']:05d}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--task", default="classify_objects")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layouts", default="0,1,2,3,4")
    p.add_argument("--revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assets = a.root / "Assets"
    scopes = {
        assets / "Robots/x5",
        assets / "Sensor/Camera",
        assets / "Room/Simple_Room_nolight",
        assets / "Background/brown_photostudio_02_4k.hdr",
        assets / "Material/material_0122",
        assets / "Material/material_0564",
    }
    # Interactive tasks load a native demonstration arm and fixed trajectories.
    if a.task in {"imitate_sorting_sequence", "make_kong"}:
        scopes.add(assets / "Robots/franka")
    if a.task == "make_kong":
        scopes.add(assets / "Traj/RoboDojo/make_kong")
    missing = []
    for layout in map(int, a.layouts.split(",")):
        path = assets / f"Eval_Layout/RoboDojo/arx_x5/{a.seed}/{a.task}_{layout}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        scopes.add(path)
        if a.task == "imitate_sorting_sequence":
            scopes.add(
                assets / f"Traj/RoboDojo/imitate_sorting_sequence/{a.seed}/{layout}.pkl"
            )
        data = json.loads(path.read_text())
        for kind, groups in data.items():
            if not isinstance(groups, dict):
                continue
            for category, entries in groups.items():
                if not isinstance(entries, list):
                    continue
                for item in entries:
                    if not isinstance(item, dict) or "category_idx" not in item:
                        continue
                    scopes.add(object_scope(assets, kind, category, item))
                    visual = item.get("visual", {}).get("visual_usd_path", "")
                    prefix = "$Robodojo_ASSETS/"
                    if visual.startswith(prefix):
                        material = assets / visual[len(prefix) :]
                        scopes.add(material)
                        # Garment USDs reference companion texture directories.
                        # Audit the bounded material family, not the global Assets tree.
                        if kind == "Garment":
                            scopes.add(material.parent)
    files = set()
    for scope in scopes:
        if not scope.exists():
            missing.append(str(scope))
            continue
        entries = scope.rglob("*") if scope.is_dir() else [scope]
        for entry in entries:
            if entry.is_file() and not any(
                x.startswith(".") or x == "__pycache__"
                for x in entry.relative_to(assets).parts
            ):
                files.add(entry)
    names = sorted(str(f.relative_to(a.root)) for f in files)
    remote = {}
    for offset in range(0, len(names), 100):
        request = urllib.request.Request(
            f"https://huggingface.co/api/datasets/RoboDojo-Benchmark/RoboDojo/paths-info/{a.revision}",
            data=json.dumps(
                {"paths": names[offset : offset + 100], "expand": False}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            for row in json.load(response):
                remote[row["path"]] = row
        print(f"Metadata {min(offset + 100, len(names))}/{len(names)}", flush=True)
    rows = []
    for name in names:
        path = a.root / name
        digest = sha256(path)
        row = remote.get(name)
        if row and row.get("lfs"):
            expected = row["lfs"]["oid"]
            actual = digest
            algorithm = "sha256"
        elif row:
            data = path.read_bytes()
            actual = hashlib.sha1(
                f"blob {len(data)}\\0".replace("\\0", "\0").encode() + data
            ).hexdigest()
            expected = row["oid"]
            algorithm = "git_blob_sha1"
        else:
            actual = digest
            expected = None
            algorithm = "sha256"
        rows.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": digest,
                "official_digest": expected,
                "digest_algorithm": algorithm,
                "matches": expected == actual,
            }
        )
    report = {
        "official_repository": "RoboDojo-Benchmark/RoboDojo",
        "revision": a.revision,
        "task": a.task,
        "seed": a.seed,
        "layouts": a.layouts,
        "scopes": sorted(str(x) for x in scopes),
        "files": rows,
        "missing": missing,
        "matched": sum(x["matches"] for x in rows),
        "total": len(rows),
        "all_match": not missing and all(x["matches"] for x in rows),
        "scope_note": "Task object categories, X5 robot, camera, default room/background/table/ground materials and selected layouts; interactive Franka/trajectories and layout-selected garment material families where applicable. Not a global asset-tree certification.",
    }
    write_json(a.output, report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["scopes", "files"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
