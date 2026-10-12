"""Resolve a downloaded context release into a local simulation configuration."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(bundle, robodojo_root, simulation_python, codex_bin, output_root):
    bundle = Path(bundle).resolve()
    config = json.loads((bundle / "protocol/reproduction_config.json").read_text())
    contexts = json.loads((bundle / "protocol/context_bundle.json").read_text())
    for context in contexts.values():
        for relative, expected in [
            (context["input"], context["portable_input_sha256"]),
            *((x["path"], x["sha256"]) for x in context["images"]),
        ]:
            p = Path(relative)
            if p.is_absolute() or ".." in p.parts or sha256(bundle / p) != expected:
                raise ValueError("Context path/checksum mismatch: " + relative)
    for key in ("prepared_inputs", "asset_audits"):
        config[key] = {task: str((bundle / rel).resolve())
                       for task, rel in config[key].items()}
    binary = shutil.which(str(codex_bin)) or str(Path(codex_bin).expanduser().resolve())
    config.update(robodojo_root=str(Path(robodojo_root).expanduser().resolve()),
                  simulation_python=str(Path(simulation_python).expanduser().resolve()),
                  codex_bin=binary, output_root=str(Path(output_root).expanduser().resolve()),
                  simulation_environment={})
    if config["control_step_limit"] != "native" or config["live_image_window"] is not None:
        raise ValueError("Unexpected released evaluation protocol")
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--robodojo-root", type=Path, required=True)
    p.add_argument("--simulation-python", type=Path, required=True)
    p.add_argument("--codex-bin", default="codex")
    p.add_argument("--output-root", type=Path, default=Path("var/robodojo"))
    p.add_argument("--config", type=Path, default=Path("configs/robodojo.local.json"))
    a = p.parse_args()
    config = configure(a.bundle, a.robodojo_root, a.simulation_python,
                       a.codex_bin, a.output_root)
    a.config.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previously reviewed local runtime configuration.
    with a.config.open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    print(f"Verified 10 contexts / 480 images. Wrote {a.config}")


if __name__ == "__main__":
    main()
