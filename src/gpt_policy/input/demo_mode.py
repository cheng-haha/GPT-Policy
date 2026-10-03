"""Resolve automatic demonstration modes from source data, before task lookup."""

from pathlib import Path

from .mcap_demo import TOPICS, is_mcap_episode
from .recorded_demo import read_json, read_rows


def resolve_demo_mode(source: Path, mode: str | None = None) -> str:
    """Explicit modes win; auto includes any available recorded state/action."""
    if mode in ("video", "video+action"):
        return mode
    if mode not in (None, "auto"):
        raise ValueError("Demo mode must be auto, video or video+action")
    source = Path(source)
    if source.is_dir() and (source / "demo.json").is_file():
        source /= "demo.json"
    if source.suffix.lower() == ".json" and not source.is_dir():
        data = read_json(source)
        frames = data.get("keyframes") if isinstance(data, dict) else None
        if not isinstance(frames, list) or any(not isinstance(f, dict) for f in frames):
            raise ValueError(f"Expected demonstration keyframes in {source}")
        has_actions = any(f.get("state") or f.get("action") for f in frames)
    elif source.is_dir() and is_mcap_episode(source):
        # Inspect real messages, without decoding protobufs or opening cameras.
        from mcap.reader import make_reader
        with (source / "episode.mcap").open("rb") as stream:
            has_actions = next(make_reader(stream).iter_messages(topics=TOPICS), None) is not None
    elif source.is_dir() and (source / "states.jsonl").is_file():
        has_actions = any(row.get("state") for row in read_rows(source / "states.jsonl"))
    else:
        has_actions = False
    return "video+action" if has_actions else "video"
